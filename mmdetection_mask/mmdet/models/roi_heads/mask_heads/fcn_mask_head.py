# Copyright (c) OpenMMLab. All rights reserved.
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@HEADS.register_module()
class FCNMaskHead(BaseModule):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=dict(type='Conv'),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(FCNMaskHead, self).__init__(init_cfg)
        self.upsample_cfg = upsample_cfg.copy() # dict(type='deconv', scale_factor=2)
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs # 4
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size) # (14, 14)
        self.in_channels = in_channels # 256
        self.conv_kernel_size = conv_kernel_size # 3
        self.conv_out_channels = conv_out_channels # 256
        self.upsample_method = self.upsample_cfg.get('type') # deconv
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None) # 2
        self.num_classes = num_classes # 80
        self.class_agnostic = class_agnostic # False
        self.conv_cfg = conv_cfg # None
        self.norm_cfg = norm_cfg # None
        self.predictor_cfg = predictor_cfg # dict(type='Conv')
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask) # 'CrossEntropyLoss'

        self.convs = ModuleList() # 构建卷积模块列表
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels, # 256
                    self.conv_out_channels, # 256
                    self.conv_kernel_size, # 3
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels) # 256
        upsample_cfg_ = self.upsample_cfg.copy()  # dict(type='deconv', scale_factor=2)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels, # 256
                out_channels=self.conv_out_channels, # 256
                kernel_size=self.scale_factor, # 2
                stride=self.scale_factor) # 2
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes # 80
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels) # 256
        self.conv_logits = build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, out_channels, 1) # Conv, 256, 80, 1
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        super(FCNMaskHead, self).init_weights()
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        # 1.先经过4层256的3x3的卷积，并且padding保持特征图尺寸
        for conv in self.convs:
            x = conv(x) # (N, 256, 14, 14)
        # 2.进行上采样2倍，恢复到原图尺寸
        if self.upsample is not None:
            x = self.upsample(x) # (N, 256, 28, 28)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        # 3.将256转为80的分类-->(N, 80, 28, 28)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results] # 从sampling_results中提取正例的bbox组成list
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ] # 从sampling_results中提取正例的gt索引组成list
        # 计算预测bbox对应的target mask --> (N, 28, 28) 通过roi_align去mask中对齐和截取对应的gt mask
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                """
                mask_pred:(N, 28, 28, 80)
                mask_targets: (N, 28, 28)
                labels: (N,)
                """
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss


    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(ndarray | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.

        Example:
            >>> import mmcv
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> det_bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> det_labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = mmcv.Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = torch.FloatTensor((1, 1))
            >>> rescale = False
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self.get_seg_masks(
            >>>     mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape,
            >>>     scale_factor, rescale
            >>> )
            >>> assert len(encoded_masks) == C
            >>> assert sum(list(map(len, encoded_masks))) == N
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid() # 对预测的分数取sigmoid限制在0-1之间 (N, 80, 28, 28).
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device # 获取mask的device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes 逐个类别生成
        bboxes = det_bboxes[:, :4] # 提取bbox的前4维 (N, 4)
        labels = det_labels # 提取bbox的label (N,)

        # In most cases, scale_factor should have been
        # converted to Tensor when rescale the bbox
        if not isinstance(scale_factor, torch.Tensor):
            if isinstance(scale_factor, float):
                scale_factor = np.array([scale_factor] * 4)
                warn('Scale_factor should be a Tensor or ndarray '
                     'with shape (4,), float would be deprecated. ')
            assert isinstance(scale_factor, np.ndarray)
            scale_factor = torch.Tensor(scale_factor)

        if rescale:
            img_h, img_w = ori_shape[:2] # 获取原始图像大小 
            bboxes = bboxes / scale_factor.to(bboxes) # 将bbox除以缩放尺度
        else:
            w_scale, h_scale = scale_factor[0], scale_factor[1] # 获取宽高的缩放尺度
            img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32) # 将图片高乘以缩放尺度
            img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

        N = len(mask_pred) # 计算mask的数量
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                        GPU_MEM_LIMIT)) # 向上取整，为1
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks) # ([0,1,2,3...16],) 这里是一个tuple

        threshold = rcnn_test_cfg.mask_thr_binary # 0.5
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8) # 初始化N个图片矩阵，初始值为False

        if not self.class_agnostic: # 如果不忽略类别
            mask_pred = mask_pred[range(N), labels][:, None] # (N, 1, 28, 28)

        # 逐个块进行处理
        for inds in chunks: # 这里的inds是([0,1,2,3...16],)
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds], # (17,1,28,28)
                bboxes[inds], # (17 ,4)
                img_h, # 900
                img_w, # 1600
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool) # (17, 900, 1600)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk # 进行赋值

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy()) # 按照类别整理
        return cls_segms


    def onnx_export(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, **kwargs):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor): shape (n, #class, h, w).
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)

        Returns:
            Tensor: a mask of shape (N, img_h, img_w).
        """

        mask_pred = mask_pred.sigmoid()
        bboxes = det_bboxes[:, :4]
        labels = det_labels
        # No need to consider rescale and scale_factor while exporting to ONNX
        img_h, img_w = ori_shape[:2]
        threshold = rcnn_test_cfg.mask_thr_binary
        if not self.class_agnostic:
            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, labels][:, None]
        masks, _ = _do_paste_mask(
            mask_pred, bboxes, img_h, img_w, skip_empty=False)
        if threshold >= 0:
            # should convert to float to avoid problems in TRT
            masks = (masks >= threshold).to(dtype=torch.float)
        return masks


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device # 获取masks的device
    if skip_empty:
        # 对bbox的坐标进行截断
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    # 获取bbox的点坐标
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0] # 获取mask的数量 17

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5 # (0.5,...,899.5)
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5 # (0.5,...,1599.5)
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1 # (N, h)-->(17, 900)
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1 # (N, w)-->(17, 1600)
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1)) # (17, 900, 1600)
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1)) # (17, 900, 1600)
    grid = torch.stack([gx, gy], dim=3) # (17, 900, 1600, 2)

    """
    masks: (17, 1, 28, 28)
    grid: (17, 1, 900, 1600)
    """
    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False) # (17, 1, 900, 1600)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else: 
        return img_masks[:, 0], () # (17, 900, 1600)
