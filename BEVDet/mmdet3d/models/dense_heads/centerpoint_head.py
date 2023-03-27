# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply


@HEADS.register_module()
class SeparateHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels, # 64
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SeparateHead, self).__init__(init_cfg=init_cfg)
        # heads: dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)，headtma=(x, 2))
        self.heads = heads
        self.init_bias = init_bias # -2.19
        # 逐项取出
        for head in self.heads:
            classes, num_conv = self.heads[head] # num_conv一直为2

            conv_layers = []
            c_in = in_channels # 64
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in, # 64
                        head_conv, # 64
                        kernel_size=final_kernel, # 1
                        stride=1,
                        padding=final_kernel // 2, # 0
                        bias=bias, # 'auto'
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg)) # 将conv，norm和act绑定到一个模块
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,  # 64
                    classes, # 2 (classes针对不同fild是不同的)
                    kernel_size=final_kernel, # 1
                    stride=1,
                    padding=final_kernel // 2, # 0
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers) # （64, 64）--> (64, 2)

            self.__setattr__(head, conv_layers) # self.head_name = conv_layers[i]

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x) # 将经过共享卷积后的特征图，分别送入6个head得到6个特征图

        return ret_dict # 返回的是字典值，包含head的6个信息（如注释所示）


@HEADS.register_module()
class DCNSeparateHead(BaseModule):
    r"""DCNSeparateHead for CenterHead.

    .. code-block:: none
            /-----> DCN for heatmap task -----> heatmap task.
    feature
            \-----> DCN for regression tasks -----> regression tasks

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        dcn_config (dict): Config of dcn layer.
        num_cls (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """  # noqa: W605

    def __init__(self,
                 in_channels, # 64
                 num_cls, # 1, 2, 2, 1, 2, 2
                 heads, # heads: dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)，headtma=(x, 2))
                 dcn_config,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(DCNSeparateHead, self).__init__(init_cfg=init_cfg)
        if 'heatmap' in heads:
            heads.pop('heatmap')
        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = build_conv_layer(dcn_config) # 构建分类DCN

        self.feature_adapt_reg = build_conv_layer(dcn_config) # 构建回归DCN

        # heatmap prediction head
        cls_head = [
            ConvModule(
                in_channels, # 64
                head_conv, # 64
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                bias=bias,
                norm_cfg=norm_cfg),
            build_conv_layer(
                conv_cfg,
                head_conv, # 64
                num_cls, # 2
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
        ]
        self.cls_head = nn.Sequential(*cls_head)
        self.init_bias = init_bias
        # other regression target
        self.task_head = SeparateHead(
            in_channels,
            heads, # 此时的head中没有heatmap: dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
            head_conv=head_conv,
            final_kernel=final_kernel,
            bias=bias)
        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        self.cls_head[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for DCNSepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        center_feat = self.feature_adapt_cls(x) # 特征图经过分类DCN生成分类分支特征图
        reg_feat = self.feature_adapt_reg(x) # 特征图经过回归DCN生成回归分支特征图 

        cls_score = self.cls_head(center_feat) # 单独处理分类分支
        ret = self.task_head(reg_feat) # 单独处理回归分支
        ret['heatmap'] = cls_score

        return ret


@HEADS.register_module()
class CenterHead(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128], # 256
                 tasks=None, # 6个task list[dict]
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None, # 'CenterPointBBoxCoder'
                 common_heads=dict(), # dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None,
                 task_specific=True,
                 loss_prefix=''):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterHead, self).__init__(init_cfg=init_cfg)

        num_classes = [len(t['class_names']) for t in tasks] # [1,2,2,1,2,2]
        self.class_names = [t['class_names'] for t in tasks]
        # 10类:[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels # 512
        self.num_classes = num_classes # [1,2,2,1,2,2]
        self.norm_bbox = norm_bbox # True

        self.loss_cls = build_loss(loss_cls) # GaussianFocalLoss:mmdet.models.losses.gaussina_focal_loss
        self.loss_bbox = build_loss(loss_bbox) # L1 Loss: mmdet.models.losses.smooth_l1_loss
        self.bbox_coder = build_bbox_coder(bbox_coder) # CenterPointBBoxCoder:mmdet3d.core.bbox.coders.centerpoint_bbox_coders
        self.num_anchor_per_locs = [n for n in num_classes] # [1,2,2,1,2,2]
        self.fp16_enabled = False

        # a shared convolution 将512变为64
        self.shared_conv = ConvModule(
            in_channels, # 512
            share_conv_channel, # 64
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()
        # 逐task添加head，每个task_head中又有6个head
        for num_cls in num_classes:
            # common_heads:dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
            heads = copy.deepcopy(common_heads)
            # num_classes:[1,2,2,1,2,2]  num_heatmap_convs:2
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs))) # 加入heatmap参数，各head主要是heatmap不同
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls) # 更新separate_head share_conv_channe:64
            # mmdet3d.models.dense_heads.conterpoint_head.SeparateHead或mmdet3d.models.dense_heads.conterpoint_head.DCNSeparateHead
            self.task_heads.append(builder.build_head(separate_head))


        self.task_specific = task_specific # True
        self.loss_prefix = loss_prefix # ''

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x) # （2, 64, 128, 128）
        # cfg中有6个不同的task对应init中的6个Head:{reg, height, dim, rot, vel, headtmap}，主要是heatmap不同（类别）
        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats) # 调用forward_single函数（对于centerpoint实际只有一个尺度）

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2) # eg:10
        # (2, 500) --> (2, 500, 1) --> (2, 500, 10)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind) # （2, 16384, 10）--> (2, 500, 10) gather的用法是在对应的维度替换索引值，取出对应的元素值
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat # (2, 500, 10)

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        # 这里返回为Tuple(List[List[torch.Tensor]]), 最外层的List表示不同帧点云，内部List表示不同的task，最内层的torch.Tensor为该task内所有物体的信息
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d) # 这里的赋值自动将Tuple拆解
        # Transpose heatmaps 6个list对应6个task
        heatmaps = list(map(list, zip(*heatmaps))) # 解压为6个list, 一个batch内相同的字段放到一起，组成一个新的list eg:batch为2则6个list中的每个元素为2个list
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps] # 在每个list内进行stack --> (2, 1, 128, 128) 或 (2, 2, 128, 128)
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes] # 每个元素是(2, 500, 10)
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds] # (2, 500)
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks] # (2, 500)
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device # cuda:0
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device) # gravity_center是box的中心，LiDARInstance3DBoxes的center在box的底部(z=0)-->[N, 9]
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg'] # 500 × 1 = 500
        grid_size = torch.tensor(self.train_cfg['grid_size']) # [1024, 1024, 40]
        pc_range = torch.tensor(self.train_cfg['point_cloud_range']) # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size = torch.tensor(self.train_cfg['voxel_size']) # [0.1, 0.1, 0.2]

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        # ---------------------------------------------
        # 1.找到6个task类别在gt_labels_3d中的位置生成6个list
        # ---------------------------------------------
        task_masks = []
        flag = 0
        # 逐个类别生成mask
        # self.class_names:[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']]
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag) # gt_labels_3d:(N,) 这里之所以加flag是因为class_name.index(i)是局部索引
                for i in class_name # 生成为List(Tuple)
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        # ---------------------------------------------
        # 2.根据task_masks逐task处理，添加对应的bbox和class
        # ---------------------------------------------
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            # mask是一个list，内部元素为tuple，表示该class的索引
            for m in mask:
                task_box.append(gt_bboxes_3d[m]) # 根据index提取box
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2) # 根据index提取label，减去flag2后表示的是在该task中的label只有1和2，0表示背景
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask) # 1或2
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], [] # 初始化结果list

        # ---------------------------------------------
        # 3.逐task，逐个物体处理，绘制heatmap
        # ---------------------------------------------
        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0])) # eg:（1, 128, 128）初始化heatmap

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32) # (500, 10)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64) # (500,)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8) # (500,)

            num_objs = min(task_boxes[idx].shape[0], max_objs) # 当前task有几个物体 eg:6

            # 逐个物体处理
            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1 # 获取class id，这里是局部id，一般为0或1

                width = task_boxes[idx][k][3] # 获取bbox的宽
                length = task_boxes[idx][k][4] # 获取bbox的长
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor'] # 计算特征图上的宽：先除voxle_size计算voxle大小，再除下采样因子
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    # 根据box的长和宽以及overlap计算高斯半径
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap']) # gaussian_overlap：0.1 比较小的原因是在bev视角下重叠小
                    radius = max(self.train_cfg['min_radius'], int(radius)) # min_radius=2 限定最小值s

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2] # 获取box的中心坐标x y和z

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor'] # 计算box的x坐标在特征图上的位置-->小数 eg:[66.566, 85.593]
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device) # 组合center，小数
                    center_int = center.to(torch.int32) # 整数 --> eg:[66, 85]

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    # 获取gaussian mask的heatmap
                    # 参数为初始化的heatmap, 中心点坐标和高斯半径
                    draw_gaussian(heatmap[cls_id], center_int, radius) # 在该类对应位置生成gaussian map的并赋值给heatmap

                    new_idx = k # 表示该类别的第几个物体
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    # 特征图展开成一维向量的索引赋值给ind，记录该类别的物体在特征图中的位置 eg:ind[0] = 2122
                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    vx, vy = task_boxes[idx][k][7:] # 获取速度
                    rot = task_boxes[idx][k][6] # 获取旋转 eg:0.2759
                    box_dim = task_boxes[idx][k][3:6] # 获取长宽高 eg:[1.9308, 5.0753, 1.6253]
                    if self.norm_bbox:
                        box_dim = box_dim.log() # 对长宽高取log
                    # 重新组装新的anno_box
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device), # 偏移量, center是小数，x和y是整数
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0), # 对旋转角度取sin
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0)
                    ]) # (10,）

            # 逐个task加入
            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask) # 表示各task的有效物体
            inds.append(ind) # 表示有效物体的中心点在特征图展开向量中的索引
        return heatmaps, anno_boxes, inds, masks # 都是包含6个Tensor的list

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        # 生成6个task对应的heatmap, anno_boxes，有效位置索引和有效位置mask
        # heatmap的每个item shape: (2, 1, 128, 128) 或 (2, 2, 128, 128)
        # anno_boxes的每个item shape: (2, 500, 10)
        # inds的每个item shape: (2, 500) 表示有效物体的中心点在特征图展开向量(一维向量)中的索引
        # masks的每个item shape: (2, 500) 表示有效物体mask
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict() # 初始化loss的dict

        # 逐个task计算loss
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap']) # 先计算预测heatmap的sigmoid,这里取0的原因是preds_dict是List[dict] (8, 1, 128, 128)
            num_pos = heatmaps[task_id].eq(1).float().sum().item() # 取出对应的heatmap，先变为bool，在变float，最后求和，计算有多少个物体 eg:8
            # 1. 计算类别损失: GaussianFocalLoss
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'], # (2, 1, 128, 128)
                heatmaps[task_id], # (2, 1, 128, 128)
                avg_factor=max(num_pos, 1)) # --> eg:1281.2971
            
            # 2. 计算回归损失 L1 Loss
            target_box = anno_boxes[task_id] # 获取该task的target box eg:(2, 500, 10)
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1) # (2, 10, 128, 128) 10 = 2 + 1 + 3 + 2 + 2

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id] # 获取有效位置索引 (2, 500)
            num = masks[task_id].float().sum() # 获取有效物体个数 eg:8
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous() # 获取该task的预测值（2, 128, 128, 10）
            # --------------------------------------------
            # 这两行实现了target assign
            # --------------------------------------------
            pred = pred.view(pred.size(0), -1, pred.size(3)) # 将中间两维合并 --> (2, 16384, 10)
            # target提取对应位置的预测值
            pred = self._gather_feat(pred, ind) # (2, 500, 10)

            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float() # (2, 500) --> (2, 500, 1) --> (2, 500, 10)
            isnotnan = (~torch.isnan(target_box)).float() # (2, 500, 10) 
            mask *= isnotnan # 忽略nan

            code_weights = self.train_cfg.get('code_weights', None) # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))
            bbox_weights = mask * mask.new_tensor(code_weights) # (1, 500, 10)  与mask相乘，使得没有assign的reg损失别忽略
            
            if self.task_specific:
                name_list=['xy','z','whl','yaw','vel']
                clip_index = [0,2,3,6,8,10]
                for reg_task_id in range(len(name_list)):
                    pred_tmp = pred[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    target_box_tmp = target_box[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    bbox_weights_tmp = bbox_weights[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    loss_bbox_tmp = self.loss_bbox(
                        pred_tmp, target_box_tmp, bbox_weights_tmp, avg_factor=(num + 1e-4))
                    loss_dict[f'%stask{task_id}.loss_%s'%(self.loss_prefix,name_list[reg_task_id])] = loss_bbox_tmp
            else:
                # 计算回归损失
                loss_bbox = self.loss_bbox(
                    pred, target_box, bbox_weights, avg_factor=(num + 1e-4)) # eg：1.9046 target_box没有assign的位置为0
                # 记录loss
                loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            loss_dict[f'%stask{task_id}.loss_heatmap'%(self.loss_prefix)] = loss_heatmap
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        # ----------------------------------
        # 1.逐个task处理
        # ----------------------------------
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id] # self.num_classes:[1, 2, 2, 1, 2, 2]
            batch_size = preds_dict[0]['heatmap'].shape[0] # 计算batch size
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid() # 对heatmap取sigmoid，将类别预测结果限制在0-1之间 --> (1, 1, 128, 128)

            batch_reg = preds_dict[0]['reg'] # 取出回归预测--> (1, 2, 128, 128)
            batch_hei = preds_dict[0]['height'] # 取出高度预测--> (1, 1, 128, 128)

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim']) # 对长宽高取exp --> (1, 3, 128, 128)
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1) # 旋转预测sin --> (1, 1, 128, 128)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1) # 旋转预测cos --> (1, 1, 128, 128)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'] # 速度预测 --> (1, 2, 128, 128)
            else:
                batch_vel = None
            # 内部进行了逐帧处理
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id) # List[Dict{}] 每个dict记录了一帧中的bboxes，scores和lables, 比如有2个dict则表示2帧数据
            # assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp] # 将batch中相同的字段组合，每个tensor代表一帧 --> List[tensor] eg:[(8, 9), ...]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            # nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
            nms_type = self.test_cfg.get('nms_type')
            # 根据task id决定nms策略
            if isinstance(nms_type,list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                ret_task = []
                # 在task内逐帧处理
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes'] # 取出该帧的预测结果tensor:(8, 9)
                    scores = temp[i]['scores'] # (8,)
                    labels = temp[i]['labels'] # (8,)
                    centers = boxes3d[:, [0, 1]] # (8, 2)
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1) # (8, 3)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(), # (8, 3)
                            self.test_cfg['min_radius'][task_id], # [4, 12, 10, 1, 0.85, 0.175]
                            post_max_size=self.test_cfg['post_max_size']), # 83
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels) # 将该帧的预测结果整合
                    ret_task.append(ret) # List[Dict{}] 每个dict表示一帧
                rets.append(ret_task)
            else:
                # 实际上这里就是根据decode的结果进行NMS
                # List[List[Dict]] 内层每个list代表一帧, 每个List包含6个Dict表示6个task的预测结果'bbox','scores','labels'
                rets.append(
                    self.get_task_detections(num_class_with_bg, # eg:1或2
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas, task_id))

        # Merge branches results
        num_samples = len(rets[0]) # eg:1

        ret_list = []
        # ----------------------------------------
        # 2.逐帧处理，将不同task处理合并到一起
        # ----------------------------------------
        for i in range(num_samples):
            for k in rets[0][i].keys(): # rets[0][i].keys():['bbox','scores','labels']
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets]) # 将该帧内的6个task预测的bbox进行拼接 eg:(89, 9)
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5 # 将bbox的中心高度移到地面
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size) # 将预测的bbox封装为LidarInstance3DBBoxeds
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets]) # 将该帧内的6个task预测的score进行拼接 eg:(89, )
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag # 从局部标签恢复全局标签 task,batch_index,item
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets]) # 将恢复全局标签的帧组合
            ret_list.append([bboxes, scores, labels]) # 组合该帧6个task的全部预测结果，并加入ret_list
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas, task_id):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the \
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the \
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the \
                    shape of [N].
        """
        predictions_dicts = [] # 初始化预测结果字典
        post_center_range = self.test_cfg['post_center_limit_range'] # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device) # 构造tensor

        # 逐帧处理
        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]]
            nms_rescale_factor = self.test_cfg.get('nms_rescale_factor', [1.0 for _ in range(len(self.task_heads))])[task_id]
            if isinstance(nms_rescale_factor,list):
                for cid in range(len(nms_rescale_factor)):
                    box_preds[cls_labels==cid, 3:6] = box_preds[cls_labels==cid, 3:6] * nms_rescale_factor[cid]
            else:
                box_preds[:,3:6] = box_preds[:,3:6] * nms_rescale_factor

            # Apply NMS in birdeye view

            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']], # 0.1
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh # 根据阈值生成mask
                top_scores = top_scores.masked_select(top_scores_keep) # 根据mask过滤score

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep] # 过滤bbox
                    top_labels = top_labels[top_scores_keep] # 过滤label

                # img_metas[i]['box_type_3d']是LIDARInstance3DBoxes
                # box_preds[:, :] --> (8, 9)
                # self.bbox_coder.code_size:9
                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev) # 将bev视角的bbox从中心点+宽高+yaw --> 左上角+右下角+yaw (8, 9)
                # the nms in 3d detection just remove overlap boxes.
                
                # nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
                if isinstance(self.test_cfg['nms_thr'],list):
                    nms_thresh = self.test_cfg['nms_thr'][task_id]
                else:
                    nms_thresh = self.test_cfg['nms_thr']
                
                selected = nms_gpu(
                    boxes_for_nms,
                    top_scores,
                    thresh=nms_thresh, # 0.2
                    pre_maxsize=self.test_cfg['pre_max_size'], # 1000
                    post_max_size=self.test_cfg['post_max_size']) # 83 --> eg:[0, 4]
            else:
                selected = []

            if isinstance(nms_rescale_factor, list):
                for cid in range(len(nms_rescale_factor)):
                    box_preds[cls_labels == cid, 3:6] = box_preds[cls_labels == cid, 3:6] / nms_rescale_factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] / nms_rescale_factor

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                # 根据距离范围对bbox进行过滤
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                # 如果selecte没有值，则构造0值并返回
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
