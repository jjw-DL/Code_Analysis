# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer, # dict(type='RoIAlign', output_size=7, sampling_ratio=0)
                 out_channels, # 256
                 featmap_strides, # [4, 8, 16, 32]
                 finest_scale=56,
                 init_cfg=None):
        super(SingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides, init_cfg)
        self.finest_scale = finest_scale # 56

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.
        通过roi的大小映射到对应尺度的特征图上

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])) # roi面积的开方
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6)) # 计算roi所属特征层
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long() # 对所属level进行截断
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function.
            feats(list[Tensor]): list of multi-level img features, each item 
                                 with shape NCHW 多个尺度的特征图
            rois: multi image roi which shape is (N, 5) 多张图片的ROI
        """
        # 1.初始化输出特征
        out_size = self.roi_layers[0].output_size # 7
        num_levels = len(feats) # 4
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1]) # (-1, 256 * 7 * 7)
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size) # (N, 256, 7, 7)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        # 正对只有一个特征层的情况单独处理
        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        # 2.将roi映射到对应的特征图尺度上
        target_lvls = self.map_roi_levels(rois, num_levels) 

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # 3.逐层进行roi特征提取
        for i in range(num_levels):
            mask = target_lvls == i # 根据索引提取该尺度特征图的roi
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1)
                # select target level rois and reset the rest rois to zero.
                rois_i = rois.clone().detach()
                rois_i *= mask
                mask_exp = mask.expand(*expand_dims).reshape(roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois_i)
                roi_feats_t *= mask_exp
                roi_feats += roi_feats_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1) # 获取roi中的非0索引
            if inds.numel() > 0:
                rois_ = rois[inds] # 提出该层的roi，5维的第一维包含batch id用于在特征图中指定batch
                roi_feats_t = self.roi_layers[i](feats[i], rois_) # ROI Align进行特征提取 --> (M, 256, 7, 7)
                roi_feats[inds] = roi_feats_t # 将提取的特征填入对应位置
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats