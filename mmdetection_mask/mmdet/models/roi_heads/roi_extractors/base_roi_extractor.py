# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv import ops
from mmcv.runner import BaseModule


class BaseRoIExtractor(BaseModule, metaclass=ABCMeta):
    """Base class for RoI extractor.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 init_cfg=None):
        super(BaseRoIExtractor, self).__init__(init_cfg)
        # roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides) # 构建roi layer
        self.out_channels = out_channels # 256
        self.featmap_strides = featmap_strides # [4, 8, 16, 32]
        self.fp16_enabled = False

    @property
    def num_inputs(self):
        """int: Number of input feature maps."""
        return len(self.featmap_strides) # 4

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.
            针对单个尺度的特征图
        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (List[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy() # 复制配置
        layer_type = cfg.pop('type') # 将type弹出，并赋值
        assert hasattr(ops, layer_type) # 确定ops中存在该RoI模块
        layer_cls = getattr(ops, layer_type) # 获取该ROI模块类
        # 根据featmap_strides初始化不同尺度的ROI特征提取层
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """
        # roi的坐标为x1y1x2y2
        cx = (rois[:, 1] + rois[:, 3]) * 0.5 # 计算roi的中心x坐标
        cy = (rois[:, 2] + rois[:, 4]) * 0.5 # 计算roi的中心y坐标
        w = rois[:, 3] - rois[:, 1] # 计算roi的宽
        h = rois[:, 4] - rois[:, 2] # 计算roi的长
        new_w = w * scale_factor # 乘缩放尺度
        new_h = h * scale_factor
        # 计算缩放后的x1y1x2y2
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        # 将batch id叠加在坐标前
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @abstractmethod
    def forward(self, feats, rois, roi_scale_factor=None):
        pass
