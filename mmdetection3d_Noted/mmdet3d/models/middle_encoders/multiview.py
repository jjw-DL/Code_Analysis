# Copyright (c) Gorilla-Lab. All rights reserved.
from mmcv.runner import auto_fp16
import numpy as np
from torch import nn

from mmdet3d.ops import spconv as spconv
from mmdet3d.ops import SparseBasicBlock
from mmcv.cnn import build_norm_layer
from ..builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module
class MultiViewBackbone(nn.Module):  # Small Depth
    def __init__(
        self, 
        num_input_features=64,
        sparse_shape = [100, 1280, 1280],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='SubMConv3d'),
        **kwargs
    ):
        super().__init__()

        self.sparse_shape = sparse_shape
        # input: # [100, 1280, 1280]
        self.middle_conv = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True),
            # stage1: [101, 1280, 1280] -> [50, 640, 640]
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            spconv.SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            # stage2: [50, 640, 640] -> [25, 320, 320]
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            spconv.SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ), 
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            # stage3:[25, 320, 320] -> 12, 160, 160]
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            spconv.SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            # stage4:保持[12, 160, 160]
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
        )
        self.bev_conv = spconv.SparseSequential(
            spconv.SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [12, 160, 160] -> [5, 160, 160]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )
        self.rv_conv = spconv.SparseSequential(
            # [12, 160, 160] -> [12, 160, 80] 同时通道从128降低到32
            spconv.SparseConv3d(
                128, 32, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False),  
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # [12, 160, 80] -> [12, 160, 40]
            spconv.SparseConv3d(
                32, 32, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, conv_cfg=conv_cfg),
            # [12, 160, 40] -> [12, 160, 20]
            spconv.SparseConv3d(
                32, 32, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False
            ),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_features, coors, batch_size):

        # input: (101, 1280, 1280）
        sparse_shape = np.array(self.sparse_shape) + [1, 0, 0] # （101, 1280, 1280）

        coors = coors.int() # (70563, 4)
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size) # 构建稀疏Tensor
        ret = self.middle_conv(ret) # (12, 160, 160)
        
        ret_bev = self.bev_conv(ret) # (5, 160, 160)
        ret_bev = ret_bev.dense() # (4, 128, 5, 160, 160)
        N, C, D, H, W = ret_bev.shape # 4, 128, 5, 160, 160
        ret_bev = ret_bev.view(N, C * D, H, W) # (4, 640, 160, 160) 在bev视角压缩

        ret_rv = self.rv_conv(ret) #  (12, 160, 20)
        ret_rv = ret_rv.dense() # (4, 32, 12, 160, 20)
        N, C, D, H, W = ret_rv.shape # 4, 32, 12, 160, 20
        ret_rv = ret_rv.permute(0, 1, 4, 3, 2) # (4, 32, 20, 160, 12)
        ret_rv = ret_rv.contiguous().view(N, C * W, H, D) # (4, 640, 160, 12)
        return ret_bev, ret_rv