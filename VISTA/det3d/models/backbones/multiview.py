# Copyright (c) Gorilla-Lab. All rights reserved.

import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
import spconv.pytorch as spconv
from spconv.pytorch import SparseConv3d, SubMConv3d

from det3d.models.utils import Empty, change_default_args
from det3d.torchie.cnn import constant_init, kaiming_init
from det3d.torchie.trainer import load_checkpoint

from .scn import SparseBasicBlock
from .. import builder
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


@BACKBONES.register_module
class MultiViewBackbone(nn.Module):  # Small Depth
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="MultiViewBackbone", **kwargs
    ):
        super().__init__()
        self.name = name


        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-5, momentum=0.1)
        # input: # [1280, 1280, 101]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1280, 1280, 101] -> [640, 640, 50]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [640, 640, 50] -> [320, 320, 25]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [320, 320, 25] -> [160, 160, 12]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )
        self.bev_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [160, 160, 12] -> [160, 160, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )
        self.rv_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 32, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False
            ),  # [160, 160, 12] -> [160, 160, 5] 同时通道从128降低到32
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res4"),
            SparseConv3d(
                32, 32, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False
            ),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res5"),
            SparseConv3d(
                32, 32, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False
            ),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0] # （101, 1280, 1280）
        #print('======', sparse_shape)
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