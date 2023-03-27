# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torch import nn as nn

from mmcv.runner import auto_fp16
from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from ..builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class SparseEncoderMultiView(nn.Module):
    r"""Sparse encoder for VISTA

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels, # 64
                 sparse_shape, # [100, 1280, 1280]
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
                 block_type='basicblock'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape # [100, 1280, 1280]
        self.in_channels = in_channels # 64
        self.base_channels = base_channels # 16
        self.output_channels = output_channels # 128
        self.encoder_channels = encoder_channels # ((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128))
        self.encoder_paddings = encoder_paddings # ((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0))
        self.fp16_enabled = False

        # Spconv init all weight on its own
        self.conv_input = make_sparse_convmodule(
            in_channels, # 64
            self.base_channels, # 16
            3,
            norm_cfg=norm_cfg, # dict(type='BN1d', eps=1e-3, momentum=0.01)
            padding=1, # # padding为1维持了shape的大小
            indice_key='subm1',
            conv_type='SubMConv3d')
        
        #             (16, 16, 32),      (32, 32, 64),     (64, 64, 128),     (128, 128)
        #              (0, 0, 1),         (0, 0, 1),     (0, 0, [0, 1, 1]),     (0, 0)
        # 中间stage:[101, 1280, 1280] -> [50, 640, 640] -> [25, 320, 320] -> [12, 160, 160]
        #                   16       ->       32       ->      64       ->      128
        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type) # ‘basicblock’

        # [12, 160, 160] -> [5, 160, 160]
        self.bev_conv = make_sparse_convmodule(
            encoder_out_channels, # 128
            self.output_channels, # 128
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1), # z在前
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down_bev',
            conv_type='SparseConv3d')

        self.rv_conv = spconv.SparseSequential(
            # [12, 160, 160] -> [12, 160, 80] 同时通道从128降低到32
            make_sparse_convmodule(
                encoder_out_channels, # 128
                32,
                kernel_size=(1, 1, 3),
                stride=(1, 1, 2),
                norm_cfg=norm_cfg,
                padding=(0, 0, 1),
                indice_key='spconv_down_rv_1',
                conv_type='SparseConv3d'),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, 
                             conv_cfg=dict(type='SubMConv3d')),
            # [12, 160, 80] -> [12, 160, 40]
            make_sparse_convmodule(
                32, 
                32,
                kernel_size=(1, 1, 3),
                stride=(1, 1, 2), 
                norm_cfg=norm_cfg,
                padding=(0, 0, 1),
                indice_key='spconv_down_rv_2',
                conv_type='SparseConv3d'),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg,
                             conv_cfg=dict(type='SubMConv3d')),
            # [12, 160, 40] -> [12, 160, 20]
            make_sparse_convmodule(
                32, 
                32,
                kernel_size=(1, 1, 3),
                stride=(1, 1, 2),
                norm_cfg=norm_cfg,
                padding=(0, 0, 1),
                indice_key='spconv_down_rv_3',
                conv_type='SparseConv3d'),
        )

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        sparse_shape = np.array(self.sparse_shape) + [1, 0, 0] # （101, 1280, 1280）
        coors = coors.int()
        # 对输入voxle特征进行稀疏编码，转化为稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        ret_bev = self.bev_conv(x)
        ret_bev = ret_bev.dense() # (4, 128, 5, 160, 160)
        N, C, D, H, W = ret_bev.shape # 4, 128, 5, 160, 160
        ret_bev = ret_bev.view(N, C * D, H, W) # (4, 640, 160, 160) 在bev视角压缩

        ret_rv = self.rv_conv(x) #  (12, 160, 20)
        ret_rv = ret_rv.dense() # (4, 32, 12, 160, 20)
        N, C, D, H, W = ret_rv.shape # 4, 32, 12, 160, 20
        ret_rv = ret_rv.permute(0, 1, 4, 3, 2) # (4, 32, 20, 160, 12)
        ret_rv = ret_rv.contiguous().view(N, C * W, H, D) # (4, 640, 160, 12)

        return ret_bev, ret_rv

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = spconv.SparseSequential() # 初始化稀疏卷积序列

        # 逐block构建
        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            # 逐层构建
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j] # 获取padding值
                # 如果该block是conv_module
                # each stage started with a spconv layer except the first stage
                # 每个stage的第一层都使用spconv layer，除了第一个stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg, # dict(type='BN1d', eps=1e-3, momentum=0.01)
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                # 如果该block是basicblock
                elif block_type == 'basicblock':
                    # 每个stage的最后一层都使用spconv layer，除了最后一个block
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels, # 一般通道数会x2
                                3,
                                norm_cfg=norm_cfg, # dict(type='BN1d', eps=1e-3, momentum=0.01)
                                stride=2, # 这里stride为2，会使特征图尺寸/2
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        # 如果该block是basicblock且不满足上面的条件则进入该判断
                        # Sparse basic block implemented with submanifold sparse convolution
                        # 继承自resnet的BasicBlock和spconv的SparseModule
                        # 采用残差结构，这里不改变通道数和输入特征图尺寸，padding在BasicBlock中采用默认值
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg, # dict(type='BN1d', eps=1e-3, momentum=0.01)
                                conv_cfg=conv_cfg)) # dict(type='SubMConv3d')
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d')) # 非残差结构
                in_channels = out_channels # 更新输入通道
            stage_name = f'encoder_layer{i + 1}' # 更新block(stage)名
            stage_layers = spconv.SparseSequential(*blocks_list) # 序列化该block
            self.encoder_layers.add_module(stage_name, stage_layers) # 将该layer添加进encoder_layers
        return out_channels # 128
