# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.ckpt_convert import swin_converter
from ..utils.transformer import PatchEmbed, PatchMerging


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.
    multi-head Attention，将window和batch size合并，变为以window为单位，
    添加了相对位置编码和mask(防止不同块之间计算attn)
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims # 96
        self.window_size = window_size  # Wh, Ww # (7, 7)
        self.num_heads = num_heads # 3
        head_embed_dims = embed_dims // num_heads # 32
        self.scale = qk_scale or head_embed_dims**-0.5 # 0.176
        self.init_cfg = init_cfg # None

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH --> (169, 3)

        # About 2x faster than original impl
        Wh, Ww = self.window_size # 7, 7
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww) # (1, 49) 相对索引坐标
        rel_position_index = rel_index_coords + rel_index_coords.T # (49, 49) 行和列坐标相加
        rel_position_index = rel_position_index.flip(1).contiguous() # (49, 49)
        self.register_buffer('relative_position_index', rel_position_index) # 注册buffer

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias) # (96, 288)
        self.attn_drop = nn.Dropout(attn_drop_rate) # 0
        self.proj = nn.Linear(embed_dims, embed_dims) # (96, 96)
        self.proj_drop = nn.Dropout(proj_drop_rate) # 0

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape # 3120, 49, 96
        # (3120, 49, 288) --> (3120, 49, 3, 3, 32) --> (3, 3120, 3, 49, 32)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2] # (3120, 3, 49, 32)

        q = q * self.scale # 0.176
        attn = (q @ k.transpose(-2, -1)) # (3120, 3, 49, 49)

        # self.relative_position_bias_table: (169, 3)
        relative_position_bias = self.relative_position_bias_table[
            # (49, 49) --> (2401,) --> (49, 49, 1) --> (49, 49, 3)
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww --> (3, 49, 49)
        attn = attn + relative_position_bias.unsqueeze(0) # 注意力权重 + 相对位置编码 (3120, 3, 49, 49)

        if mask is not None:
            nW = mask.shape[0] # 260
            # (12, 260, 3, 49, 49) + (260, 49, 49) --> (1, 260, 1, 49, 49) --> (12, 260, 3, 49, 49)
            # 每个batch内的每个head的mask都是相同的
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0) 
            attn = attn.view(-1, self.num_heads, N, N) # (3120, 3, 49, 49)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        
        # (3120, 3, 49, 49) @ (3120, 3, 49, 32) --> (3120, 3, 49, 32) --> (3120, 49, 3, 32) --> (3120, 49, 96)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (3120, 49, 96)
        x = self.proj(x) # (3120, 49, 96)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1) # (start, end, step) [0, 1, 2, 3, 4, 5, 6]
        seq2 = torch.arange(0, step2 * len2, step2) # [0, 13, 26, 39, 52, 65, 78]
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1) # (7, 1) + (1, 7) --> (7, 7) --> (1, 49)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size # 7
        self.shift_size = shift_size # 3
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims, # 96
            num_heads=num_heads, # 3
            window_size=to_2tuple(window_size), # 7
            qkv_bias=qkv_bias, # True
            qk_scale=qk_scale, # None
            attn_drop_rate=attn_drop_rate, # 0
            proj_drop_rate=proj_drop_rate, # 0
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape # 12, 12264, 96
        H, W = hw_shape # 64, 176
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C) # 将query变为view为图片形式-->(12, 64, 176, 96)

        # pad feature maps to multiples of window size --> pad特征图使得可以被window size整除
        pad_r = (self.window_size - W % self.window_size) % self.window_size # 6
        pad_b = (self.window_size - H % self.window_size) % self.window_size # 6
        # pad里面每两个元素为1组，指定了由低维到高维，每一维度，前面填充和后面填充的数值单位。
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b)) # (12, 70, 182, 96)
        H_pad, W_pad = query.shape[1], query.shape[2] # (70, 182)

        # cyclic shift 如果shift_size大于0则进行循环移位和mask计算
        if self.shift_size > 0:
            # shifts张量元素移位的位数, 如果该参数是一个元组（例如shifts=(x,y)），
            # dims必须是一个相同大小的元组（例如dims=(a,b)），相当于在第a维度移x位，在b维度移y位
            # shifts的值为正数相当于向下挤牙膏，挤出的牙膏又从顶部塞回牙膏里面；
            # shifts的值为负数相当于向上挤牙膏，挤出的牙膏又从底部塞回牙膏里面
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)) # (12, 70, 182, 96)

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device) # (1, 70, 182, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None)) # (0, -7), (-7, -3), (-3, None)
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None)) # (0, -7), (-7, -3), (-3, None)
            
            # 为不同的块赋予不同的值
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt # (1, 70, 182, 1)
                    cnt += 1

            # nW, window_size, window_size, 对mask进行窗口分割，合并batch size
            mask_windows = self.window_partition(img_mask) # (260, 7, 7, 1)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size) # (260, 49)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # (260, 1, 49) - (260, 49, 1)-->(260, 49, 49)
            # 每个window的mask --> (260, 49, 49)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query # (12, 70, 182, 96)
            attn_mask = None

        # nW*B, window_size, window_size, C 分割窗口，合并batch
        query_windows = self.window_partition(shifted_query) # (3120, 7, 7, 96)
        # nW*B, window_size*window_size, C 将win拉直
        query_windows = query_windows.view(-1, self.window_size**2, C) # (3120, 49, 96)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask) # 

        # merge windows
        # (3120, 49, 96) --> (3120, 7, 7, 96)
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C 恢复窗口
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad) # (12, 70, 182, 96)
        # reverse cyclic shift 反向循环恢复原始位置
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)) # (12, 70, 182, 96)
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous() # (12, 64, 176, 96)

        x = x.view(B, H * W, C) # (12, 12264, 96)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size # 7
        B = int(windows.shape[0] / (H * W / window_size / window_size)) # 计算batch size-->12
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1) # (12, 10, 26, 7, 7, 96)
        # (12, 10, 26, 7, 7, 96) --> (12, 10, 7, 26, 7, 96) --> (12, 70, 182, 96)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x # (12, 70, 182, 96)

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape # (12, 70, 182, 96)
        window_size = self.window_size # 7 
        # 分割窗口
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)  # (12, 10, 7, 26, 7, 96)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() # (12, 10, 26, 7, 7, 96)
        windows = windows.view(-1, window_size, window_size, C) # (3120, 7, 7, 96)
        return windows # (3120, 7, 7, 96)


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg # GELU
        self.with_cp = with_cp # False

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1] # LN
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims, # 96
            num_heads=num_heads, # 3
            window_size=window_size, # 7
            shift_size=window_size // 2 if shift else 0, # 3 或 0
            qkv_bias=qkv_bias, # True
            qk_scale=qk_scale, # None
            attn_drop_rate=attn_drop_rate, # 0
            proj_drop_rate=drop_rate, # 0
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1] # LN
        self.ffn = FFN(
            embed_dims=embed_dims, # 96
            feedforward_channels=feedforward_channels, # 384
            num_fcs=2,
            ffn_drop=drop_rate, # 0.0
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate), # 0.1818
            act_cfg=act_cfg, # GELU
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x # 保留残差 (12, 11264, 96)
            x = self.norm1(x) 
            x = self.attn(x, hw_shape) # hw_shape:(64, 176) ShiftWindowMSA

            x = x + identity # 残差连接 (12, 11264, 96)

            identity = x 
            x = self.norm2(x)
            x = self.ffn(x, identity=identity) # (12, 11264, 96) FFN

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate # [0, 0.01818]
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList() # 初始化ModuleList
        # 逐个模块构建
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims, # 96
                num_heads=num_heads, # 3
                feedforward_channels=feedforward_channels, # 384
                window_size=window_size, # 7
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias, # True
                qk_scale=qk_scale, # None
                drop_rate=drop_rate, # 0
                attn_drop_rate=attn_drop_rate, # 0
                drop_path_rate=drop_path_rates[i], # 0
                act_cfg=act_cfg, # GELU
                norm_cfg=norm_cfg, # LN
                with_cp=with_cp, # False
                init_cfg=None) # None
            self.blocks.append(block) # 将构建后的block加入ModuleList

        self.downsample = downsample # (adap_padding, sampler, norm, reduction)

    def forward(self, x, hw_shape):
        # 每个stage内逐个block处理
        for block in self.blocks:
            x = block(x, hw_shape)

        # 每个stage后会进行PatchMerging, 通过nn.Unfold实现
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape) # (12, 2816, 192) 和 (32, 88)
            return x_down, down_hw_shape, x, hw_shape # 下采样后的x和下采样前的x
        else:
            return x, hw_shape, x, hw_shape


@BACKBONES.register_module()
class SwinTransformer(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights # True
        self.frozen_stages = frozen_stages # -1
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size) # (224, 224)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths) # 4
        self.out_indices = out_indices # [1, 2, 3]
        self.use_abs_pos_embed = use_abs_pos_embed # False

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate) # 0

        # set stochastic depth decay rule
        total_depth = sum(depths) # [2, 2, 6, 2] --> 12
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth) # (0, 0.2, 12)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers): # 4
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels, # eg:96
                    out_channels=2 * in_channels, # eg:96 * 2
                    stride=strides[i + 1], # 2 
                    norm_cfg=norm_cfg if patch_norm else None, # "LN"
                    init_cfg=None) # 
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels, # 96
                num_heads=num_heads[i], # [3, 4, 12, 24]
                feedforward_channels=mlp_ratio * in_channels, # 4 * 96
                depth=depths[i], # [2, 2, 6, 2]
                window_size=window_size, # 7
                qkv_bias=qkv_bias, # True
                qk_scale=qk_scale, # None
                drop_rate=drop_rate, # 0.0
                attn_drop_rate=attn_drop_rate, # 0.0
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])], # 0.2
                downsample=downsample, # (ASNR)
                act_cfg=act_cfg, # GELU
                norm_cfg=norm_cfg, # LN
                with_cp=with_cp, # False
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels # 192

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)] # [96, 192, 384, 768]
        # Add a norm layer for each output
        for i in out_indices: # [1, 2, 3]
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1] # LN
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        # 冻结模型的步骤：
        # 1.将该模块设置为eval模式
        # 2.禁止该模块的param反向传播，将param.requires_grad = False
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval() # 将norm_layer层设置为eval模式
                for param in norm_layer.parameters():
                    param.requires_grad = False # 禁止param反向传播

            m = self.stages[i - 1] # 获取第i-1个stage
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        # 初始化权重
        logger = get_root_logger() # 获取logger
        if self.init_cfg is None: # 如果不存在init_cfgz,则log记录
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            # 逐个模块初始化
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            # 在init_cfg中通过Pretrained指定checkpoint路径
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu') # 加载checkpoint权重文件
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict'] # 赋值state_dict
            elif 'model' in ckpt:
                _state_dict = ckpt['model'] # 赋值model
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict) # 将原始仓库的预训练权重转换为mmdet3d形式

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v # 提取backbone后的key值，作为新key，赋予value(去除backbone前中)

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()} # 去除module前缀

            # reshape absolute position embedding 处理绝对位置嵌入
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed 处理相对位置嵌入bias
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False) # 模型加载权重

    def forward(self, x):
        x, hw_shape = self.patch_embed(x) # (12, 11264, 96)和（64, 176）

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        # 逐个stage处理
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape) # 下采样后的x和下采样前的x eg: (12, 704, 384)和[16, 44]和[12, 2861, 192]和[32, 88]
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous() # (12, 32, 88, 192)-->(12, 192, 32, 88)
                outs.append(out) # (12, 192, 32, 88)和(12, 384, 16, 44)和(12, 768, 8, 22)

        return outs # (12, 192, 32, 88)和(12, 384, 16, 44)和(12, 768, 8, 22)
