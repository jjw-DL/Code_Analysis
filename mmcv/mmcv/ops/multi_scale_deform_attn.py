# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable

from mmcv import deprecated_api_warning
from mmcv.cnn import constant_init, xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import BaseModule
from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class MultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step # 64
        # value: (2, 12231, 8, 32)
        # spatial_shapes: (4, 2) --> [[100, 92], [50, 46], [25, 23], [13, 12]]
        # level_start_index: (4,) --> (0, 9200, 11500, 12075)
        # sampling_locations: (2, 12231, 8, 4, 4, 2)
        # attention_weights: (2, 12231, 8, 4, 4)
        output = ext_module.ms_deform_attn_forward(
            value, # (2, 12231, 8, 32)
            value_spatial_shapes, # (4, 2)
            value_level_start_index, # (4,)
            sampling_locations, # (2, 12231, 8, 4, 4, 2)
            attention_weights, # (2, 12231, 8, 4, 4)
            im2col_step=ctx.im2col_step) # --> (2, 12231, 256)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """
    # value: (2, 12231, 8, 32)
    # spatial_shapes: (4, 2) --> [[100, 92], [50, 46], [25, 23], [13, 12]]
    # level_start_index: (4,) --> (0, 9200, 11500, 12075)
    # sampling_locations: (2, 12231, 8, 4, 4, 2)
    # attention_weights: (2, 12231, 8, 4, 4)
    # self.im2col_step: 64
    bs, _, num_heads, embed_dims = value.shape # (2, 12231, 8, 32)
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape # 12231, 8, 4, 4
    # [[100, 92], [50, 46], [25, 23], [13, 12]]
    # 将value按层分隔(9200, 2300, 575, 156)
    # [(2, 9200, 8, 32), (2, 2300, 8, 32), (2, 575, 8, 32), (2, 156, 8, 32)]
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1) 
    sampling_grids = 2 * sampling_locations - 1 # 为grid_sample做准备, 将坐标转换到[-1, 1] --> (2, 12231, 8, 4, 4, 2)
    sampling_value_list = []
    # 逐层处理
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims -> (2, 9200, 8, 32)
        # bs, H_*W_, num_heads*embed_dims -> (2, 9200, 256)
        # bs, num_heads*embed_dims, H_*W_ -> (2, 256, 9200)
        # bs*num_heads, embed_dims, H_, W_ --> (16, 32, 100, 92)
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 -> (2, 12231, 8, 4, 2)
        # bs, num_heads, num_queries, num_points, 2 -> (2, 8, 12231, 4, 2)
        # bs*num_heads, num_queries, num_points, 2 --> (16, 12231, 4, 2)
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points --> (16, 32, 12231, 4)
        sampling_value_l_ = F.grid_sample(
            value_l_, # (16, 32, 100, 92)
            sampling_grid_l_, #  (16, 12231, 4, 2)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False) # (16, 32, 12231, 4)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) -> (2, 12231, 8, 4, 4)
    # (bs, num_heads, num_queries, num_levels, num_points) -> (2, 8, 12231, 4, 4)
    # (bs*num_heads, 1, num_queries, num_levels*num_points) --> (16, 1, 12231, 16)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    # (16, 32, 12231, 4, 4) --> (16, 32, 12231, 16) * (16, 1, 12231, 16) --> (16, 32, 12231)--> (2, 256, 12231)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries) # (2, 256, 12231)
    return output.transpose(1, 2).contiguous() # (2, 12231, 256)


@ATTENTION.register_module()
class MultiScaleDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads # 256 // 8 = 32
        self.norm_cfg = norm_cfg # None
        self.dropout = nn.Dropout(dropout) # 0.1
        self.batch_first = batch_first # False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step # 64
        self.embed_dims = embed_dims # 256
        self.num_levels = num_levels # 4
        self.num_heads = num_heads # 8
        self.num_points = num_points # 4
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2) # 通过query学习参考点(逐head逐level逐点)偏移 256-->8*4*4*2=256
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points) # 256-->8*4*4=128
        self.value_proj = nn.Linear(embed_dims, embed_dims) # 256-->256
        self.output_proj = nn.Linear(embed_dims, embed_dims) # 256-->256
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.) # 采样点的权重初始化
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads) # 将2pi均分8份
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1) # 对角度取cos和sin，并拼接 (8, 2), 一个圆上的点
        # [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1) # (8, 2)-->(8, 1, 1, 2)-->(8, 4, 4, 2)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1 # 在点的维度上乘索引

        self.sampling_offsets.bias.data = grid_init.view(-1) # 采样点的bias初始化
        constant_init(self.attention_weights, val=0., bias=0.) # attention权重全0初始化 
        xavier_init(self.value_proj, distribution='uniform', bias=0.) # 值权重初始化
        xavier_init(self.output_proj, distribution='uniform', bias=0.) # 输出权重初始化
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first: # not False-->True
            # change to (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2) # Encoder:(2, 12231, 256) Decoder:(2, 300, 256)
            value = value.permute(1, 0, 2) # Encoder:(2, 12231, 256) Decoder:(2, 12231, 256)

        bs, num_query, _ = query.shape # 2, 12231 或 2， 300
        bs, num_value, _ = value.shape # 2, 12231
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        # value
        value = self.value_proj(value) # 将value进行映射-->(2, 12231, 256)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0) # key_padding_mask:(2, 12231), 将value的padding部分用0填充
        value = value.view(bs, num_value, self.num_heads, -1) # (2, 12231, 8, 32) 将value在channel上拆分为8个head
        # sampling_offsets
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2) # (2, 12231, 256) --> (2, 12231, 8, 4, 4, 2)
        # attention_weights
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points) # (2, 12231, 128) --> (2, 12231, 8, 16)
        attention_weights = attention_weights.softmax(-1) # softmax处理 --> (2, 12231, 8, 16)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points) # (2, 12231, 8, 4, 4)
        
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # (4, 2) --> [[100, 92], [50, 46], [25, 23], [13, 12]]
            # reference_points:(2, 12231, 4, 2) --> (2, 12231, 1, 4, 1, 2)
            # sampling_offsets: (2, 12231, 8, 4, 4, 2)
            # offset_normalizer:(4, 2)-->(1, 1, 1, 4, 1, 2) 每一层的offsets都要进行归一化
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :] # (2, 12231, 8, 4, 4, 2)
        elif reference_points.shape[-1] == 4:
            # 前2维是中心坐标, 后2维度是宽高 + offsets / 采样点个数 * 宽和高的一半
            # 因为采样点初始化最大值就是采样点的个数, 这里是进行归一化 self.num_points:4
            # reference_points:(2, 300, 4, 4) --> (2, 300, 1, 4, 1, 2)
            # sampling_offsets:(2, 300, 8, 4, 4, 2)
            # --> (2, 300,8, 4, 4, 2)
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            # value: (2, 12231, 8, 32)
            # spatial_shapes: (4, 2) --> [[100, 92], [50, 46], [25, 23], [13, 12]]
            # level_start_index: (4,) --> (0, 9200, 11500, 12075)
            # sampling_locations: (2, 12231, 8, 4, 4, 2)
            # attention_weights: (2, 12231, 8, 4, 4)
            # self.im2col_step: 64
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step) # (2, 12231, 256)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights) 

        output = self.output_proj(output) # (2, 12231, 256) 特征整合

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2) # (12231, 2, 256)

        return self.dropout(output) + identity # ADD
