# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import to_2tuple
from torch.nn.init import normal_

from mmdet.models.utils.builder import TRANSFORMER

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size) # (2, 2)
        stride = to_2tuple(stride) # (2, 2)
        padding = to_2tuple(padding) # 'corner'
        dilation = to_2tuple(dilation) # (1, 1)

        self.padding = padding # 'corner'
        self.kernel_size = kernel_size # (2, 2)
        self.stride = stride # (2, 2)
        self.dilation = dilation # (1, 1)

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape # 256, 704
        kernel_h, kernel_w = self.kernel_size # 4, 4
        stride_h, stride_w = self.stride # 4, 4
        output_h = math.ceil(input_h / stride_h) # 64
        output_w = math.ceil(input_w / stride_w) # 176
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0) # 0
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0) # 0
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:]) # (256, 704) --> ()
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h]) # 右下角pad
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2 # 中心pad
                ])
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        conv_type='Conv2d',
        kernel_size=16,
        stride=16,
        padding='corner',
        dilation=1,
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims # 96
        if stride is None:
            stride = kernel_size # 4

        kernel_size = to_2tuple(kernel_size) # (4, 4)
        stride = to_2tuple(stride) # (4, 4)
        dilation = to_2tuple(dilation) # (1, 1)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size, # (4, 4)
                stride=stride, # (4, 4)
                dilation=dilation, # (1, 1)
                padding=padding) # (0, 0)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding) # (0, 0)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels, # 3
            out_channels=embed_dims, # 96
            kernel_size=kernel_size, # (4, 4)
            stride=stride, # (4, 4)
            padding=padding, # (0, 0)
            dilation=dilation, # (1, 1)
            bias=bias) # True

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1] # (LN, 96)
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None # None
            self.init_out_size = None # None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x) # pad图片

        x = self.projection(x) # 3 --> 96 (12, 96, 64, 176) 图片缩小为原来的1/4，通道变为96
        out_size = (x.shape[2], x.shape[3]) # 64, 176
        x = x.flatten(2).transpose(1, 2) # 将图片拉直，使像素变token，并且交换通道 --> (12, 11264, 96)
        if self.norm is not None:
            x = self.norm(x) 
        return x, out_size # (12, 11264, 96) 和（64, 176）


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified..
            Default: True.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels # 96
        self.out_channels = out_channels # 192
        if stride:
            stride = stride # 2
        else:
            stride = kernel_size # 2

        kernel_size = to_2tuple(kernel_size) # (2, 2)
        stride = to_2tuple(stride) # (2, 2)
        dilation = to_2tuple(dilation) # (1, 1)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding) # (0, 0)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size, # (2, 2)
            dilation=dilation, # (1, 1)
            padding=padding, # 0
            stride=stride) # (2, 2)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels # 2 * 2 * 96 = 384

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias) # 384-->192

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape # (12, 11264, 96)
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size # 64, 176
        assert L == H * W, 'input feature has wrong size'
        # (12, 11264, 96) --> (12, 64, 176, 96) --> (12, 96, 64, 176)
        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x) # (12, 384, 2816) : 2 * 2 * 96 = 384
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1 # 32
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1 # 88

        output_size = (out_h, out_w) # (32, 88)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C --> (12, 2816, 384)
        x = self.norm(x) if self.norm else x
        x = self.reduction(x) # (12, 2816, 192)
        return x, output_size # (12, 2816, 192) 和 (32, 88)


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1) # 对预测值进行clamp
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2) # (2, 300, 4)


@TRANSFORMER_LAYER.register_module()
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None # LN
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DetrTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x # (525, 2, 256)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(DetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query)) # List[Tensor(100, 2, 256)] 6层query输出
                else:
                    intermediate.append(query)
        return torch.stack(intermediate) # (6, 100, 2, 256)


@TRANSFORMER.register_module()
class Transformer(BaseModule):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(Transformer, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder) # 'DetrTransformerEncoder'
        self.decoder = build_transformer_layer_sequence(decoder) # 'DetrTransformerDecoder'
        self.embed_dims = self.encoder.embed_dims # 256

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `Transformer`.

        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape # 2, 256, 25, 21
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c] (525, 2, 256) num_query_first
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1) # (525, 2, 256)
        # query_embed:nn.Embedding(可学习query)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim] (100, 256)-->(100, 1, 256)-->(100, 2, 256)
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w] # (2, 25, 21)-->(2, 525)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask) # (525, 2, 256)
        target = torch.zeros_like(query_embed) # (100, 2, 256) 全0初始化
        # out_dec: [num_layers, num_query, bs, dim] --> (6, 100, 2, 256)
        out_dec = self.decoder(
            query=target, # (100, 2, 256)
            key=memory, # (525, 2, 256)
            value=memory, # (525, 2, 256)
            key_pos=pos_embed, # (525, 2, 256)
            query_pos=query_embed, # (100, 2, 256)
            key_padding_mask=mask) # (2, 525)
        out_dec = out_dec.transpose(1, 2) # (6, 2, 100, 256)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w) # (525, 2, 256)-->(2, 256, 525)-->(2, 256, 25, 21)
        return out_dec, memory


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):

        super(DeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate # True

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query # (300, 2, 256)
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                # (2, 300, 4) --> (2, 300, 1, 4) 
                # (2, 4, 2) --> (2, 4, 4) --> (2, 1, 4, 4)
                # (2, 300, 1, 4) * (2, 1, 4, 4) --> (2, 300, 4, 4)
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None] # (2, 300, 1, 2) / (2, 1, 4, 2)-->(2, 300, 4, 2)
            output = layer(
                output, # (300, 2, 256)
                *args,
                reference_points=reference_points_input, # (2, 300, 4, 2) 或 (2, 300, 4, 4)
                **kwargs)
            output = output.permute(1, 0, 2) # (2, 300, 256)

            if reg_branches is not None:
                tmp = reg_branches[lid](output) # 每个decoder layer都要进行回归
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points) # 更新参考点
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach() # 参考点不参与梯度计算

            output = output.permute(1, 0, 2) # (300, 2, 256)
            if self.return_intermediate:
                intermediate.append(output) # 将中间结果保存
                intermediate_reference_points.append(reference_points) # 将中间参考点保存

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points) # (6, 300, 2, 256)和(6, 2, 300, 2)

        return output, reference_points 


@TRANSFORMER.register_module()
class DeformableDetrTransformer(Transformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 **kwargs):
        super(DeformableDetrTransformer, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage # False
        self.num_feature_levels = num_feature_levels # 4
        self.two_stage_num_proposals = two_stage_num_proposals # 300
        self.embed_dims = self.encoder.embed_dims # 256
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)) # (4, 256) 层嵌入

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims) # 256-->256
            self.enc_output_norm = nn.LayerNorm(self.embed_dims) # 256
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                       self.embed_dims * 2) # 512-->512
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2) # 512
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2) # 256-->2

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape # (2, 13805, 256)
        proposals = []
        _cur = 0
        # 逐个特征层处理
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1) # 提取该特征层的mask eg:(2, 88, 118, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1) # 在88得维度取和--> eg:(88, 60)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1) # 在118的维度取和--> eg:(118, 70)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device)) # 计算网格 eg:(88, 118)
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # (88, 118, 2)
            # (2,)-->(2, 1)-->(2, 2)-->(2, 1, 1, 2)
            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            # (88, 118, 2)-->(1, 88, 118, 2)-->(2, 88, 118, 2) + 0.5 移动到中心 / 有效宽高 --> 归一化
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale # (2, 88, 118, 2)
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl) # (2, 88, 118, 2)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4) # 中心和宽高拼接 --> (2, 10384, 4)
            proposals.append(proposal) # 将该层预测的proposal加入proposal list
            _cur += (H * W) # 更新_cur索引
        output_proposals = torch.cat(proposals, 1) # 将各层的proposal进行拼接-->(2, 13805, 4)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True) # (2, 13805, 1)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf')) # 将padding部分设置为inf --> (2, 13805, 4)
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf')) # 将无效部分设置为inf --> (2, 13805, 4)

        output_memory = memory # (2, 13805, 256)
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)) # 将padding部分0填充
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0)) # 将无效部分0填充
        output_memory = self.enc_output_norm(self.enc_output(output_memory)) # (2, 13805, 256)
        return output_memory, output_proposals # (2, 13805, 256)和(2, 13805, 4)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        # 逐个特征层处理
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device), # eg:(92)
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device)) # (100,) --> (92, 100)
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H) # (1, 9200) / (2, 1) --> (2, 9200)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W) # (2, 4, 2)-->(2, 1, 4, 2)-->(2, 1)
            ref = torch.stack((ref_x, ref_y), -1) # (2, 9200, 2)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1) # (2, 12231, 2)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # (2, 12232, 1, 2) * (2, 1, 4, 2) -->(2, 12231, 4, 2)
        return reference_points # (2, 12231, 4, 2)

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape # eg: 92, 100
        valid_H = torch.sum(~mask[:, :, 0], 1) # eg:[92, 81] 计算特征图的有效高和宽
        valid_W = torch.sum(~mask[:, 0, :], 1) # eg:[69, 100]
        valid_ratio_h = valid_H.float() / H # 归一化
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # 在最后一维度拼接 --> (2, 2)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi # 2*pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device) # (128,)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale # (2, 300, 4)
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t # (2, 300, 4, 1) / (128) --> (2, 300, 4, 128)
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2) # (2, 300, 4, 64, 2) --> (2, 300, 512)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs, num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # 逐特征层处理
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape # eg:(2, 256, 92, 100)
            spatial_shape = (h, w) # eg:(92, 100)
            spatial_shapes.append(spatial_shape)
            # 将feat, mask和pos_embed的长和宽拉成1维
            feat = feat.flatten(2).transpose(1, 2) # (2, 256, 9200) --> (2, 9200, 256)
            mask = mask.flatten(1) # (2, 9200)
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # (2, 256, 9200) --> (2, 9200, 256)
            # 在原始位置编码基础上增加层编码
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1) # (2, 9200, 256) + (1, 1, 256) --> (2, 9200, 256)
            
            lvl_pos_embed_flatten.append(lvl_pos_embed) # (2, 9200, 256)
            feat_flatten.append(feat) # (2, 9200, 256)
            mask_flatten.append(mask) # (2, 9200)
        feat_flatten = torch.cat(feat_flatten, 1) # 在query维度拼接 (2, 12232, 256)
        mask_flatten = torch.cat(mask_flatten, 1) # (2, 12232)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # (2, 12232, 256)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device) # [[92, 100], [46, 50], [23, 25], [13, 13]]
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [0, 9200, 11500, 12075]
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1) # （2, 4, 2)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device) # (2, 12231, 4, 2)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims): (2, 12231, 256) --> (12231, 2, 256)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims):(2, 12231, 256) --> (12231, 2, 256)
        memory = self.encoder(
            query=feat_flatten, # (12231, 2, 256)
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten, # (12231, 2, 256)
            query_key_padding_mask=mask_flatten, # # (2, 12232)
            spatial_shapes=spatial_shapes, # (4, 2) [[92, 100], [46, 50], [23, 25], [13, 13]]
            reference_points=reference_points, # (2, 12231, 4, 2)
            level_start_index=level_start_index, # (4,)
            valid_ratios=valid_ratios, # (2, 4, 2)
            **kwargs) # (12231, 2, 256)

        memory = memory.permute(1, 0, 2) # (2, 12231, 256)
        bs, _, c = memory.shape # 2, 256
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes) # (2, 13805, 256)和(2, 13805, 4)
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory) # (2, 13805, 80)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals # (2, 13805, 4) 预测+参考点和偏移量

            topk = self.two_stage_num_proposals # 300
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1] # (2, 300) 这里只取第一类的前300(不合理）
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # (2, 300, 4)
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid() # (2, 300 4) 归一化0-1之间
            init_reference_out = reference_points # (2, 300 4)
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))) # (2, 300, 512)
            query_pos, query = torch.split(pos_trans_out, c, dim=2) # (2, 300, 256)和(2, 300, 256)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1) # (300, 512) --> (300, 256)和(300, 256)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1) # (300, 256)-->(1, 300, 256)-->(2, 300, 256)
            query = query.unsqueeze(0).expand(bs, -1, -1) # (300, 256)-->(1, 300, 256)-->(2, 300, 256)
            reference_points = self.reference_points(query_pos).sigmoid() # reference_points是可学习的 --> (2, 300, 2)
            init_reference_out = reference_points # (2, 300, 2)

        # decoder
        query = query.permute(1, 0, 2) # (2, 300, 256)-->(300, 2, 256)
        memory = memory.permute(1, 0, 2) # (2, 12231, 256)-->(12231, 2, 256)
        query_pos = query_pos.permute(1, 0, 2) # (2, 300, 256)-->(300, 2, 256)
        inter_states, inter_references = self.decoder(
            query=query, # (300, 2, 256)
            key=None,
            value=memory, # (12231, 2, 256)
            query_pos=query_pos, # (300, 2, 256)
            key_padding_mask=mask_flatten, # (2, 12231)
            reference_points=reference_points, # (2, 300, 2)
            spatial_shapes=spatial_shapes, # (4, 2)
            level_start_index=level_start_index, # (4,)
            valid_ratios=valid_ratios, # (2, 4, 2)
            reg_branches=reg_branches, # None
            **kwargs) # --> (6, 300, 2, 256)和(6, 2, 300, 2)

        inter_references_out = inter_references # Two Stage: (6, 2, 300, 4)
        # inter_states: (6, 300, 2, 256)
        # init_reference_out: (2, 300, 4)
        # inter_references_out: (6, 2, 300, 4)
        # enc_outputs_class: (2, 13805, 80)
        # enc_outputs_coord_unact: (2, 13805, 80)
        if self.as_two_stage:
            return inter_states, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact
        return inter_states, init_reference_out, \
            inter_references_out, None, None


@TRANSFORMER.register_module()
class DynamicConv(BaseModule):
    """Implements Dynamic Convolution.

    This module generate parameters for each sample and
    use bmm to implement 1*1 convolution. Code is modified
    from the `official github repo <https://github.com/PeizeSun/
    SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py#L258>`_ .

    Args:
        in_channels (int): The input feature channel.
            Defaults to 256.
        feat_channels (int): The inner feature channel.
            Defaults to 64.
        out_channels (int, optional): The output feature channel.
            When not specified, it will be set to `in_channels`
            by default
        input_feat_shape (int): The shape of input feature.
            Defaults to 7.
        with_proj (bool): Project two-dimentional feature to
            one-dimentional feature. Default to True.
        act_cfg (dict): The activation config for DynamicConv.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 with_proj=True,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(DynamicConv, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.with_proj = with_proj
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        num_output = self.out_channels * input_feat_shape**2
        if self.with_proj:
            self.fc_layer = nn.Linear(num_output, self.out_channels)
            self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, param_feature, input_feature):
        """Forward function for `DynamicConv`.

        Args:
            param_feature (Tensor): The feature can be used
                to generate the parameter, has shape
                (num_all_proposals, in_channels).
            input_feature (Tensor): Feature that
                interact with parameters, has shape
                (num_all_proposals, in_channels, H, W).

        Returns:
            Tensor: The output feature has shape
            (num_all_proposals, out_channels).
        """
        input_feature = input_feature.flatten(2).permute(2, 0, 1)

        input_feature = input_feature.permute(1, 0, 2)
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels, self.out_channels)

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = torch.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = torch.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        if self.with_proj:
            features = features.flatten(1)
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features
