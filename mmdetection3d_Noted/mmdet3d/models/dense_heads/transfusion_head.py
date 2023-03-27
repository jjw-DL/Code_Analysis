import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_batch


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1), # 2 --> 128
            nn.BatchNorm1d(num_pos_feats), # 128
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1)) # 128 --> 128

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous() # (2, 2, 200)
        position_embedding = self.position_embedding_head(xyz) # (2, 128, 200)
        return position_embedding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout) # 多头注意力机制 self attention
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout) # cross attention
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward) # 128-->256
        self.dropout = nn.Dropout(dropout) # 0.1
        self.linear2 = nn.Linear(dim_feedforward, d_model) # 256-->128

        self.norm1 = nn.LayerNorm(d_model) # 128
        self.norm2 = nn.LayerNorm(d_model) # 128
        self.norm3 = nn.LayerNorm(d_model) # 128
        self.dropout1 = nn.Dropout(dropout) # 0.1
        self.dropout2 = nn.Dropout(dropout) # 0.1
        self.dropout3 = nn.Dropout(dropout) # 0.1

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation) # relu

        self.self_posembed = self_posembed # 2-->128-->128
        self.cross_posembed = cross_posembed # 2-->128-->128

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq (2, 128, 200)  
        :param key: B C Pk (2, 128, 32400)
        :param query_pos: B Pq 2 (2, 200, 2)
        :param key_pos: B Pk 2 (2, 32400, 2)
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1) # (2, 200, 2)-->(2, 200, 128)-->(200, 2, 128)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1) # (2, 32400, 2)-->(2, 32400, 128)-->(32400, 2, 128)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1) # (2, 128, 200) --> (200, 2, 128)
        key = key.permute(2, 0, 1) # (2, 128, 32400) --> (32400, 2, 128)

        # query的self attention, query first
        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed) # (200, 2, 128) 在embed token中嵌入pos embed
            query2 = self.self_attn(q, k, value=v)[0] # 自注意力 （200, 2, 128）
            query = query + self.dropout1(query2)
            query = self.norm1(query)
        # cross attention
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0] # (200, 2, 128)
        query = query + self.dropout2(query2) # (200, 2, 128)
        query = self.norm2(query)
        # FFN
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query)))) # (200, 2, 128)
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0) # (200, 2, 128) --> (2, 128, 200)
        return query


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim # 128
        self.kdim = kdim if kdim is not None else embed_dim # 128
        self.vdim = vdim if vdim is not None else embed_dim # 128
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim # True

        self.num_heads = num_heads # 8
        self.dropout = dropout # 0.1
        self.head_dim = embed_dim // num_heads # 16
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim)) # 初始化为(3*128, 128)

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim)) # (384,)
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias) # (128, 128)

        if add_bias_kv: # False
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn # False

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim: # True
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None: # None
            xavier_uniform_(self.bias_k)
        if self.bias_v is not None: # None
            xavier_uniform_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


def multi_head_attention_forward(query,  
                                 key,  
                                 value,  
                                 embed_dim_to_check,  
                                 num_heads,  
                                 in_proj_weight,  
                                 in_proj_bias,  
                                 bias_k,  
                                 bias_v,  
                                 add_zero_attn,  
                                 dropout_p,  
                                 out_proj_weight, 
                                 out_proj_bias,
                                 training=True,
                                 key_padding_mask=None, 
                                 need_weights=True,  
                                 attn_mask=None,  
                                 use_separate_proj_weight=False,  
                                 q_proj_weight=None,
                                 k_proj_weight=None,
                                 v_proj_weight=None,
                                 static_k=None,
                                 static_v=None,
                                 ):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value) # False
    kv_same = torch.equal(key, value) # True

    tgt_len, bsz, embed_dim = query.size() # 200, 2, 128
    assert embed_dim == embed_dim_to_check # 128
    assert list(query.size()) == [tgt_len, bsz, embed_dim] # 200, 2, 128
    assert key.size() == value.size() # (32400, 2, 128)

    head_dim = embed_dim // num_heads # 128 / 8 = 16
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5 # 1/sqrt(16)=0.25

    if use_separate_proj_weight is not True: # q,k和v一起计算
        if qkv_same: # q,k和v均相同
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1) # (200, 2, 128)

        elif kv_same: # 只有kv相同
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias # (384,)
            _start = 0
            _end = embed_dim # 128
            _w = in_proj_weight[_start:_end, :] # (128, 128)
            if _b is not None:
                _b = _b[_start:_end] # (128, )
            q = F.linear(query, _w, _b) # (200, 2, 128) * (128, 128) --> (200, 2, 128)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias # (256,)
                _start = embed_dim # 128
                _end = None
                _w = in_proj_weight[_start:, :] # (256, 128)
                if _b is not None:
                    _b = _b[_start:] # (256,)
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1) # (32400, 2, 128)和(32400, 2, 128)

        else: # q,k和v均不相同
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim # 128
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim # 128
            _end = embed_dim * 2 # 256
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2 # 256
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else: # q,k和v分开计算
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling # q * scale --> (200, 2, 128)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)]) # 在k和v后面拼接bias
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1) # 在atten mask后面拼接0
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1) # 在key_padding_mask后拼接0
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None
    
    # head拆分
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1) # (200, 2 * 8, 16)-->(16, 200, 16)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) # (32400, 2 * 8, 16)-->(16, 32400, 16)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) # (32400, 2 * 8, 16)-->(16, 32400, 16)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1) # 32400

    if key_padding_mask is not None: # None
        assert key_padding_mask.size(0) == bsz # 2
        assert key_padding_mask.size(1) == src_len # 32400

    if add_zero_attn: # False
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1) # (16, 32401, 16)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) # Q*K^T --> (16, 200, 32400)
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None: # None
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None: # 在pad为True的部分设置为-inf，在softmax的时候变0
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1) # softmax处理
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v) # softmax(Q*K^T)*V --> (16, 200, 16)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    # (16, 200, 16) --> (200, 16, 16) --> (200, 2, 128)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim) 
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias) # 多一次合并的卷积-->(200, 2, 128)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len) # (2, 8, 200, 32400)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads # (200, 2, 128)和(2, 200, 32400)
    else:
        return attn_output, None


class FFN(nn.Module):
    def __init__(self,
                 in_channels, # 128
                 heads, # (center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2), heatmap=(10, 2))
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(FFN, self).__init__()

        self.heads = heads # (center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2), heatmap=(10, 2))
        self.init_bias = init_bias # -2.19
        # 逐个模块构建
        for head in self.heads: # 只会取key值
            classes, num_conv = self.heads[head] # 全部经过1层卷积(128-->64,做了num_conv - 1)后分类

            conv_layers = []
            c_in = in_channels # 128
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in, # 128
                        head_conv, # 64
                        kernel_size=final_kernel, # 1
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv, # 64
                    classes, # eg:2
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers) # 以head命名变量并赋值

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

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
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x) # 将feature map逐个通过各head

        return ret_dict


@HEADS.register_module()
class TransFusionHead(nn.Module):
    def __init__(self,
                 fuse_img=False,
                 num_views=0,
                 in_channels_img=64, 
                 out_size_factor_img=4, # 4
                 num_proposals=128, # 200
                 auxiliary=True, # True
                 in_channels=128 * 3, # 256 * 2
                 hidden_channel=128, # 128
                 num_classes=4, # 10
                 # config for Transformer
                 num_decoder_layers=3, # 1
                 num_heads=8, # 8
                 learnable_query_pos=False, # False
                 initialize_by_heatmap=False, # True
                 nms_kernel_size=1, # 3
                 ffn_channel=256, # 256
                 dropout=0.1, # 0.1
                 bn_momentum=0.1, # 0.1
                 activation='relu', # relu
                 # config for FFN
                 common_heads=dict(), # dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'), # FocalLoss
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'), # L1Loss
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'), # GaussianFocalLoss
                 # others
                 train_cfg=None, # train_cfg
                 test_cfg=None, # test_cfg
                 bbox_coder=None, # 'TransFusionBBoxCoder'
                 ):
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes # 10
        self.num_proposals = num_proposals # 200
        self.auxiliary = auxiliary # True
        self.in_channels = in_channels # 256 * 2
        self.num_heads = num_heads # 8
        self.num_decoder_layers = num_decoder_layers # 1
        self.bn_momentum = bn_momentum # 0.1
        self.learnable_query_pos = learnable_query_pos # False
        self.initialize_by_heatmap = initialize_by_heatmap # True
        self.nms_kernel_size = nms_kernel_size # 3
        if self.initialize_by_heatmap is True:
            assert self.learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"
        self.train_cfg = train_cfg 
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False) # True
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls) # FocalLoss
        self.loss_bbox = build_loss(loss_bbox) # L1Loss
        self.loss_iou = build_loss(loss_iou) # VarifocalLoss
        self.loss_heatmap = build_loss(loss_heatmap) # GaussianFocalLoss

        self.bbox_coder = build_bbox_coder(bbox_coder) # TransFusionBBoxCoder
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels, # 512
            hidden_channel, # 128
            kernel_size=3,
            padding=1,
            bias=bias, # auto
        )

        if self.initialize_by_heatmap:
            layers = []
            layers.append(ConvModule(
                hidden_channel, # 128
                hidden_channel, # 128
                kernel_size=3,
                padding=1,
                bias=bias, # auto
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
            layers.append(build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel, # 128
                num_classes, # 10
                kernel_size=3,
                padding=1,
                bias=bias, # auto
            ))
            self.heatmap_head = nn.Sequential(*layers) # 128-->128-->10
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1) # 10 --> 128
        else:
            # query feature
            self.query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals)) # (1, 128, 200)
            self.query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=learnable_query_pos) # (1, 200, 2)

        # transformer decoder layers for object query with LiDAR feature --> transformer decoder(layersequence)
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers): # 1
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation, # 128, 8, 256, 0.1, relu
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel), # 2, 128
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel), # 2, 128
                ))

        # Prediction Head --> init_layer
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads) # dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs))) # 10, 2
            self.prediction_heads.append(FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)) # 128

        self.fuse_img = fuse_img # True
        if self.fuse_img:
            self.num_views = num_views # 6
            self.out_size_factor_img = out_size_factor_img # 4
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,  # 256
                hidden_channel, # 128
                kernel_size=3,
                padding=1,
                bias=bias, # auto
            )
            # -------------------------------------
            # image guided query initialization 注释
            # -------------------------------------
            self.heatmap_head_img = copy.deepcopy(self.heatmap_head) # 深拷贝一份heatmap_head
            
            # transformer decoder layers for img fusion
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation, # 128, 8, 256, 0.1, relu
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel), # 2, 128
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel), # 2, 128
                ))
            # -------------------------------------
            # image guided query initialization 注释
            # -------------------------------------
            # cross-attention only layers for projecting img feature onto BEV
            for i in range(num_views): # 6
                self.decoder.append(
                    TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_only=True,
                    ))
            self.fc = nn.Sequential(*[nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)]) # 128-->128

            heads = copy.deepcopy(common_heads) # dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel * 2, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)) # 256(预测头)

        self.init_weights()
        self._init_assigner_sampler() # 初始化assigner和sampler-->HungarianAssigner3D和PseudoSampler

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'] # 180
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'] # 180
        self.bev_pos = self.create_2D_grid(x_size, y_size) # (1, 32400, 2)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]] # [0, 179, 180]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]) # (180, 180)
        batch_x = batch_x + 0.5 # 取中心点
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None] # (1, 2, 180, 180)
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1) # (1, 2, 32400) --> (1, 32400, 2)
        return coord_base # (1, 32400, 2)

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m) # 初始化decoder
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query) # 初始化query
        self.init_bn_momentum() # 初始化BN

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum # 0.1

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling: # False
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler() # 直接返回正例和负例的索引，伪采样
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner) # HungarianAssigner3D
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, img_inputs, img_metas):
        """Forward function for CenterPoint.

        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 180(H), 180(W)]. (consistent with L748)
            img_inputs (torch.Tensor): [6, 256, 232, 400]

        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0] # 2
        lidar_feat = self.shared_conv(inputs) # (2, 128, 180, 180)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]--> (2, 128, 32400)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device) # (1, 32400, 2) --> (2, 32400, 2)

        if self.fuse_img:
            img_feat = self.shared_conv_img(img_inputs)  # [BS*n_views, C, H, W]-->(6, 128, 232, 400)

            img_h, img_w, num_channel = img_inputs.shape[-2], img_inputs.shape[-1], img_feat.shape[1] # 232, 400, 128
            raw_img_feat = img_feat.view(batch_size, self.num_views, num_channel, img_h, img_w).permute(0, 2, 3, 1, 4) # [BS, C, H, n_views, W]-->(1, 128, 232, 6, 400)
            # -------------------------------------
            # image guided query initialization 注释
            # -------------------------------------
            img_feat = raw_img_feat.reshape(batch_size, num_channel, img_h, img_w * self.num_views)  # [BS, C, H, n_views*W]-->(1, 128, 232, 2400)
            img_feat_collapsed = img_feat.max(2).values # [BS, C, n_views*W]-->(1, 128, 2400)
            img_feat_collapsed = self.fc(img_feat_collapsed).view(batch_size, num_channel, img_w * self.num_views) # [BS, C, n_views*W]-->(1, 128, 2400)

            # positional encoding for image guided query initialization
            if self.img_feat_collapsed_pos is None:
                # (1, 2400, 2)
                img_feat_collapsed_pos = self.img_feat_collapsed_po = self.create_2D_grid(1, img_feat_collapsed.shape[-1]).to(img_feat.device)
            else:
                img_feat_collapsed_pos = self.img_feat_collapsed_pos

            bev_feat = lidar_feat_flatten # (2, 128, 32400)
            for idx_view in range(self.num_views): # img column and lidar bev cross attention
                # bev_feat:(2, 128, 32400)
                # bev_feat:(2, 128, 2400) --> (2, 128, 400)
                # bev_pos:(1, 32400, 2)
                # img_feat_collapsed_pos:(1, 2400, 2) --> (1, 400, 2) 这里非常占内存, 6个encoder, 是否可以减少到一个？ cross atten--> lidar bev_feat
                bev_feat = self.decoder[2 + idx_view](bev_feat, img_feat_collapsed[..., img_w * idx_view:img_w * (idx_view + 1)], bev_pos, img_feat_collapsed_pos[:, img_w * idx_view:img_w * (idx_view + 1)])

        ###################################
        # image guided query initialization
        ################################### 
        if self.initialize_by_heatmap:
            dense_heatmap = self.heatmap_head(lidar_feat) # (2, 10, 180, 180)
            dense_heatmap_img = None
            # -------------------------------------
            # image guided query initialization 注释
            # -------------------------------------
            if self.fuse_img:
                dense_heatmap_img = self.heatmap_head_img(bev_feat.view(lidar_feat.shape))  # [BS, num_classes, H, W] --> (2, 10, 180, 180)
                # 融合图像特征后的分类分数 --> 取二者的平均值 (图像特征辅助分类, 从而产生更加精确的query)
                heatmap = (dense_heatmap.detach().sigmoid() + dense_heatmap_img.detach().sigmoid()) / 2 
            else:
                heatmap = dense_heatmap.detach().sigmoid() # (2, 10, 180, 180)
            padding = self.nms_kernel_size // 2 # 3 // 2 = 1
            local_max = torch.zeros_like(heatmap) # (2, 10, 180, 180) 全0初始化
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0) # (2, 10, 178, 178)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner # (2, 10, 180, 180) max后的最值
            ## for Pedestrian & Traffic_cone in nuScenes
            if self.test_cfg['dataset'] == 'nuScenes':
                local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            # for Pedestrian & Cyclist in Waymo
            elif self.test_cfg['dataset'] == 'Waymo':  
                local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max) # 巧妙处理
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1) # (2, 10, 32400)

            """
            heatmap_first = heatmap.clone()
            heatmap_second = heatmap.clone()
            heatmap_first[:, [7, 8], :] = 0
            heatmap_second[:, [0, 1, 2, 3, 4, 5, 6, 9]] = 0
            ratio = 0.66
            num_proposals_first = int(self.num_proposals * ratio)
            num_proposals_second = self.num_proposals - num_proposals_first
            top_proposals_first = heatmap_first.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :num_proposals_first]
            top_proposals_second = heatmap_second.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :num_proposals_second]
            top_proposals = torch.cat([top_proposals_first, top_proposals_second], dim=-1)
            """
            
            # top #num_proposals among all classes (2, 324000) --> (2, 200)
            top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals] # 取出前200个最大值
            
            top_proposals_class = top_proposals // heatmap.shape[-1] # (2, 200) 分类类别
            top_proposals_index = top_proposals % heatmap.shape[-1] # (2, 200) 表示预测该类的特征图索引(那个特征预测了该类)
            # 根据索引在lidar的bev特征图中抽取query的特征
            # lidar_feat_flatten:(2, 128, 32400)   index:(2, 1, 200)-->(2, 128, 200)   query_feat:(2, 128, 200)
            query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
            
            # 赋值query的分类类别
            self.query_labels = top_proposals_class # (2, 200) 分类类别
            # add category embedding 将query的类别进行one hot编码，然后计算类别嵌入特征
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1) # (2, 200, 10)-->(2, 10, 200)
            query_cat_encoding = self.class_encoding(one_hot.float()) # 10-->128 : (2, 128, 200)
            query_feat += query_cat_encoding # (2, 128, 200) # 使得query特征融合类别特征
            # bev_pos:(2, 32400, 2)  index:(2, 1, 200)-->(2, 200, 1)-->(2, 200, 2) query_pos:(2, 200, 2)
            query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
        else:
            query_feat = self.query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            base_xyz = self.query_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, num_proposals, 2]

        ###################################################
        # transformer decoder layer (LiDAR feature as K,V)
        ###################################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_' # last

            # Transformer Decoder Layer
            # query_feat: (2, 128, 200)  
            # lidar_feat_flatten: (2, 128, 32400)
            # query_pos: B Pq 2 (2, 200, 2)
            # bev_pos: (2, 32400, 2)
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos) # (2, 128, 200)

            # Prediction
            # {'center':(2, 2, 200), 'height':(2, 1, 200), 'dim':(2, 3, 200), 'rot':(2, 2, 200),'vel':(2, 2, 200),'heatmap':(2, 10, 200)}
            res_layer = self.prediction_heads[i](query_feat) # (2, 128, 200)-->dict(6个head输出)
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1) # 加特征图上的点坐标，预测为偏移量 (2, 200, 2) 这里体现了query的重要性
            first_res_layer = res_layer
            if not self.fuse_img:
                ret_dicts.append(res_layer)

            # for next level positional embedding 
            # 如果lidar transformer为多层，则为下一层的位置编码做准备，同时也是lidar和image融合的位置编码
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        #################################################
        # transformer decoder layer (img feature as K,V)
        #################################################
        if self.fuse_img:
            # positional encoding for image fusion
            # img_feat_flatten = img_feat.view(batch_size, self.num_views, img_feat.shape[1], -1)  # [BS, n_views, C, H*W]
            # raw_img_feat:(1, 128, 232, 6, 400)
            # img_feat:(1, 6, 128, 232, 400)
            img_feat = raw_img_feat.permute(0, 3, 1, 2, 4) # [BS, n_views, C, H, W]
            # [BS, n_views, C, H*W] --> (1, 6, 128, 92800)
            img_feat_flatten = img_feat.view(batch_size, self.num_views, num_channel, -1)
            if self.img_feat_pos is None:
                (h, w) = img_inputs.shape[-2], img_inputs.shape[-1] # (232, 400) 或者（120, 200）
                img_feat_pos = self.img_feat_pos = self.create_2D_grid(h, w).to(img_feat_flatten.device) # (1, 92800, 2)
            else:
                img_feat_pos = self.img_feat_pos

            prev_query_feat = query_feat.detach().clone() # (1, 128, 200)
            query_feat = torch.zeros_like(query_feat)  # create new container for img query feature
            # 计算query的实际位置 --> (1, 2, 200)
            query_pos_realmetric = query_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]
            # (1, 3, 200)
            query_pos_3d = torch.cat([query_pos_realmetric, res_layer['height']], dim=1).detach().clone()
            if 'vel' in res_layer:
                vel = copy.deepcopy(res_layer['vel'].detach())
            else:
                vel = None
            # 解码lidar的预测bbox
            pred_boxes = self.bbox_coder.decode(
                copy.deepcopy(res_layer['heatmap'].detach()),
                copy.deepcopy(res_layer['rot'].detach()),
                copy.deepcopy(res_layer['dim'].detach()),
                copy.deepcopy(res_layer['center'].detach()),
                copy.deepcopy(res_layer['height'].detach()),
                vel,
            ) # List[{'bboxes', 'scores', 'labels'}]

            on_the_image_mask = torch.ones([batch_size, self.num_proposals]).to(query_pos_3d.device) * -1 # (2, 200) 初始化-1

            # 逐帧处理
            for sample_idx in range(batch_size if self.fuse_img else 0):
                # 获取该帧的lidar2img的变换矩阵
                lidar2img_rt = query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img']) # (6, 4, 4)
                # 获取图像的缩放比例
                img_scale_factor = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                            if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])) # [1, 1]
                # 获取图像的翻转信息
                img_flip = img_metas[sample_idx]['flip'] if 'flip' in img_metas[sample_idx].keys() else False # True
                # 获取图像的裁剪信息
                img_crop_offset = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['img_crop_offset'])
                    if 'img_crop_offset' in img_metas[sample_idx].keys() else 0) # 0
                # 获取图像的形状
                img_shape = img_metas[sample_idx]['img_shape'][:2] # (900, 1600)
                # 获取pad后图像的形状
                img_pad_shape = img_metas[sample_idx]['input_shape'][:2] # (928, 1600)
                # 根据lidar预测的boxes构造LiDARInstance3DBoxes
                boxes = LiDARInstance3DBoxes(pred_boxes[sample_idx]['bboxes'][:, :7], box_dim=7) # (200, 7)
                # query_pos_3d[sample_idx]:(3, 200)
                # boxes.corners:(200, 8, 3)-->(3, 200, 8)-->(3, 1600)
                # query的3d位置和角点信息 (3, 200) cat (3, 1600) --> (3, 1800)
                query_pos_3d_with_corners = torch.cat([query_pos_3d[sample_idx], boxes.corners.permute(2, 0, 1).view(3, -1)], dim=-1)  # [3, num_proposals] + [3, num_proposals*8]
                # transform point clouds back to original coordinate system by reverting the data augmentation
                points = apply_3d_transformation(query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()
                num_points = points.shape[0]
                
                # 计算query点坐标后，在6个图像上逐个处理
                for view_idx in range(self.num_views):
                    ########################################################
                    # 1.根据恢复后的真实坐标进行投影计算coord和on_the_image_mask
                    ########################################################
                    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1) # 还原齐次坐标 (1800, 4)
                    pts_2d = pts_4d @ lidar2img_rt[view_idx].t() # 将点投影到图像坐标系下 (1800, 4)
                    # 将点转换到像素坐标系下
                    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
                    pts_2d[:, 0] /= pts_2d[:, 2]
                    pts_2d[:, 1] /= pts_2d[:, 2]

                    # 对图像做反向变换
                    # img transformation: scale -> crop -> flip
                    # the image is resized by img_scale_factor
                    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2 (1800, 2)
                    img_coors -= img_crop_offset

                    # grid sample, the valid grid range should be in [-1,1]
                    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1 (1800, 1)

                    if img_flip:
                        # by default we take it as horizontal flip
                        # use img_shape before padding for flip
                        orig_h, orig_w = img_shape
                        coor_x = orig_w - coor_x
                    
                    # 分离中心坐标和角点坐标：中心坐标用于计算on_the_image_mask, 角点坐标用于计算radius
                    coor_x, coor_corner_x = coor_x[0:self.num_proposals, :], coor_x[self.num_proposals:, :] # (200, 1)和(1600, 1)
                    coor_y, coor_corner_y = coor_y[0:self.num_proposals, :], coor_y[self.num_proposals:, :]
                    coor_corner_x = coor_corner_x.reshape(self.num_proposals, 8, 1) # (200, 8, 1)
                    coor_corner_y = coor_corner_y.reshape(self.num_proposals, 8, 1) # (200, 8, 1)
                    coor_corner_xy = torch.cat([coor_corner_x, coor_corner_y], dim=-1) # (200, 8, 2)

                    h, w = img_pad_shape # (928, 1600)
                    on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h) # 判断中心点是否在图像上 (200, 1) 用于筛选query
                    on_the_image = on_the_image.squeeze() # (200,)
                    # skip the following computation if no object query fall on current image
                    if on_the_image.sum() <= 1:
                        continue
                    on_the_image_mask[sample_idx, on_the_image] = view_idx # 记录有效query所在batch和图像id

                    ################################################
                    # add spatial constraint
                    ################################################
                    # 计算中心点在该图像上的特征点坐标
                    center_ys = (coor_y[on_the_image] / self.out_size_factor_img)
                    center_xs = (coor_x[on_the_image] / self.out_size_factor_img)
                    centers = torch.cat([center_xs, center_ys], dim=-1).int()  # center on the feature map
                    # coor_corner_xy:(200, 8, 2)
                    # coor_corner_xy[on_the_image]:(68, 8, 2)
                    corners = (coor_corner_xy[on_the_image].max(1).values - coor_corner_xy[on_the_image].min(1).values) / self.out_size_factor_img # 8个角点的最大值减去最小值(类似对角线) eg:(68,2)
                    radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()  # radius of the minimum circumscribed circle of the wireframe 求外接圆半径 eg:(68,)
                    sigma = (radius * 2 + 1) / 6.0 # 根据外接圆边界初始化sigma值 (68,)
                    # (68, 1, 2) - (1, 92800, 2) --> (68, 92800, 2) --> (68, 92800)
                    distance = (centers[:, None, :] - (img_feat_pos - 0.5)).norm(dim=-1) ** 2 # 计算中心到图像各位置的距离 eg:(68, 92800)
                    # (68, 92800) / (68, 1)
                    gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp() # eg:(68, 92800)
                    gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0 # 将小值设置为0 (68, 92800)
                    attn_mask = gaussian_mask # 生成高斯mask eg:(68, 92800) --> 权重计算时相加，辅助快速学习attn

                    ################################################
                    # 该帧图像的query和图像特征value进行decoder --> 6帧图像共用一个decoder
                    ################################################
                    # 抽取query feat, 计算query pos
                    query_feat_view = prev_query_feat[sample_idx, :, on_the_image] # query feat (128, 68)
                    query_pos_view = torch.cat([center_xs, center_ys], dim=-1) # query pos (68, 2)
                    # query_feat_view[None]:(1, 128, 68)
                    # img_feat_flatten[sample_idx:sample_idx + 1, view_idx]:(1, 128, 92800)
                    # query_pos_view[None]:(1, 68, 2)
                    # img_feat_pos: (1, 92800, 2)
                    # attn_mask:(68, 92800)
                    query_feat_view = self.decoder[self.num_decoder_layers](
                        query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx], 
                        query_pos_view[None], img_feat_pos, attn_mask=attn_mask.log())
                    # 将cross attn后的query记录到对应位置
                    query_feat[sample_idx, :, on_the_image] = query_feat_view.clone() # 更新对应的query值

            self.on_the_image_mask = (on_the_image_mask != -1) # 基本都为true-->(1, 200)
            # 两个query_feat合并进行预测
            res_layer = self.prediction_heads[self.num_decoder_layers](torch.cat([query_feat, prev_query_feat], dim=1))
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            # 逐个head进行处理，将没有进行增强的query使用原始lidar的预测值
            for key, value in res_layer.items():
                pred_dim = value.shape[1]
                # repeat dim次是组合对应维度的mask
                res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)] = \
                    first_res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)] # (1, 1, 200)
            ret_dicts.append(res_layer)

        if self.initialize_by_heatmap:
            # heatmap:(2, 10, 32400)
            # top_proposals_index:(2, 200) --> (2, 1, 200) --> (2, 10, 200) 取出query在heatmap上的分数 对heatmap的生成做约束
            # query在BEV heatmap上的特征
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            if self.fuse_img:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap_img
            else:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap # (2, 10, 180, 180) 记录dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1) # 如果是多层，会在这里进行拼接，如果只取最后一层不进行拼接
            else:
                new_res[key] = ret_dicts[0][key] 
        return [new_res]

    def forward(self, feats, img_feats, img_metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if img_feats is None:
            img_feats = [None]
        res = multi_apply(self.forward_single, feats, img_feats, [img_metas])
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        # 逐帧处理, 将pred_dict按帧进行重新整理
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            # 逐个head处理
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1] # 取出该帧的预测结果
            list_of_pred_dict.append(pred_dict) # 将该帧结果加入list

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)
        # 逐帧assign target
        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, np.arange(len(gt_labels_3d)))
        # 将各target结果进行拼接
        labels = torch.cat(res_tuple[0], dim=0) # eg:(1, 200) --> (bs, num_proposal)
        label_weights = torch.cat(res_tuple[1], dim=0) # eg:(1, 200)
        bbox_targets = torch.cat(res_tuple[2], dim=0) # eg:(1, 200, 10)
        bbox_weights = torch.cat(res_tuple[3], dim=0) # eg:(1, 200, 10)
        ious = torch.cat(res_tuple[4], dim=0) # eg: (1, 200)
        num_pos = np.sum(res_tuple[5]) # eg:27
        matched_ious = np.mean(res_tuple[6]) # eg:0.0339
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0) # (1, 10, 180, 180)
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap
        else:
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        # ----------------------------------
        # 1.拷贝预测输出并对预测bbox进行编码
        # ----------------------------------
        num_proposals = preds_dict['center'].shape[-1] # 200

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach()) # (1, 10, 200)
        center = copy.deepcopy(preds_dict['center'].detach()) # (1, 2, 200)
        height = copy.deepcopy(preds_dict['height'].detach()) # (1, 1, 200)
        dim = copy.deepcopy(preds_dict['dim'].detach()) # (1, 3, 200)
        rot = copy.deepcopy(preds_dict['rot'].detach()) # (1, 2, 200)
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach()) # (1, 2, 200)
        else:
            vel = None

        # decode the prediction to real world metric bbox
        # boxes_dict:{'bboxes', 'scores', 'labels'}
        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  

        # ------------------------------------------------
        # 2.根据pred boxes和gt boxes进行target assign和sample
        # ------------------------------------------------
        bboxes_tensor = boxes_dict[0]['bboxes'] # (200, 9)
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device) # eg:(78, 9)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1
        # ------------------------------------
        # 2.1 assign target
        # ------------------------------------
        assign_result_list = []
        for idx_layer in range(num_layer):
            # pred boxes的几何参数和分类分数
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :] # (200, 9)
            score_layer = score[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)] # (1, 10, 200)

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]), # 合并gt的数量
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]), # 合并gt inds
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]), # 合并iou
            labels=torch.cat([res.labels for res in assign_result_list]), # 合分配的lables
        )

        # ------------------------------------
        # 2.2 pseudo sampler
        # ------------------------------------
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds # 预测bbox中正例bbox的索引 eg:78
        neg_inds = sampling_result.neg_inds # 预测bbox中负例bbox的索引 eg:112
        assert len(pos_inds) + len(neg_inds) == num_proposals
        
        # ------------------------------------
        # 3.create target for loss computation
        # ------------------------------------
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device) # (200, 10)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device) # (200, 10)
        ious = assign_result_ensemble.max_overlaps # (200,)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) # (200,)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) # (200,)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes # labels全部为初始化为10

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            # 对gt进行编码
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes) # (78, 10) 对gt进行编码

            bbox_targets[pos_inds, :] = pos_bbox_targets # 将编码后的gt box赋予pos inds的对应位置
            bbox_weights[pos_inds, :] = 1.0 # 将权重设置为1

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds] # 将gt label赋予正例的位置
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0 # 将正例的权重设置为1
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0 # 此时lable_weights全1
        
        # ----------------------------------
        # 4.compute dense heatmap targets
        # ----------------------------------
        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device) # (78, 10)
            grid_size = torch.tensor(self.train_cfg['grid_size']) # (1440, 1440, 40)
            pc_range = torch.tensor(self.train_cfg['point_cloud_range']) # [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
            voxel_size = torch.tensor(self.train_cfg['voxel_size']) # [0.075, 0.075, 0.2]
            feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len] (180, 180)
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0]) # 初始化(10, 180, 180)
            # 逐个gt bbox处理,计算heatmap
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3] # bbox的实际宽
                length = gt_bboxes_3d[idx][4] # bbox的实际长
                width = width / voxel_size[0] / self.train_cfg['out_size_factor'] # bbox的特征图宽
                length = length / voxel_size[1] / self.train_cfg['out_size_factor'] # bbox的特征图长
                if width > 0 and length > 0:
                    # 根据bbox的长和宽计算高斯半径
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap']) # gaussian_overlap:0.1
                    radius = max(self.train_cfg['min_radius'], int(radius)) # min_radius:2
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1] # gt bbox的实际中心坐标

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor'] # 计算bbox的特征中心坐标
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    # 在该类别所在的heatmap上根据中心点和半径绘制gaussian圆
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) 

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1) # 计算iou的平均值
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]

        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        if self.initialize_by_heatmap:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        else:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask # eg: (1, 200)
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None] # (1, 200, 10)
            num_pos = bbox_weights.max(-1).values.sum() # eg:27
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()
        # -------------------------------------
        # image guided query initialization 注释
        # -------------------------------------
        if self.initialize_by_heatmap:
            # compute heatmap loss GaussianFocalLoss
            # preds_dict['dense_heatmap']: (1, 10, 180, 180)
            # heatmap: (1, 10, 180, 180)
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer 只有一层
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'
            
            # 计算分类损失
            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1) # (400,) 
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1) # （400,)
            layer_score = preds_dict['heatmap'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals] # (2, 10, 200)
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes) # (2, 10, 200)-->(2, 200, 10)-->(400, 10)
            # 进入分类损失函数 focal loss
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos, 1))

            # 计算回归损失
            layer_center = preds_dict['center'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals] # (2, 2, 200)
            layer_height = preds_dict['height'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals] # (2, 1, 200)
            layer_rot = preds_dict['rot'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals] # (2, 2, 200)
            layer_dim = preds_dict['dim'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals] # (2, 3, 200)
            # (2, 8, 200)-->(2, 200, 8)
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals] # (2, 2, 200)
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]-->(2, 200, 10)
            code_weights = self.train_cfg.get('code_weights', None) # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            layer_bbox_weights = bbox_weights[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :] # (2, 200, 10)
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights) # (2, 200, 10)
            layer_bbox_targets = bbox_targets[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :] # (2, 200, 10)
            # 进入回归损失函数 L1 loss
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls # 记录分类损失
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox # 记录回归损失
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious) # 这里只有记录matched_ious，并不会反传，因为单纯计算没有grad_fn

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        # 逐层处理
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0] # 计算batch size
            batch_score = preds_dict[0]['heatmap'][..., -self.num_proposals:].sigmoid() # (1, 10, 200)
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1) # 提取query的分类类别并进行one hot编码
            batch_score = batch_score * preds_dict[0]['query_heatmap_score'] * one_hot # 融合初始预测和最终预测分数 (1, 10, 200)
            
            # 提取预测几何参数
            batch_center = preds_dict[0]['center'][..., -self.num_proposals:] # (1, 2, 200)
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:] # (1, 1, 200)
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:] # (1, 3, 200)
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:] # (1, 2, 200)
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:] # (1, 2, 200)

            # 对预测的bbox进行解码
            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ] # 对行人和锥筒单独处理
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'], indices=[0], radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[1], radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'], indices=[2], radius=0.7),
                ]

            ret_layer = []
            # 逐帧处理
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes'] # (200, 9)
                scores = temp[i]['scores'] # (200,)
                labels = temp[i]['labels'] # (200,)
                ## adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores) # 初始化保持mask
                    # 逐个任务处理
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                # circle_nms
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1) # 提取中心点和分数
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                ) # 进行circle_nms计算被keep的bbox的索引
                            else:
                                # num_gpu
                                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev) # 构造LidarInstanceBox后转换bev
                                top_scores = scores[task_mask] # 提取对应分数
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize'],
                                ) # 进行NMS计算被keep的bbox的索引
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool() # 保留下的mask
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        # 构造输出结果
        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res
