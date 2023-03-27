import copy
import warnings
import torch
import torch.nn as nn
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.core import LiDARInstance3DBoxes
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

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
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout) # self attention
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout) # cross attention 多头注意力机制
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
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
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
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

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


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
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

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
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
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
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

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
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
            conv_layers = nn.Sequential(*conv_layers) # 以head命名变量并赋值

            self.__setattr__(head, conv_layers)

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


class DynamicConv(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = 128
        self.dim_dynamic = 128
        self.num_dynamic = 2
        self.num_params = self.hidden_dim * self.dim_dynamic # 128*128
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params) # (128, 2*128*128)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU()

        pooler_resolution = 7
        num_output = self.hidden_dim * pooler_resolution ** 2 # 128*7^2=6272
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2) # (49, 39, 128)-->(39, 49, 128)
        # pro_features:(1, 39, 128)-->(1, 39, 32768)-->(39, 1, 32768)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2) # 利用query学习两组注意力参数，更新roi特征

        # (39, 1, 16384)-->(39, 128, 128)
        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        # (39, 49, 128)*(39, 128, 128)-->(39, 49, 128)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        # (39, 49, 128)-->(39, 6272)
        features = features.flatten(1)
        features = self.out_layer(features) # (39, 128)
        features = self.norm3(features)
        features = self.activation(features)

        return features # (39, 128)


class ImageRCNNBlock(nn.Module):

    def __init__(self, num_views, num_proposals, out_size_factor_img, test_cfg, bbox_coder, hidden_channel, num_heads, dropout):
        super(ImageRCNNBlock, self).__init__()
        self.num_views = num_views # 6
        self.num_proposals = num_proposals # 200
        self.out_size_factor_img = out_size_factor_img # 4
        self.test_cfg = test_cfg
        self.bbox_coder = bbox_coder # TransFusionBBoxCoder
        self.pooler = ROIPooler(
            output_size=7,
            scales=[1.0 / self.out_size_factor_img, ], # 0.25
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        ) # detectron2中的类
        self.dyconv = DynamicConv(None)
        self.dyconv_pre_self_attn = nn.MultiheadAttention(hidden_channel, num_heads, dropout=dropout) # (128, 8, 0.1)
        self.dropout1 = nn.Dropout(dropout) # 0.1
        self.norm1 = nn.LayerNorm(hidden_channel) 
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_channel)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_channel)
        self.linear1 = nn.Linear(hidden_channel, hidden_channel*4) # (128, 128*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_channel*4, hidden_channel) # (128*4, 128)
        self.activation = nn.GELU()

    def forward(self, query_feat, res_layer, new_lidar_feat, bev_pos, img_feat_flatten, img_feat_pos, img_metas, img_h, img_w, **kwargs):
        # query_feat:(1, 128, 200)
        # res_layer:预测结果 {'center', 'height', 'dim', 'rot', 'vel', 'heatmap'}
        # img_feat_flatten:(1, 6, 128, 22400)
        batch_size = query_feat.shape[0] # 1
        query_pos = res_layer['center'].detach().clone().permute(0, 2, 1) # (1, 200, 2)
        prev_query_feat = query_feat # (1, 128, 200)
        prev_res_layer = res_layer
        query_feat = torch.zeros_like(query_feat)
        # 计算query的实际位置 --> (1, 2, 200)
        query_pos_realmetric = query_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]
        query_pos_3d = torch.cat([query_pos_realmetric, res_layer['height']], dim=1).detach().clone() # (1, 3, 200)
        if 'vel' in res_layer:
            vel = copy.deepcopy(res_layer['vel'].detach())
        else:
            vel = None
        pred_boxes = self.bbox_coder.decode(
                copy.deepcopy(res_layer['heatmap'].detach()),
                copy.deepcopy(res_layer['rot'].detach()),
                copy.deepcopy(res_layer['dim'].detach()),
                copy.deepcopy(res_layer['center'].detach()),
                copy.deepcopy(res_layer['height'].detach()),
                vel,
            ) # 解码lidar的预测bbox List[{'bboxes', 'scores', 'labels'}]
        on_the_image_mask = torch.ones([batch_size, self.num_proposals]).to(query_pos_3d.device) * -1 # (1, 200) 全-1初始化
        # 逐帧处理
        for sample_idx in range(batch_size):
            lidar2img_rt = query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img']) # 获取该帧的lidar2img的变换矩阵
            img_scale_factor = (query_pos_3d.new_tensor([1.0, 1.0])) # 获取图像的缩放比例
            img_flip = img_metas[sample_idx]['flip'] if 'flip' in img_metas[sample_idx].keys() else False # 获取图像的翻转信息
            img_crop_offset = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['img_crop_offset'])
                    if 'img_crop_offset' in img_metas[sample_idx].keys() else 0) # 获取图像的裁剪信息
            img_shape = img_metas[sample_idx]['img_shape'][0][:2] # 获取图像的形状 (448, 800)
            img_pad_shape = img_metas[sample_idx]['input_shape'][:2] # 获取pad后图像的形状
            boxes = LiDARInstance3DBoxes(pred_boxes[sample_idx]['bboxes'][:, :7], box_dim=7) # 根据lidar预测的boxes构造LiDARInstance3DBoxes
            # query_pos_3d[sample_idx]:(3, 200)
            # boxes.corners:(200, 8, 3)-->(3, 200, 8)-->(3, 1600)
            # query的3d位置和角点信息 (3, 200) cat (3, 1600) --> (3, 1800)
            query_pos_3d_with_corners = torch.cat([query_pos_3d[sample_idx], boxes.corners.permute(2, 0, 1).view(3, -1)], dim=-1) # [3, num_proposals] + [3, num_proposals*8]
            points = apply_3d_transformation(query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()
            num_points = points.shape[0] # 1800

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
                img_coors = pts_2d[:, 0:2] * img_scale_factor # Nx2 (1800, 2)
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

                h, w = img_pad_shape # 448, 800
                on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h) # 判断中心点是否在图像上 (200, 1) 用于筛选query
                on_the_image = on_the_image.squeeze() # (200,)
                # skip the following computation if no object query fall on current image
                if on_the_image.sum() <= 1:
                    continue
                on_the_image_mask[sample_idx, on_the_image] = view_idx # 记录有效query所在batch和图像id
                
                # add spatial constraint
                circumscribed_rectangle_on_feature_max_height = coor_corner_xy[on_the_image, :, 1].max(1).values # 取8个角点高度的最大值 eg:(39,)
                circumscribed_rectangle_on_feature_max_width = coor_corner_xy[on_the_image, :, 0].max(1).values # 取8个角点宽度的最大值 eg:(39,)
                circumscribed_rectangle_on_feature_min_height = coor_corner_xy[on_the_image, :, 1].min(1).values # 取8个角点高度的最小值 eg:(39,)
                circumscribed_rectangle_on_feature_min_width = coor_corner_xy[on_the_image, :, 0].min(1).values # 取8个角点宽度的最小值 eg:(39,)

                circumscribed_rectangle_on_feature_coor = torch.stack([circumscribed_rectangle_on_feature_min_width,
                                                                        circumscribed_rectangle_on_feature_min_height,
                                                                        circumscribed_rectangle_on_feature_max_width,
                                                                        circumscribed_rectangle_on_feature_max_height], dim=1) # (39, 4)
                # @param: img_feat_flattenL:[1, 6, 128, 22400]-->(1, 1, 112, 200)
                # @param: circumscribed_rectangle_on_feature_coor eg:(39, 4)
                # @return: roi_features:(39, 128, 7, 7)
                roi_features = self.pooler([img_feat_flatten[sample_idx:sample_idx + 1,
                                                                        view_idx].reshape(1, -1, img_h, img_w)],
                                            [Boxes(circumscribed_rectangle_on_feature_coor)])
                
                # 0.准备lidar query特征和image roi特征
                # prev_query_feat:(1, 128, 200)-->(128, 39)-->(1, 128, 39)-->(39, 1, 128)
                query_feat_view = prev_query_feat[sample_idx, :, on_the_image][None].permute(2, 0, 1)
                # (39, 128, 7, 7)-->(39, 128, 49)-->(49, 39, 128)
                roi_features = roi_features.flatten(2).permute(2, 0, 1)

                # 1.query_feat_view之间自注意力机制-->(39, 1, 128) [1]是注意力权重
                query_feat_view2 = self.dyconv_pre_self_attn(query_feat_view, query_feat_view, value=query_feat_view)[0]
                query_feat_view = query_feat_view + self.dropout1(query_feat_view2)
                query_feat_view = self.norm1(query_feat_view)

                # 2.融合点云roi信息
                query_feat_view = query_feat_view.permute(1, 0, 2) # (1, 39, 128)
                query_feat_view2 = self.dyconv(query_feat_view, roi_features) # (39, 128) 用query feat学习权重更新RoI特征
                query_feat_view = query_feat_view + self.dropout2(query_feat_view2)
                query_feat_view = self.norm2(query_feat_view)

                # 3.FFN进一步融合
                query_feat_view2 = self.linear2(self.dropout(self.activation(self.linear1(query_feat_view))))
                query_feat_view = query_feat_view + self.dropout3(query_feat_view2)
                query_feat_view = self.norm3(query_feat_view) # (1, 39, 128)

                query_feat_view = query_feat_view[0].permute(1, 0) # [128, 39]
                query_feat[sample_idx, :, on_the_image] = query_feat_view.clone()  # 1, 128, 200 重叠区域没有特殊处理 只是用后一视图的信息进行覆盖

        return query_feat, on_the_image_mask # (1, 128, 200)和(1, 200)


class PointRCNNBlock(nn.Module):
    def __init__(self, hidden_channel, num_heads, dropout, bbox_coder):
        super(PointRCNNBlock,self).__init__()
        self.bbox_coder = bbox_coder # TransFusionBBoxCoder
        self.pooler_pts = ROIPooler(
            output_size=7,
            scales=[1.0 / 1, ],
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        ) # ROIAlignV2
        self.dyconv_pts = DynamicConv(None)
        self.dyconv_pre_self_attn_pts = nn.MultiheadAttention(hidden_channel, num_heads, dropout=dropout) # (128, 8, 0.1)
        self.dropout1_pts = nn.Dropout(dropout) # 0.1
        self.norm1_pts = nn.LayerNorm(hidden_channel) # 128
        self.dropout2_pts = nn.Dropout(dropout)
        self.norm2_pts = nn.LayerNorm(hidden_channel)
        self.dropout3_pts = nn.Dropout(dropout)
        self.norm3_pts = nn.LayerNorm(hidden_channel)
        self.linear1_pts = nn.Linear(hidden_channel, hidden_channel*4)
        self.dropout_pts = nn.Dropout(dropout)
        self.linear2_pts = nn.Linear(hidden_channel*4, hidden_channel)
        self.activation_pts = nn.GELU()

    def forward(self, query_feat, res_layer, new_lidar_feat, bev_pos, img_feat_flatten, img_feat_pos, img_metas, img_h, img_w, **kwargs):
        batch_size = query_feat.shape[0] # 1
        prev_query_feat = query_feat # (1, 128, 200)
        query_feat = torch.zeros_like(query_feat)
        if 'vel' in res_layer:
            vel = copy.deepcopy(res_layer['vel'].detach())
        else:
            vel = None
        pred_boxes = self.bbox_coder.decode(
                copy.deepcopy(res_layer['heatmap'].detach()),
                copy.deepcopy(res_layer['rot'].detach()),
                copy.deepcopy(res_layer['dim'].detach()),
                copy.deepcopy(res_layer['center'].detach()),
                copy.deepcopy(res_layer['height'].detach()),
                vel,
            ) # 解码lidar的预测bbox List[{'bboxes', 'scores', 'labels'}]
        corners = []
        for sample_idx in range(batch_size):
            box = pred_boxes[sample_idx]['bboxes'][:, :7]
            box[:,3:6] *= 2 # 将lidar box扩大2倍，因为BEV视角物体都较小
            corners.append(LiDARInstance3DBoxes(box, box_dim=7).corners)
        corners = torch.stack(corners, 0) # (1, 200, 8, 3) 8个角点
        # (1, 200, 8, 2) 计算bev视角下特征图坐标
        corners_coor = ((corners[...,:2] - self.bbox_coder.pc_range[0]) / (self.bbox_coder.voxel_size[0] * self.bbox_coder.out_size_factor))
        circumscribed_rectangle_on_feature_max_height = corners_coor[...,1].max(-1).values # (1, 200)
        circumscribed_rectangle_on_feature_max_width = corners_coor[...,0].max(-1).values
        circumscribed_rectangle_on_feature_min_height = corners_coor[...,1].min(-1).values
        circumscribed_rectangle_on_feature_min_width = corners_coor[...,0].min(-1).values
        # (1, 200, 4)
        circumscribed_rectangle_on_feature_coor = torch.stack([circumscribed_rectangle_on_feature_min_width,
                                                                           circumscribed_rectangle_on_feature_min_height,
                                                                           circumscribed_rectangle_on_feature_max_width,
                                                                           circumscribed_rectangle_on_feature_max_height], dim=-1)
        # 逐帧处理
        for sample_idx in range(batch_size):
            # new_lidar_feat：(1, 128, 180, 180)
            roi_features = self.pooler_pts([new_lidar_feat[sample_idx:sample_idx + 1]],
                                   [Boxes(circumscribed_rectangle_on_feature_coor[sample_idx])]) # (200, 128, 7, 7)
            query_feat_view = prev_query_feat[sample_idx:sample_idx + 1].permute(2, 0, 1) # [1, 128, 200]
            # (200, 128, 49)-->(49, 200, 128)
            roi_features = roi_features.flatten(2).permute(2, 0, 1) # [49, N, C]

            # 1.query_feat_view之间自注意力机制-->(39, 1, 128) [1]是注意力权重
            query_feat_view2 = self.dyconv_pre_self_attn_pts(query_feat_view, query_feat_view, value=query_feat_view)[0] # (200, 1, 128)
            query_feat_view = query_feat_view + self.dropout1_pts(query_feat_view2)
            query_feat_view = self.norm1_pts(query_feat_view)

            # 2.融合图像roi信息
            query_feat_view = query_feat_view.permute(1, 0, 2) # [1, 23, 128]
            query_feat_view2 = self.dyconv_pts(query_feat_view, roi_features) # [23, 128]
            query_feat_view = query_feat_view + self.dropout2_pts(query_feat_view2)
            query_feat_view = self.norm2_pts(query_feat_view)

            # 3.FFN进一步融合
            query_feat_view2 = self.linear2_pts(self.dropout_pts(self.activation_pts(self.linear1_pts(query_feat_view))))
            query_feat_view = query_feat_view + self.dropout3_pts(query_feat_view2)
            query_feat_view = self.norm3_pts(query_feat_view)

            query_feat[sample_idx, : , :] = query_feat_view.permute(0,2,1)[0] # (1, 128, 200)

        return query_feat, None # (1, 128, 200)
