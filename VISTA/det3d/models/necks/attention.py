# Copyright (c) Gorilla-Lab. All rights reserved.

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..registry import NECKS


class ConvAttentionLayer_Decouple(nn.Module):
    def __init__(self, input_channels: int, numhead: int = 1, reduction_ratio=2):
        r"""
        Args:
            input_channels (int): input channel of conv attention 384
            numhead (int, optional): the number of attention heads. Defaults to 1.
        """
        super().__init__()
        # self.q_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, 3, 1, 1)
        self.q_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1) # 384, 192
        self.q_geo_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1) # 384, 192
        # self.k_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, 3, 1, 1)
        self.k_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1) # 384, 192
        self.k_geo_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1) # 384, 192
        # self.v_conv = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
        self.v_conv = nn.Conv2d(input_channels, input_channels, 1, 1, 0) # 384, 384
        self.out_sem_conv = nn.Conv2d(input_channels, input_channels, 1, 1) # 384, 384
        self.out_geo_conv = nn.Conv2d(input_channels, input_channels, 1, 1) # 384, 384
        self.softmax = nn.Softmax(dim=-1)
        self.channels = input_channels // reduction_ratio # 192
        self.numhead = numhead # 1
        self.head_dim = self.channels // numhead # 192
        self.sem_norm = nn.LayerNorm(input_channels) # 384
        self.geo_norm = nn.LayerNorm(input_channels) # 384

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                q_pos_emb: Optional[torch.Tensor] = None,
                k_pos_emb: Optional[torch.Tensor] = None):
        r"""
        Args:
            query (torch.Tensor, [B, C, H_qk, W_qk]): feature of query （4, 384, 40, 40)
            key (torch.Tensor, [B, C, H_qk, W_qk]): feature of key （4, 384, 40, 12)
            value (torch.Tensor, [B, C, H_v, W_v]): feature of value （4, 384, 160, 12)
            q_pos_emb (Optional[torch.Tensor], optional, [[B, C, H_q, W_q]]): (4, 384, 40, 40)
                positional encoding. Defaults to None.
            k_pos_emb (Optional[torch.Tensor], optional, [[B, C, H_kv, W_kv]]): (4, 384, 40, 12)
                positional encoding. Defaults to None.
        """

        view = query + 0  # NOTE: a funny method to deepcopy --> (4, 384, 40, 40)
        input_channel = view.shape[1] # 384
        if q_pos_emb is not None:
            query += q_pos_emb # positional encoding相加 --> (4, 384, 40, 40)
        if k_pos_emb is not None:
            key += k_pos_emb # positional encoding相加 --> (4, 384, 40, 12)

        # to qkv forward
        # q = self.q_conv(query)
        q_sem = self.q_sem_conv(query) # (4, 192, 40, 40)
        q_geo = self.q_geo_conv(query) # (4, 192, 40, 40)
        qs = [q_sem, q_geo] # 特征拼接 List[Tensor]
        # k = self.k_conv(key)
        k_sem = self.k_sem_conv(key) # (4, 192, 40, 12) 
        k_geo = self.k_geo_conv(key) # (4, 192, 40, 12) 
        ks = [k_sem, k_geo] # 特征拼接 List[Tensor]
        v = self.v_conv(value) # (4, 384, 40, 12) 
        vs = [v, v] # 特征拼接 List[Tensor]
        out_convs = [self.out_sem_conv, self.out_geo_conv] # 卷积层拼接
        norms = [self.sem_norm, self.geo_norm] # 正则化层拼接
        outputs = []
        attentions = []

        # 逐个处理sem和geo
        for q, k, v, out_conv, norm in zip(qs, ks, vs, out_convs, norms):
            # read shape of qkv
            bs = q.shape[0] # 4
            qk_channel = q.shape[1]  # equal to the channel of `k` 192
            v_channel = v.shape[1]  # channel of `v` 384
            h_q, w_q = q.shape[2:]  # height and weight of query map 40, 40
            h_kv, w_kv = k.shape[2:]  # height and weight of key and value map 40, 12
            numhead = self.numhead # 1
            qk_head_dim = qk_channel // numhead # 192
            v_head_dim = v_channel // numhead # 384

            # scale query
            scaling = float(self.head_dim) ** -0.5 # 0.07216
            q = q * scaling # (4, 192, 40, 40)

            # reshape(sequentialize) qkv
            q = rearrange(q, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_q, w=w_q) # (4, 192, 1600)
            q = rearrange(q, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                          n=numhead, h=h_q, w=w_q, d=qk_head_dim) # (4, 1600, 192)
            q = q.contiguous()
            k = rearrange(k, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_kv, w=w_kv) # (4, 192, 480)
            k = rearrange(k, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                          n=numhead, h=h_kv, w=w_kv, d=qk_head_dim) # (4, 480, 192)
            k = k.contiguous()
            v = rearrange(v, "b c h w -> b c (h w)", b=bs, c=v_channel, h=h_kv, w=w_kv) # (4, 384, 480)
            v = rearrange(v, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                          n=numhead, h=h_kv, w=w_kv, d=v_head_dim) # (4, 480, 384)
            v = v.contiguous()

            # get the attention map (4, 1600, 192) x (4, 192, 480) --> (4, 1600, 480)
            energy = torch.bmm(q, k.transpose(1, 2))  # [h_q*w_q, h_kv*w_kv]
            attention = F.softmax(energy, dim=-1)  # [h_q*w_q, h_kv*w_kv]
            attentions.append(attention) # 记录attention
            # get the attention output
            # (4, 1600, 480) x (4, 480, 384) --> (4, 1600, 384)
            r = torch.bmm(attention, v)  # [bs * nhead, h_q*w_q, C'] --> (4, 1600, 384)
            r = rearrange(r, "(b n) (h w) d -> b (n d) h w", b=bs,
                          n=numhead, h=h_q, w=w_q, d=v_head_dim) # (4, 384, 40, 40)
            r = r.contiguous()
            r = out_conv(r) # (4, 384, 40, 40)

            # residual
            temp_view = view + r # (4, 384, 40, 40) 残差结构
            # (1600, 4, 384) 将batch size放到第1维度，为layer norm做准备
            temp_view = temp_view.view(bs, input_channel, -1).permute(2, 0, 1).contiguous() 
            temp_view = norm(temp_view) # layer norm处理(似乎因为上面的处理变为batch norm了)
            outputs.append(temp_view) # 将结果加入输出
        return outputs, attentions # List[Tenosr] (1600, 4, 384), (4, 1600, 480)


class FeedFowardLayer(nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channel: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, hidden_channel) # 384, 512
        self.linear2 = nn.Linear(hidden_channel, input_channel) # 512, 384

        self.norm = nn.LayerNorm(input_channel) # 384

        self.activation = nn.ReLU()  # con be modify as GeLU or GLU

    def forward(self, view: torch.Tensor):
        ffn = self.linear2(self.activation((self.linear1(view)))) # (1600, 4, 384)
        view = view + ffn # (1600, 4, 384)
        view = self.norm(view) # (1600, 4, 384)
        return view # (1600, 4, 384)


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, res: Sequence[int], num_pos_feats=384):
        r"""
        Absolute pos embedding, learned.
        Args:
            res (Sequence[int]): resolution (height and width)
            num_pos_feats (int, optional): the number of feature channel. Defaults to 384.
        """
        super().__init__()
        h, w = res # 40, 40
        num_pos_feats = num_pos_feats // 2 # 192
        self.row_embed = nn.Embedding(h, num_pos_feats) # (40, 192)
        self.col_embed = nn.Embedding(w, num_pos_feats) # (40, 192)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:] # 40， 40
        i = torch.arange(w, device=x.device) # (40,)
        j = torch.arange(h, device=x.device) # (40,)
        x_emb = self.col_embed(i) # (40, 192)
        y_emb = self.row_embed(j) # (40, 192)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1), # (40, 40, 192)
            y_emb.unsqueeze(1).repeat(1, w, 1), # (40, 40, 192)
        ], dim=-1).permute(2, 0, 1).contiguous().unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # (40, 40, 384)-->(384, 40, 40)-->(1, 384, 40, 40)-->(4, 384, 40, 40)
        return pos


class CrossAttenBlock_Decouple(nn.Module):
    def __init__(self,
                 input_channels: int, # 384
                 numhead: int = 1, # 1
                 hidden_channel: int = 2048, # 512
                 reduction_ratio=2):
        r"""
        Block of cross attention (one cross attention layer and one ffn)
        Args:
            input_channels (int): input channel of conv attention 384
            numhead (int, optional): the number of attention heads. Defaults to 1.
            hidden_channel (int, optional): channel of ffn. Defaults to 2048.
        """
        super().__init__()
        self.cross_atten = ConvAttentionLayer_Decouple(input_channels, numhead, reduction_ratio) # 384, 1, 2
        self.ffn_sem = FeedFowardLayer(input_channels, hidden_channel) # 384, 512
        self.ffn_geo = FeedFowardLayer(input_channels, hidden_channel) # 384, 512

    def forward(self,
                view_1: torch.Tensor,
                view_2: torch.Tensor,
                pos_emb_1: Optional[torch.Tensor] = None,
                pos_emb_2: Optional[torch.Tensor] = None):
        """
            bev_feat_block: （4, 384, 40, 40)
            rv_feat_block: （4, 384, 40, 12)
            bev_pos: (4, 384, 40, 40)
            rv_pos: (4, 384, 40, 12)
        """
        B, C, H, W = view_1.shape # 4, 384, 40, 40
        views, atten_maps= self.cross_atten(view_1, view_2, view_2, pos_emb_1, pos_emb_2) # List[Tenosr] (1600, 4, 384), (4, 1600, 480)
        ffns = [self.ffn_sem, self.ffn_geo] # ffc层
        outputs = []
        # 逐个处理sem和geo
        for i in range(len(views)):
            ffn = ffns[i]
            view = ffn(views[i]) # (1600, 4, 384)
            # (1600, 4, 384) --> (40, 40, 4, 384) --> (4, 384, 40, 40)
            view = view.view(H, W, B, C).permute(2, 3, 0, 1).contiguous()
            outputs.append(view) 
        return outputs, atten_maps # List[Tensor] (4, 384, 40, 40)和(4, 1600, 480)

@NECKS.register_module
class Cross_Attention_Decouple(nn.Module):
    def __init__(self,
                 bev_input_channel: int, # 384
                 rv_input_channel: int, # 384
                 embed_dim: int, # 384
                 num_heads: int, # 1
                 bev_size: Sequence[int] = (262, 64), # (160, 160)
                 bev_block_res: Sequence[int] = (16, 16), # (40, 40)
                 rv_size: Sequence[int] = (262, 64), # (160, 12)
                 rv_block_res: Sequence[int] = (16, 16), # (40, 12)
                 hidden_channels: Sequence[int] = 1024): # 512
        r"""
        Convolutional Cross-View Transformer module
        Args:
            bev_input_channel (int): input channels of bev feature
            rv_input_channel (int): input channels of rv feature
            embed_dim (int): channels of downsample(input for attention)
            num_heads (int): the number of attention heads
            num_conv (int, optional): the number of convolutional layers. Defaults to 5.
            bev_size (Sequence[int], optional): size of bev feature map. Defaults to (262, 64).
            bev_block_res (Sequence[int], optional): size of bev feature map. Defaults to (16, 16).
            rv_size (Sequence[int], optional): size of bev feature map. Defaults to (262, 64).
            rv_block_res (Sequence[int], optional): size of rv feature map. Defaults to (16, 16).
            hidden_channels (Sequence[int], optional): channels of ffn. Defaults to [512, 1024].
            block_feat_mode (str, optional): the manner to obtain the block-wise feature
                                             "AVERAGE" : the block feature will be the average feature inside the block
                                             "CONV" : use conv layer to obtain the block feature. Defaults to "AVERAGE".
        """
        super().__init__()
        assert len(bev_block_res) == 2
        self.bev_input_channel = bev_input_channel # 384
        self.rv_input_channel = rv_input_channel # 384
        self.embed_dim = embed_dim # 384
        self.num_heads = num_heads # 1
        self.bev_size = bev_size # (160, 160)
        self.rv_size = rv_size # (160, 12)
        self.bev_block_res = self.adjust_res(bev_block_res, bev_size) # (40, 40)和(160, 160)
        self.rv_block_res = self.adjust_res(rv_block_res, rv_size) # (40, 12)和(160, 12)

        # positional encoding
        self.bev_pos_emb = PositionEmbeddingLearned(self.bev_block_res, embed_dim) # (40, 40)和(160, 160) 384
        self.rv_pos_emb = PositionEmbeddingLearned(self.rv_block_res, embed_dim) # (40, 12)和(160, 12) 384

        self.cross_atten = CrossAttenBlock_Decouple(embed_dim, num_heads, hidden_channels) # 384, 1, 512

    def adjust_res(self, block_res: Sequence[int], size: Sequence[int]):
        h, w = block_res[0], block_res[1]
        H, W = size[0], size[1]
        assert H % h == 0, f"H must be divisible by h, but got H-{H}, h-{h}"
        assert W % w == 0, f"H must be divisible by h, but got H-{W}, h-{w}"
        h_step = math.ceil(H / h)
        w_step = math.ceil(W / w)
        h_length = math.ceil(H / h_step)
        w_length = math.ceil(W / w_step)
        return (h_length, w_length)

    def generate_feat_block(self, feat_map: torch.Tensor, block_res: Sequence[int]):
        R"""
        generate feat block and the size of each block by scatter_mean
        Args:
            feat_map (torch.Tensor, [B, C, H, W]): input feature map
            block_res  (Sequence[int]): (block_x, block_y)
        Returns:
            block_feat_map (torch.Tensor, [B, C, x_block, y_block]): feature map of blocks
            kernel_size: (Tuple[int]) avg pooling kernel size
        """
        H, W = feat_map.shape[-2:] # 160, 160
        kernel_size = (int(H / block_res[0]), int(W / block_res[1])) # (4, 4)
        block_feat_map = F.avg_pool2d(feat_map, kernel_size, kernel_size) # (4, 384, 40, 40)

        return block_feat_map, kernel_size # (4, 384, 40, 40)和(4, 4)

    def residual_add(self,
                     feat_map: torch.Tensor,
                     attn_block: torch.Tensor,
                     kernel_size: Sequence[int]) -> torch.Tensor:
        r"""
        residual realization briefly
        Args:
            feat_map (torch.Tensor, [B, D, H, W]): origin feature map  (4, 384, 160, 160)
            attn_block (torch.Tensor, [B, D, h_block, w_block]): attention block feature map (4, 384, 40, 40)
            kernel_size  (4, 4)
        Returns:
            torch.Tensor, [B, D, H, W]: origin feature map add attention back prjection feature map
        """
        h_backproj_feature_new = torch.repeat_interleave(attn_block, kernel_size[0], dim=2) # (4, 384, 160, 40)
        hw_backproj_feature = torch.repeat_interleave(h_backproj_feature_new, kernel_size[1], dim=3) # (4, 384, 160, 160)
        # output = torch.cat([feat_map, hw_backproj_feature], 1)
        output = feat_map + hw_backproj_feature # (4, 384, 160, 160)

        return output

    def forward(self, x: Sequence[torch.Tensor]):
        r"""
        Args:
            x (Sequence[torch.Tensor]):
                bev_feat_map_up (torch.Tensor, [B, embed_dim, H1, W1]), feature map of bev
                rvv_feat_map_up (torch.Tensor, [B, embed_dim, H2, W2]), feature map of rv
        """
        (bev_feat_map, rv_feat_map) = x # （4, 384, 160, 160)和(4, 384, 160, 12)

        assert (*bev_feat_map.shape[-2:],) == tuple(self.bev_size), (f"get the size of bev feature map - {bev_feat_map.shape[-2:]}, "
                                                                     f"which does not match the given size {self.bev_size}")
        assert (*rv_feat_map.shape[-2:],) == tuple(self.rv_size), (f"get the size of rv feature map - {rv_feat_map.shape[-2:]}, "
                                                                   f"which does not match the given size {self.rv_size}")

        # generate feature block of bev (for attention) --> (4, 384, 40, 40)
        bev_feat_block, bev_kernel_size = self.generate_feat_block(
            bev_feat_map, self.bev_block_res) # (40, 40)和(160, 160) （4, 384, 160, 160)

        # generate feature block of rv (for attention) --> (4, 384, 40 , 12)
        rv_feat_block, rv_kernel_size = self.generate_feat_block(
            rv_feat_map, self.rv_block_res) # (40, 12)和(160, 12) （4, 384, 160, 12)

        # generate positional encoding
        bev_pos = self.bev_pos_emb(bev_feat_block) # (4, 384, 40, 40)
        rv_pos = self.rv_pos_emb(rv_feat_block) # (4, 384, 40, 12)
        bev_output_feat_maps = []

        """
        bev_feat_block: （4, 384, 160, 160)
        rv_feat_block: （4, 384, 160, 12)
        bev_pos: (4, 384, 40, 40)
        rv_pos: (4, 384, 40, 12)
        """
        bev_atten_outputs, atten_maps= self.cross_atten(
            bev_feat_block, rv_feat_block, bev_pos, rv_pos) # List[Tensor] (4, 384, 40, 40)和(4, 1600, 480)
        # 逐个处理sem和geo
        for bev_atten_output in bev_atten_outputs:
            bev_output_feat_map = self.residual_add(bev_feat_map, # (4, 384, 160, 160)
                                                    bev_atten_output, # (4, 384, 40, 40)
                                                    bev_kernel_size).contiguous() # (4, 4)
            bev_output_feat_maps.append(bev_output_feat_map)
        return bev_output_feat_maps, atten_maps # List[Tensor] (4, 384, 160, 160)和(4, 1600, 480)
