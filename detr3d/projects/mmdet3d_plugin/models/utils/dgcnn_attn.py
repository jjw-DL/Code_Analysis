import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule


@ATTENTION.register_module()
class DGCNNAttn(BaseModule):
    """A warpper for DGCNN-type self-attention.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        dropout (float):w A Dropout layer on attn_output_weights. Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.,
                 init_cfg=None,
                 **kwargs):
        super(DGCNNAttn, self).__init__(init_cfg)
        self.embed_dims = embed_dims # 256
        self.num_heads = num_heads # 8
        self.dropout = dropout # 0.1
        self.conv1 = nn.Sequential(nn.Conv2d(self.embed_dims*2, self.embed_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.embed_dims),
                                   nn.ReLU(inplace=True)) # 512-->256
        self.conv2 = nn.Sequential(nn.Conv2d(self.embed_dims*2, self.embed_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.embed_dims), # 512-->256
                                   nn.ReLU(inplace=True))
        self.K = kwargs['K'] # 16
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `DGCNN`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `DGCNN`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            residual (Tensor): This tensor, with the same shape as x,
                will be used for the residual link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        if residual is None:
            residual = query # (300, 2, 256)
        if query_pos is not None:
            query = query + query_pos # (300, 2, 256)
        
        query = query.permute(1, 0, 2) # [bs, num_queries, embed_dims]-->(2, 300, 256)
        edge_feats = self.edge_feats(query, K=self.K) # (2, 512, 300, 16)
        edge_feats1 = self.conv1(edge_feats) # (2, 256, 300, 16)
        edge_feats1 = edge_feats1.max(dim=-1)[0] # (2, 256, 300)
        out = edge_feats1 # (2, 256, 300)
        edge_feats1 = self.edge_feats(edge_feats1.permute(0, 2, 1)) # # (2, 512, 300, 16)
        edge_feats2 = self.conv2(edge_feats1) # (2, 256, 300, 16)
        edge_feats2 = edge_feats2.max(dim=-1)[0] # (2, 256, 300)
        out = out + edge_feats2 # (2, 256, 300)
        out = out.permute(2, 0, 1) # (300, 2, 256)
        return residual + self.dropout(out)  # (300, 2, 256)

    def edge_feats(self, query, K=16):
        # (B, N, N)
        affinity = torch.cdist(query, query) # (2, 300, 300)
        # (B, N, K)
        _, topk = torch.topk(affinity, k=K, dim=2) # (2, 300, 16)
        B, N, C = query.size() # (2, 300, 256)
        # (2, ) --> (2, 1, 1) eg:[[[0]], [[300]]]
        idx_base = torch.arange(0, B, device=query.device).view(-1, 1, 1) * N
        idx = topk + idx_base # 加上id偏移量-->(2, 300, 16)
        idx = idx.view(-1) # (9600)
        query = query.reshape(B*N, C) # (600, 256)
        query_neighbor = query[idx, :].view(B, N, K, C) # (2, 300, 16, 256)
        # (2, 300, 256)-->(2, 300, 1, 256)-->(2, 300, 16, 256)
        query = query.reshape(B, N, 1, C).repeat(1, 1, K, 1) 
        # (2, 300, 16, 512)-->(2, 512, 300, 16)
        out = torch.cat((query_neighbor, query), dim=-1).permute(0, 3, 1, 2).contiguous()
        return out # (2, 512, 300, 16)
