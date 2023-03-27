# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

from . import bev_pool_v2_ext

__all__ = ['bev_pool_v2', 'TRTBEVPoolv2']


class QuickCumsumCuda(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int() # (1508751,)
        depth = depth.contiguous().float() # (8, 6, 59, 16, 44)
        feat = feat.contiguous().float() # (8, 6, 16, 44, 80)
        ranks_depth = ranks_depth.contiguous().int() # (1508751,)
        ranks_feat = ranks_feat.contiguous().int() # (1508751,)
        interval_lengths = interval_lengths.contiguous().int() # (109981,)
        interval_starts = interval_starts.contiguous().int() # (109981,)

        out = feat.new_zeros(bev_feat_shape) # (8, 1, 128, 128, 80)

        bev_pool_v2_ext.bev_pool_v2_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        # ranks_bev:(1508751,)
        # depth:(8, 6, 59, 16, 44)
        # feat: (8, 6, 16, 44, 80)
        # ranks_feat:(1508751,)
        # ranks_depth:(1508751,)
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors # 提取forward时保存的信息

        order = ranks_feat.argsort() # 对特征索引排序
        ranks_feat, ranks_depth, ranks_bev = \
            ranks_feat[order], ranks_depth[order], ranks_bev[order] # 按照feat索引重新组织depth索引和bev索引
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int() # 计算特征起始索引
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[
            1:] - interval_starts_bp[:-1] # 计算一个特征所包含的点的数量（一个像素特征对应D个深度点）
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()
        # 需要求depth和feat以及out的梯度
        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous() # out_grad是上一层的梯度，即dy
        # 进入cuda函数
        bev_pool_v2_ext.bev_pool_v2_backward(
            out_grad,
            depth_grad,
            feat_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, \
            None, None, None


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    # depth:(8, 6, 59, 16, 44)
    # feat: (8, 6, 16, 44, 80)
    # ranks_depth:(1508751,)
    # ranks_feat:(1508751,)
    # ranks_bev:(1508751,)
    # bev_feat_shape:(8, 1, 128, 128, 80)
    # interval_starts: (109981,)       
    # interval_lengths: (109981,)
    x = QuickCumsumCuda.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts,
                              interval_lengths) # (8, 1, 128, 128, 80)
    x = x.permute(0, 4, 1, 2, 3).contiguous() # (b, c, z, x, y)-->(8, 80, 1, 128, 128)
    return x


class TRTBEVPoolv2(torch.autograd.Function):

    @staticmethod
    def symbolic(g,
                 depth,
                 feat,
                 ranks_depth,
                 ranks_feat,
                 ranks_bev,
                 interval_starts,
                 interval_lengths,
                 out_height=128,
                 out_width=128):
        """symbolic function for creating onnx op. 负责pytorch和onnx的算子接口映射, 并没有具体实现"""
        return g.op(
            'mmdeploy::bev_pool_v2', # 对应与onnx模型中的module和type--> module:mmdeploy, type:bev_pool_v2
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths, # 输入不需要赋值，对应onnx中的INPUTS
            out_height_i=out_height,
            out_width_i=out_width) # 属性值需要赋值，对应onnx中的ATTRIBUTES

    @staticmethod
    def forward(g,
                depth,
                feat,
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
                out_height=128,
                out_width=128): # 只需要forward即可，infer过程中不需要backward
        """run forward."""
        n, d, h, w = depth.shape # eg:6, 59, 16, 44
        # (6, 16, 44, 80)-->(1, 6, 80, 16, 44)
        feat = feat.view(1, n, feat.shape[3], h, w) # (1, 6, 80, 16, 44)
        feat = feat.permute(0, 1, 3, 4, 2) # (1, 6, 16, 44, 80)
        depth = depth.view(1, n, d, h, w) # (1, 6, 59, 16, 44)
        bev_feat_shape = (depth.shape[0], 1, out_height, out_width,
                          feat.shape[-1])  # (B, Z, Y, X, C)-->(1, 1, 128, 128, 80)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths) # 调用bev_pool_v2函数 -->(1, 80, 1, 128, 128)
        bev_feat = bev_feat.squeeze(2) # (1, 80, 128, 128)
        bev_feat = bev_feat.permute(0, 2, 3, 1) # (1, 128, 128, 80)
        return bev_feat # (1, 128, 128, 80)

# 学习自定义算法的unit test写法，尤其是关于backward
def test_bev_pool_v2():
    depth = np.array([0.3, 0.4, 0.2, 0.1, 0.7, 0.6, 0.8, 0.9])
    depth = torch.from_numpy(depth).float().cuda()
    # [[[[[0.3, 0.4], 
    #     [0.2, 0.1]]],
    #     [[0.7, 0.6],
    #       [0.8, 0.0]]]]] 
    depth = depth.view(1, 1, 2, 2, 2).requires_grad_() # (b, n, d, w, h)-->(1, 1, 2, 2, 2)

    feat = torch.ones(
        size=[1, 1, 2, 2, 2], dtype=torch.float,
        device='cuda').requires_grad_() # (b, n, w, h, c)-->(1, 1, 2, 2, 2)
    ranks_depth = torch.from_numpy(np.array([0, 4, 1, 6])).int().cuda()
    ranks_feat = torch.from_numpy(np.array([0, 0, 1, 2])).int().cuda()
    ranks_bev = torch.from_numpy(np.array([0, 0, 1, 1])).int().cuda()

    kept = torch.ones(
        ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool) # (4,)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1] # [True, False, True, False]
    interval_starts = torch.where(kept)[0].int() # [0, 2]
    if len(interval_starts) == 0:
        return None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1] # [2, 2]
    # [[[[[1.0, 1.2], --> 1=0.3+0.7, 1.2=0.4+0.8 一个深度*2个特征
    #     [0.0, 0.0]]],
    #     [[[1.0, 1.2],
    #       [0.0, 0.0]]]]]
    bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                           (1, 1, 2, 2, 2), interval_starts, interval_lengths) # (b, c, z, x, y)-->(1, 2, 1, 2, 2)
    loss = torch.sum(bev_feat) # 4.4
    loss.backward()
    assert loss == 4.4
    # loss是各个pillar加和的形式，因此对每一个pillar的grad为1
    # 每个depth点对应2个特征，grad_sum += *cur_out_grad * *cur_feat-->depth.grad=2
    grad_depth = np.array([2., 2., 0., 0., 2., 0., 2., 0.])
    grad_depth = torch.from_numpy(grad_depth).float()
    grad_depth = grad_depth.cuda().view(1, 1, 2, 2, 2)
    assert depth.grad.allclose(grad_depth)
    # 逐个特征维度计算梯度 eg:第0维特征的梯度是第0和第4维深度相加0.3+0.7=1
    # 每两个特征对应一个像素点,同理第1维特征的梯度也是1
    # 第1维特征的梯度是第1维深度=0.4
    grad_feat = np.array([1.0, 1.0, 0.4, 0.4, 0.8, 0.8, 0., 0.])
    grad_feat = torch.from_numpy(grad_feat).float().cuda().view(1, 1, 2, 2, 2)
    assert feat.grad.allclose(grad_feat)
