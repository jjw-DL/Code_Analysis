# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import NECKS


def gen_dx_bx(xbound, ybound, zbound):
    # xbound: [low_bound, upper_bound, size]
    # 'xbound': [-51.2, 51.2, 0.8]
    # 'ybound': [-51.2, 51.2, 0.8]
    # 'zbound': [-10.0, 10.0, 20.0]
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) # [0.8, 0.8, 20]
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]]) # [-50.8, -50.8, 0]
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]) # [128, 128, 1]
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0) # (135577, 64) 沿着行累加
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool) # 135577
    kept[:-1] = (ranks[1:] != ranks[:-1]) # trick:坐标在同一个pillar内的点只保留一个
    x, geom_feats = x[kept], geom_feats[kept] # 提取被保留的voxel特征和geo特征 (35769, 64)和(35769, 4)
    x = torch.cat((x[:1], x[1:] - x[:-1])) # (35769, 64) 计算每个pillar内的特征，并与第一行拼接
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0) # (1761017, 64) 沿着行累加
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool) # 1761017
        kept[:-1] = (ranks[1:] != ranks[:-1]) # trick:坐标在同一个pillar内的点只保留一个

        x, geom_feats = x[kept], geom_feats[kept] # 提取被保留的voxel特征和geo特征 (120991, 64)和(120991, 4)
        x = torch.cat((x[:1], x[1:] - x[:-1])) # (120991, 64) 计算每个pillar内的特征，并与第一行拼接(与cumsum结合的巧妙操作)

        # save kept for backward
        ctx.save_for_backward(kept) # 几何特征不求梯度

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0) # 对于一维特征，沿着列累加
        back[kept] -= 1 # 减一变为索引

        val = gradx[back] # 提取上下文特征对应位置的梯度，几何特征不求梯度

        return val, None, None # 对应输入


@NECKS.register_module()
class ViewTransformerLiftSplatShoot(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 image_view_supervision=False, voxel=False, **kwargs):
        super(ViewTransformerLiftSplatShoot, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False) # [0.8, 0.8, 20]
        self.bx = nn.Parameter(bx, requires_grad=False) # [-50.8, -50.8, 0]
        self.nx = nn.Parameter(nx, requires_grad=False) # [128, 128, 1]

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample # 下采样16倍

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape 
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.image_view_supervision = image_view_supervision
        self.voxel=voxel

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size'] # 256, 704
        fH, fW = ogfH // self.downsample, ogfW // self.downsample # 16, 44
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape # 59
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1) # (59, 16, 44, 3) 在原图上间隔采样16个像素采样，并且深度从1-60米，间隔1米采样
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, offset=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation 恢复到图像变换前的点 P' = R*P + t --> P = R^(-1) * (P' - t)
        # B x N x D x H x W x 3
        # self.frustum:(59, 16, 44, 3) 这里是图片变换后特征图产生的点是P'
        # post_trans:(8, 5, 3) --> (8, 6, 1, 1, 1, 3)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3) # (8, 6, 59, 16, 44, 3)
        if offset is not None:
            _,D,H,W = offset.shape
            points[:,:,:,:,:,2] = points[:,:,:,:,:,2]+offset.view(B,N,D,H,W)
        # post_rots:(8, 6, 3, 3) --> (8, 6, 1, 1, 1, 3, 3)
        # points:(8, 6, 59, 16, 44, 3)--> (8, 6, 59, 16, 44, 3, 1)
        # 这里的变换包含resize，此时已经恢复到900x1600的原始图片对应点
        # frustum生成是在特征图上均匀撒点，但是经过数据增强逆变换后会对应到原图的上的不同点，对齐特征
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_lidar
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5) # 逆归一化 --> (8, 6, 59, 16, 44, 3, 1)
        if intrins.shape[3]==4: # for KITTI
            shift = intrins[:,:,:3,3]
            points  = points - shift.view(B,N,1,1,1,3,1)
            intrins = intrins[:,:,:3,:3]
        # 这里的rot和trans都是经过点云增强后的矩阵
        # 相当于对伪点云做增强，gt随着增强，因为伪点云特征和gt是可以对应上的
        combine = rots.matmul(torch.inverse(intrins)) # 乘相机内参恢复到相机坐标系下，然后乘相机到lidar的旋转矩阵，变换到lidar系下
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1) # (8, 6, 59, 16, 44, 3)
        points += trans.view(B, N, 1, 1, 1, 3) # (8, 6, 59, 16, 44, 3)

        # points_numpy = points.detach().cpu().numpy()
        return points # (8, 6, 59, 16, 44, 3)

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape # 8, 6, 59, 16, 44, 64
        Nprime = B * N * D * H * W # 1993728
        nx = self.nx.to(torch.long) # [128, 128, 1]
        # flatten x
        x = x.reshape(Nprime, C) # (1993728, 64)

        # flatten indices
        # 这里转换为voxle(pillar)坐标
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long() # (8, 6, 59, 16, 44, 3)
        geom_feats = geom_feats.view(Nprime, 3) # # (1993728, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # (1993728, 4)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2]) # # (1993728, )
        x = x[kept] # (1761017, 64)
        geom_feats = geom_feats[kept] # (1761017, 4) 过滤在锥体范围外的点

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3] # (1761017,) 将4维索引拉成一维(tips:每一维乘后面的所有维度)
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # BEVFusion中将下面部分使用CUDA实现
        # cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks) # 将空间中相邻的排在一起（120991, 64)和（120991, 4)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device) # (8, 64, 1, 128, 128)
        # 将预测的伪点云特征放入对应位置
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        if self.voxel:
            return final.sum(2), x, geom_feats # (8, 64, 1, 128, 128)
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1) # (8, 64, 128, 128)

        return final # (8, 64, 128, 128)

    def forward(self, input):
        """
        imgs: (8, 6, 3, 256, 704)
        rots: (8, 6, 3, 3)
        trans: (8, 6, 3)
        intrins: (8, 6, 3, 3)
        post_rots: (8, 6, 3, 3)
        post_trans: (8, 6, 3)
        """
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, C, H, W = x.shape # (8, 6, 512, 16, 64)
        x = x.view(B * N, C, H, W) # (48, 512, 16, 44)
        x = self.depthnet(x) # (48, 123, 16, 44)
        depth = self.get_depth_dist(x[:, :self.D]) # (48, 59, 16, 44)
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans) # (8, 6, 59, 16, 44, 3)
        img_feat = x[:, self.D:(self.D + self.numC_Trans)] # (48, 64, 16, 44)

        # Lift
        # (48, 1, 59, 16, 44) *  (48, 64, 1, 16, 44) --> (48, 64, 59, 16, 44)
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2) # product
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W) # (8, 6, 64, 59, 16, 44)
        volume = volume.permute(0, 1, 3, 4, 5, 2) # (8, 6, 59, 16, 44, 64)

        # Splat
        bev_feat = self.voxel_pooling(geom, volume) # (8, 64, 128, 128)
        if self.image_view_supervision:
            return bev_feat, [x[:, :self.D].view(B,N,self.D,H,W), x[:, self.D:].view(B,N,self.numC_Trans,H,W)]
        return bev_feat # (8, 64, 128, 128)
