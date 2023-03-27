# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import NECKS
from mmdet3d.ops import bev_pool
from mmcv.cnn import build_conv_layer
from .. import builder


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


@NECKS.register_module()
class ViewTransformerLiftSplatShoot(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 accelerate=False, max_drop_point_rate=0.0, use_bev_pool=True,
                 **kwargs):
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
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.accelerate = accelerate
        self.max_drop_point_rate = max_drop_point_rate
        self.use_bev_pool = use_bev_pool

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
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

        # cam_to_ego
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
        geom_feats = geom_feats.view(Nprime, 3) # (1993728, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # (1993728, 4)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2]) # (1993728, )
        x = x[kept] # (1761017, 64)
        geom_feats = geom_feats[kept] # (1761017, 4) 过滤在锥体范围外的点


        if self.max_drop_point_rate > 0.0 and self.training:
            drop_point_rate = torch.rand(1)*self.max_drop_point_rate
            kept = torch.rand(x.shape[0])>drop_point_rate
            x, geom_feats = x[kept], geom_feats[kept]

        if self.use_bev_pool:
            final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0],
                                   self.nx[1])
            final = final.transpose(dim0=-2, dim1=-1)
        else:
            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3] # (1761017,) 将4维索引拉成一维(tips:每一维乘后面的所有维度)
            sorts = ranks.argsort()
            x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

            # cumsum trick
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks) # 将空间中相邻的排在一起（120991, 64)和（120991, 4)

            # griddify (B x C x Z x X x Y)
            final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device) # (8, 64, 1, 128, 128)
            # 将预测的伪点云特征放入对应位置
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1) # (8, 64, 128, 128)

        return final # (8, 64, 128, 128)

    def voxel_pooling_accelerated(self, rots, trans, intrins, post_rots, post_trans, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        max = 300
        # flatten indices
        if self.geom_feats is None:
            geom_feats = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                             device=x.device, dtype=torch.long) for ix in range(B)])
            geom_feats = torch.cat((geom_feats, batch_ix), 1)

            # filter out points that are outside box
            kept1 = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
                    & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
                    & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
            idx = torch.range(0, x.shape[0] - 1, dtype=torch.long)
            x = x[kept1]
            idx = idx[kept1]
            geom_feats = geom_feats[kept1]

            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks, idx = x[sorts], geom_feats[sorts], ranks[sorts], idx[sorts]
            repeat_id = torch.ones(geom_feats.shape[0], device=geom_feats.device, dtype=geom_feats.dtype)
            curr = 0
            repeat_id[0] = 0
            curr_rank = ranks[0]

            for i in range(1, ranks.shape[0]):
                if curr_rank == ranks[i]:
                    curr += 1
                    repeat_id[i] = curr
                else:
                    curr_rank = ranks[i]
                    curr = 0
                    repeat_id[i] = curr
            kept2 = repeat_id < max
            repeat_id, geom_feats, x, idx = repeat_id[kept2], geom_feats[kept2], x[kept2], idx[kept2]

            geom_feats = torch.cat([geom_feats, repeat_id.unsqueeze(-1)], dim=-1)
            self.geom_feats = geom_feats
            self.idx = idx
        else:
            geom_feats = self.geom_feats
            idx = self.idx
            x = x[idx]

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0], max), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0], geom_feats[:, 4]] = x
        final = final.sum(-1)
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins, post_rots, post_trans, volume)
        else:
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            bev_feat = self.voxel_pooling(geom, volume)
        return bev_feat


class SELikeModule(nn.Module):
    def __init__(self, in_channel=512, feat_channel=256, intrinsic_channel=33):
        super(SELikeModule, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0) # 512-->256
        self.fc = nn.Sequential(
            nn.BatchNorm1d(intrinsic_channel),
            nn.Linear(intrinsic_channel, feat_channel), # 33-->256
            nn.Sigmoid() )

    def forward(self, x, cam_params):
        # x:(48, 512, 16, 44)
        # cam_params: (48, 33)
        x = self.input_conv(x) # (48, 256, 16, 44)
        b,c,_,_ = x.shape # 48, 256
        y = self.fc(cam_params).view(b, c, 1, 1) # (48, 256, 1, 1)
        return x * y.expand_as(x) # (48, 256, 16, 44)


@NECKS.register_module()
class ViewTransformerLSSBEVDepth(ViewTransformerLiftSplatShoot):
    def __init__(self, extra_depth_net, loss_depth_weight, se_config=dict(), **kwargs):
        super(ViewTransformerLSSBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.extra_depthnet = builder.build_backbone(extra_depth_net)
        self.featnet = nn.Conv2d(self.numC_input, # 512
                                 self.numC_Trans, # 64
                                 kernel_size=1,
                                 padding=0)
        self.depthnet = nn.Conv2d(extra_depth_net['num_channels'][0], # 256
                                  self.D, # 59
                                  kernel_size=1,
                                  padding=0)
        self.dcn = nn.Sequential(*[build_conv_layer(dict(type='DCNv2',
                                                        deform_groups=1),
                                                   extra_depth_net['num_channels'][0], # 256
                                                   extra_depth_net['num_channels'][0], # 256
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   dilation=1,
                                                   bias=True),
                                   nn.BatchNorm2d(extra_depth_net['num_channels'][0])
                                  ])
        self.se = SELikeModule(self.numC_input, # 512
                               feat_channel=extra_depth_net['num_channels'][0], # 256
                               **se_config)

    def forward(self, input):
        """
        imgs: (8, 6, 3, 256, 704)
        rots: (8, 6, 3, 3)
        trans: (8, 6, 3)
        intrins: (8, 6, 3, 3)
        post_rots: (8, 6, 3, 3)
        post_trans: (8, 6, 3)
        depth_gt:(8, 6, 16, 44)
        """
        x, rots, trans, intrins, post_rots, post_trans, depth_gt = input
        B, N, C, H, W = x.shape # (8, 6, 512, 16, 64)
        x = x.view(B * N, C, H, W) # (48, 512, 16, 44)

        img_feat = self.featnet(x) # (48, 64, 16, 44) 
        depth_feat = x # (48, 512, 16, 44)
        cam_params = torch.cat([intrins.reshape(B*N,-1),
                               post_rots.reshape(B*N,-1),
                               post_trans.reshape(B*N,-1),
                               rots.reshape(B*N,-1),
                               trans.reshape(B*N,-1)],dim=1) # (48, 33)-->33 = 9 + 9 + 3 + 9 + 3
        depth_feat = self.se(depth_feat, cam_params) # (48, 256, 16, 44)
        depth_feat = self.extra_depthnet(depth_feat)[0] # (48, 256, 16, 44)
        depth_feat = self.dcn(depth_feat) # (48, 256, 16, 44)
        depth_digit = self.depthnet(depth_feat) # (48, 59, 16, 44) 深度值
        depth_prob = self.get_depth_dist(depth_digit) # (48, 59, 16, 44) 深度概率分布 softmax

        # Lift
        # (48, 1, 59, 16, 44) *  (48, 64, 1, 16, 44) --> (48, 64, 59, 16, 44)
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2) # product
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W) # (8, 6, 64, 59, 16, 44)
        volume = volume.permute(0, 1, 3, 4, 5, 2) # (8, 6, 59, 16, 44, 64)

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins, post_rots, post_trans, volume)
        else:
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans) # (8, 6, 59, 16, 44, 3)
            bev_feat = self.voxel_pooling(geom, volume) # (8, 64, 128, 128)
        return bev_feat, depth_digit # (8, 64, 128, 128)和(48, 59, 16, 44)

