"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)  # 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # 432-->512
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # 512-->512
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1) # 上采样2倍
        x1 = torch.cat([x2, x1], dim=1) # 拼接特征图 320+112-->432
        return self.conv(x1) # 432-->512


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D # 41
        self.C = C # 64

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0") # 

        self.up1 = Up(320+112, 512) # 432-->512
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0) # 512-->105

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # 输入x:(20, 3, 128, 352)
        x = self.get_eff_depth(x) # -->(20, 512, 8, 22) backbone
        # Depth
        x = self.depthnet(x) # (20, 105, 8, 22) depth+context

        depth = self.get_depth_dist(x[:, :self.D]) # (20, 41, 8, 22) 获取深度分布
        # (20, 1, 41, 8, 22) * (20, 64, 1, 8, 22) --> 将每个深度的点提升为64维度特征，类似点云中的VFE
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2) # (20, 64, 41, 8, 22)

        return depth, new_x # (20, 41, 8, 22)和(20, 64, 41, 8, 22)

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x))) # (20, 32, 64, 176)
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x # (20, 320, 4, 11)
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4']) # (20, 512, 8, 22)
        return x # (20, 512, 8, 22)

    def forward(self, x):
        depth, x = self.get_depth_feat(x) # (20, 41, 8, 22)和(20, 64, 41, 8, 22)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) # 64-->64
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4) # (320, 256)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        ) # 256-->128-->1

    def forward(self, x):
        # x:(4, 64, 200, 200)
        x = self.conv1(x) # (4, 64, 100, 100)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # (4, 64, 100, 100)
        x = self.layer2(x1) # (4, 128, 50, 50)
        x = self.layer3(x) # (4, 256, 25, 25)

        x = self.up1(x, x1) # (4, 256, 100, 100)
        x = self.up2(x) # (4, 1, 200, 200)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False) # (0.5, 0.5, 20)
        self.bx = nn.Parameter(bx, requires_grad=False) # (-49.75, -49.75, 0)
        self.nx = nn.Parameter(nx, requires_grad=False) # (200, 200, 1)

        self.downsample = 16 # 下采样16倍
        self.camC = 64 # 特征图通道数64
        self.frustum = self.create_frustum() # D x H x W x 3-->(41, 8, 22, 3)
        self.D, _, _, _ = self.frustum.shape # 41
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim'] # (128, 352)
        fH, fW = ogfH // self.downsample, ogfW // self.downsample # (8, 22)
        # self.grid_conf['dbound']:[4.0, 45.0, 1.0]-->(41, 1, 1)-->(41, 8, 22)
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape # 41
        
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW) # (1, 1, 22) --> (41, 8, 22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW) # (1, 8, 1) --> (41, 8, 22)

        # D x H x W x 3(x, y, d)
        frustum = torch.stack((xs, ys, ds), -1) # (41, 8, 22, 3) 在原图上间隔采样16个像素采样，并且深度从4-45米，间隔1米采样
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape # 4, 5

        # undo post-transformation (图像坐标系下，原点在图像中心)
        # B x N x D x H x W x 3
        # self.frustum:(41, 8, 22 3)
        # post_trans:(4, 5, 3) --> (4, 5, 1, 1, 1, 3)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3) # (4, 5, 41, 8, 22, 3)
        # post_rots:(4, 5, 3, 3) --> (4, 5, 1, 1, 1, 3, 3)
        # points:(4, 5, 41, 8, 22, 3,) --> (4, 5, 41, 8, 22, 3, 1)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)) # (4, 5, 41, 8, 22, 3, 1)

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5) # (4, 5, 41, 8, 22, 3, 1) --> 逆归一化
        # intrins:(4, 5, 3, 3)
        # rots:(4, 5, 3, 3)
        # combine:(4, 5, 3, 3)
        combine = rots.matmul(torch.inverse(intrins)) # 乘相机内参恢复到相机坐标系下，然后乘相机到自车的旋转矩阵，变换到自车坐标系下
        # combine:(4, 5, 3, 3)-->(4, 5, 1, 1, 1, 3, 3)
        # points:(4, 5, 41, 8, 22, 3, 1)
        # points:(4, 5, 41, 8, 22, 3) 转换到自车坐标系
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points # 所有采样voxel在自车坐标系下的坐标

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape # 4, 5, 3, 128, 352

        x = x.view(B*N, C, imH, imW) # (20, 3, 128, 352)
        x = self.camencode(x) # (20, 64, 41, 8, 22)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample) # (4, 5, 64, 41, 8, 22)
        x = x.permute(0, 1, 3, 4, 5, 2) # (4, 5, 41, 8, 22, 64) 每个3维空间点有64维特征

        return x # (4, 5, 41, 8, 22, 64)

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape # 4, 5, 41, 8, 22, 64
        Nprime = B*N*D*H*W # 144320

        # flatten x
        x = x.reshape(Nprime, C) # (144320, 64)

        # flatten indices
        # geom_feats:(4, 5, 41, 8, 22, 3)(已经转换到自车系，为真实3维空间坐标)
        # 这里转换为voxle(pillar)坐标
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3) # 将pillar坐标拉直 --> (144320, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)]) # 计算batch_id (144320, 1)
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # (144320, 4) 为pillar添加batch id

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] # (135577, 64)
        geom_feats = geom_feats[kept] # (135577, 4) 过滤在锥体范围外的点
        
        # BEVFusion中将下面部分使用CUDA实现
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3] # (135577,) 将4维索引拉成一维(tips:每一维乘后面的所有维度)
        sorts = ranks.argsort() # 计算排序索引(135577,)
        # x:(135577, 64)
        # geom_feats:(135577, 4)
        # ranks:(135577,)
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts] # 将空间中相邻的排在一起

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks) # (35769, 64)和(35769, 4)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device) # (4, 64, 1, 200, 200)
        # 将预测的伪点云特征放入对应位置
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x # (4, 64, 1, 200, 200)

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1) # (4, 64, 200, 200)

        return final # (4, 64, 200, 200)

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans) # (4, 5, 41, 8, 22, 3) 所有预定义的采样点在自车坐标系下的坐标
        x = self.get_cam_feats(x) # (4, 5, 41, 8, 22, 64)

        x = self.voxel_pooling(geom, x) # (4, 64, 200, 200)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans) # (4, 64, 200, 200)
        x = self.bevencode(x) # (4, 1, 200, 200)
        return x # (4, 1, 200, 200)


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
