import torch
import numpy as np
import math
import torch.nn as nn
from torch.nn import functional as F
from mmdet3d.models.fusion_layers import apply_3d_transformation
from .ops import locatt_ops
from .ip_basic import depth_map_utils
import pdb

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class similarFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.similar_forward(
            x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.similar_backward(
            x_loc, grad_outputs, kH, kW, True)
        grad_loc = locatt_ops.localattention.similar_backward(
            x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.weighting_forward(
            x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.weighting_backward_ori(
            x_weight, grad_outputs, kH, kW)
        grad_weight = locatt_ops.localattention.weighting_backward_weight(
            x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None
    

class LocalContextAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()

        self.f_similar = similarFunction.apply
        self.f_weighting = weightingFunction.apply

        self.kernel_size = kernel_size # 9
        self.query_project = nn.Sequential(ConvBNReLU(in_channels, # 128
                                                                  out_channels, # 128
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           ConvBNReLU(out_channels, # 128
                                                                  out_channels, # 128
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU))
        self.value_project = ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_feats, source_feats, **kwargs):
        query = self.query_project(target_feats) # (1, 128, 180, 180)
        key = self.key_project(source_feats) # (1, 128, 180, 180)
        value = self.value_project(source_feats) # (1, 128, 180, 180)

        weight = self.f_similar(query, key, self.kernel_size, self.kernel_size) # (1, 180, 180, 81)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1) # (1, 180, 180, 81)
        out = self.f_weighting(value, weight, self.kernel_size, self.kernel_size) # (1, 128, 180, 180)
        return out # (1, 128, 180, 180)


class BEVWarp(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    # 可以with no grad
    @torch.no_grad()
    def forward(self, lidar_feats, img_feats, img_metas, pts_metas, **kwargs):
        batch_size, num_views, I_C, I_H, I_W = img_feats.shape # 1, 6, 128, 112, 200
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = img_feats.new_tensor(lidar2img)
        img2lidar = torch.inverse(lidar2img) # (1, 6, 4, 4) img2lidar映射矩阵
        pts = pts_metas['pts'] # eg:[(366885, 5)] 点云
        decorated_img_feats = []
        # 逐帧处理
        for b in range(batch_size):
            img_feat = img_feats[b] # (6, 128, 112, 200)
            ori_H, ori_W = img_metas[b]['input_shape'] # (448, 800)
            pts_3d = pts[b][...,:3] # 提取点云 (366885, 5)
            pts_3d = apply_3d_transformation(pts_3d, 'LIDAR', img_metas[b], reverse=True).detach() # 逆变换点云 (366885, 3)
            pts_4d = torch.cat((pts_3d,torch.ones_like(pts_3d[...,:1])),dim=-1).unsqueeze(0).unsqueeze(-1) # 转换齐次坐标 (366885, 4)
            proj_mat = lidar2img[b].unsqueeze(1) # (6, 1, 4, 4)
            pts_2d = torch.matmul(proj_mat, pts_4d).squeeze(-1) # (6, 366885, 4)
            eps = 1e-5
            depth = pts_2d[..., 2:3] # (6, 366885, 1)
            mask = (pts_2d[..., 2:3] > eps)
            pts_2d = pts_2d[..., 0:2] / torch.maximum(
                pts_2d[..., 2:3], torch.ones_like(pts_2d[..., 2:3])*eps) # (6, 366885, 2)
            proj_x = (pts_2d[...,0:1] / ori_W - 0.5) * 2 # 归一化
            proj_y = (pts_2d[...,1:2] / ori_H - 0.5) * 2
            mask = (mask & (proj_x > -1.0) 
                             & (proj_x < 1.0) 
                             & (proj_y > -1.0) 
                             & (proj_y < 1.0))
            mask = torch.nan_to_num(mask) # (6, 366885, 1)
            depth_map = img_feat.new_zeros(num_views, I_H, I_W) # (6, 112, 200) 初始化深度图
            # 点云像素坐标/原始宽高*深度图宽高-->s深度图像素坐标
            # 利用mask将点云深度赋予到深度图的对应位置
            for i in range(num_views):
                depth_map[i, (pts_2d[i,mask[i,:,0],1]/ori_H*I_H).long(), (pts_2d[i,mask[i,:,0],0]/ori_W*I_W).long()] = depth[i,mask[i,:,0],0]
            # 采取不同方式，补充深度图
            fill_type = 'multiscale'
            extrapolate = False
            blur_type = 'bilateral'
            for i in range(num_views):
                final_depths, _ = depth_map_utils.fill_in_multiscale(
                                depth_map[i].detach().cpu().numpy(), extrapolate=extrapolate, blur_type=blur_type,
                                show_process=False) # 根据lidar投影深度图通过形态学操作补全深度图 eg:(112, 200)
                depth_map[i] = depth_map.new_tensor(final_depths)
            
            # (200,)-->(1, 1, 200)-->(6, 112, 200)
            xs = torch.linspace(0, ori_W - 1, I_W, dtype=torch.float32).to(depth_map.device).view(1, 1, I_W).expand(num_views, I_H, I_W)
            ys = torch.linspace(0, ori_H - 1, I_H, dtype=torch.float32).to(depth_map.device).view(1, I_H, 1).expand(num_views, I_H, I_W)
            # 组合拼接深度图 (6, 112, 200, 4)
            xyd = torch.stack((xs, ys, depth_map, torch.ones_like(depth_map)), dim = -1)
            xyd [..., 0] *= xyd [..., 2]
            xyd [..., 1] *= xyd [..., 2]
            # (6, 4, 4)-->(6, 1, 1, 4, 4)*(6, 112, 200, 4, 1)-->(6, 112, 200, 4, 1)-->(6, 112, 200, 4)-->(6, 112, 200, 3)
            xyz = img2lidar[b].view(num_views,1,1,4,4).matmul(xyd.unsqueeze(-1)).squeeze(-1)[...,:3] # (6, 112, 200, 3)
            # 对恢复的伪点云进行几何变换，并变换维度 (134400, 3)-->(6, 112, 200, 3)
            xyz = apply_3d_transformation(xyz.view(num_views*I_H*I_W, 3), 'LIDAR', img_metas[b], reverse=False).view(num_views, I_H, I_W, 3).detach()
            # 过滤不符合条件的伪点云
            pc_range = xyz.new_tensor([-54, -54, -5, 54, 54, 3])  #TODO: fix it to support other outdoor dataset!!!
            lift_mask = (xyz[...,0] > pc_range[0]) & (xyz[...,1] > pc_range[1]) & (xyz[...,2] > pc_range[2])\
                        & (xyz[...,0] < pc_range[3]) & (xyz[...,1] < pc_range[4]) & (xyz[...,2] < pc_range[5]) #（6, 112, 200)
            xy_bev = (xyz[...,0:2] - pc_range[0:2]) / (pc_range[3:5] - pc_range[0:2]) # 点云坐标归一化 (6, 112, 200)
            xy_bev = (xy_bev - 0.5) * 2 # 转换采样坐标 (6, 112, 200)
            # lidar_feats[b]:(128, 180, 180)-->(1, 128, 180, 180)-->(6, 128, 180, 180)
            # 采样后：(6, 128, 112, 200)-->(6, 112, 200, 128)
            decorated_img_feat = F.grid_sample(lidar_feats[b].unsqueeze(0).repeat(num_views,1,1,1), xy_bev, align_corners=False).permute(0,2,3,1) #N, H, W, C
            decorated_img_feat[~lift_mask]=0 # 将无效点特征置0
            decorated_img_feats.append(decorated_img_feat.permute(0,3,1,2)) # (6, 128, 112, 200)
        decorated_img_feats = torch.stack(decorated_img_feats, dim=0) # 将采样特征进行拼接 (1, 6, 128, 112, 200)
        return decorated_img_feats # (1, 6, 128, 112, 200)


class MMRI_P2I(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()
        self.Warp = BEVWarp()
        self.Local = LocalContextAttentionBlock(in_channels, out_channels, kernel_size, last_affine=True)
    
    def forward(self, lidar_feats, img_feats, img_metas, pts_metas, **kwargs):
        warped_img_feats = self.Warp(lidar_feats, img_feats, img_metas, pts_metas) # B, N, C, H, W-->(1, 6, 128, 112, 200)
        B, N, C, H, W = warped_img_feats.shape # 1, 6, 128, 112, 200
        # (6, 128, 112, 200)和(6, 128, 112, 200) --> (1, 6, 128, 112, 200) image和lidar信息执行交叉注意力
        decorated_img_feats = self.Local(img_feats.view(B*N,C,H,W), warped_img_feats.view(B*N,C,H,W)).view(B,N,C,H,W) # (1, 6, 128, 112, 200)
        return decorated_img_feats # (1, 6, 128, 112, 200)
    

class MMRI_I2P(nn.Module):
    # Multi-modal representational interaction（MMRI）点云投影到图像
    def __init__(self, pts_channels, img_channels, dropout):
        super().__init__()
        self.pts_channels = pts_channels # 128
        self.img_channels = img_channels # 128
        self.dropout = dropout # 0.1
        self.learnedAlign = nn.MultiheadAttention(pts_channels, 1, dropout=dropout, 
                                             kdim=img_channels, vdim=img_channels, batch_first=True) # (128, 1, 0.1, 128, 128,True)

    def forward(self, lidar_feat, img_feat, img_metas, pts_metas, **kwargs):
        batch_size = len(img_metas) # 1
        decorated_lidar_feat = torch.zeros_like(lidar_feat) # (1, 128, 180, 180)
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img']) # 整理投影矩阵信息
        lidar2img = np.asarray(lidar2img)
        lidar2img = lidar_feat.new_tensor(lidar2img) # (B,6,4,4)
        batch_cnt = lidar_feat.new_zeros(batch_size).int() # [0]
        for b in range(batch_size):
            batch_cnt[b] = (pts_metas['pillar_coors'][:,0] == b).sum() # 计算每帧点云的pillar数量 eg:5560
        batch_bound = batch_cnt.cumsum(dim=0) # 计算一个batch中的点云pillar数量 eg:(5560)
        cur_start = 0
        # 逐帧处理
        for b in range(batch_size):
            cur_end = batch_bound[b] # eg:5560
            voxel = pts_metas['pillars'][cur_start:cur_end] # 截取pillar特征 eg:(5560, 20, 5)
            voxel_coor = pts_metas['pillar_coors'][cur_start:cur_end] # eg:(5560, 4)
            pillars_num_points = pts_metas['pillars_num_points'][cur_start:cur_end] # eg:(5560,)
            proj_mat = lidar2img[b] # (6, 4, 4)
            num_cam = proj_mat.shape[0] # 6
            num_voxels, max_points, p_dim = voxel.shape # 5560, 20, 5
            num_pts = num_voxels * max_points # 111200=5560*20
            pts = voxel.view(num_pts, p_dim)[...,:3] # eg:(111200, 3)
            voxel_pts = apply_3d_transformation(pts, 'LIDAR', img_metas[b], reverse=True).detach() # eg:(111200, 3) 点云逆向变换
            voxel_pts = torch.cat((voxel_pts,torch.ones_like(voxel_pts[...,:1])),dim=-1).unsqueeze(0).unsqueeze(-1) # eg:(1, 111200, 4, 1)
            proj_mat = proj_mat.unsqueeze(1) # (6, 1, 4, 4)
            xyz_cams = torch.matmul(proj_mat, voxel_pts).squeeze(-1) # (6, 111200, 4) 将点云投影到相机上
            eps = 1e-5
            mask = (xyz_cams[..., 2:3] > eps) # 根据有效深度计算mask eg:（6, 111200, 1)
            xy_cams = xyz_cams[..., 0:2] / torch.maximum(
                xyz_cams[..., 2:3], torch.ones_like(xyz_cams[..., 2:3])*eps) # 计算点云投影到图像上的像素坐标 eg:（6, 111200, 2)
            img_shape = img_metas[b]['input_shape']
            xy_cams[...,0] = xy_cams[...,0] / img_shape[1] # 归一化
            xy_cams[...,1] = xy_cams[...,1] / img_shape[0]
            xy_cams = (xy_cams - 0.5) * 2 # 计算插值坐标

            mask = (mask & (xy_cams[..., 0:1] > -1.0) 
                 & (xy_cams[..., 0:1] < 1.0) 
                 & (xy_cams[..., 1:2] > -1.0) 
                 & (xy_cams[..., 1:2] < 1.0))
            mask = torch.nan_to_num(mask) # eg:（6, 111200, 1) 有效投影点的mask
            # img_feat[b] eg:(6, 128, 112, 200)
            # xy_cams.unsqueeze(-2) eg:(6, 111200, 1, 2)
            # (6, 128, 111200)-->(111200, 6, 128)
            sampled_feat = F.grid_sample(img_feat[b],xy_cams.unsqueeze(-2)).squeeze(-1).permute(2,0,1)
            sampled_feat = sampled_feat.view(num_voxels,max_points,num_cam,self.img_channels) # (5560, 20, 6, 128)

            mask = mask.permute(1,0,2).view(num_voxels,max_points,num_cam,1) # （6, 111200, 1)-->(111200, 6, 1)-->(5560, 20, 6, 1)

            # for i in range(num_voxels):
            #     mask[i,pillars_num_points[i]:] = False
            mask_points = mask.new_zeros((mask.shape[0],mask.shape[1]+1)) # (5560, 21)
            mask_points[torch.arange(mask.shape[0],device=mask_points.device).long(),pillars_num_points.long()] = 1 # (5560, 21)
            mask_points = mask_points.cumsum(dim=1).bool() # (5560, 21) 沿着第一维度累加，并且转换为bool值
            mask_points = ~mask_points # 取反
            mask = mask_points[:,:-1].unsqueeze(-1).unsqueeze(-1) & mask # (5560, 20, 1, 1) & (5560, 20, 6, 1) --> 标记pillar内有效数量点和有效投影点

            mask = mask.reshape(num_voxels,max_points*num_cam,1) # (5560, 20*6, 1)
            sampled_feat = sampled_feat.reshape(num_voxels,max_points*num_cam,self.img_channels) # (5560, 120, 128)

            # Q,K和V均以Pillar为单位
            K = sampled_feat # 图像采样特征 (5560, 120, 128)
            V = sampled_feat
            # 提取有效pillar特征 (128, 5560)-->(5560, 128)-->(5560, 1, 128)
            Q = lidar_feat[b,:,voxel_coor[:,2].long(),voxel_coor[:,3].long()].t().unsqueeze(1)

            valid = mask[...,0].sum(dim=1) > 0 # (5560, 120, 1)-->(5560, 120)--->(5560) 计算每一个voxle是否存在有效点
            attn_output = lidar_feat.new_zeros(num_voxels, 1, self.pts_channels) # 初始化注意力输出 eg:(5560, 1, 128)
            # attn_mask:(5560, 120, 1)-->(5432, 120, 1)-->(5432, 1, 120)
            attn_output[valid] = self.learnedAlign(Q[valid],K[valid],V[valid], attn_mask=(~mask[valid]).permute(0,2,1))[0]
            # 将融合图像的点云pillar特征填入对应位置
            decorated_lidar_feat[b,:,voxel_coor[:,2].long(),voxel_coor[:,3].long()] = attn_output.squeeze(1).t() # (128, 5560)
            cur_start = cur_end # 5560
        return decorated_lidar_feat # (1, 128, 180, 180)
