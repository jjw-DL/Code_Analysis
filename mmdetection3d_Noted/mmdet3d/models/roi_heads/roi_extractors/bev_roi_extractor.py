# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule

from mmdet.models.builder import ROI_EXTRACTORS

@ROI_EXTRACTORS.register_module()
class BEVFeatureExtractor(BaseModule):
    def __init__(self, pc_start, voxel_size, out_stride, init_cfg=None):
        super(BEVFeatureExtractor, self).__init__(init_cfg=init_cfg)
        self.pc_start = pc_start
        self.voxel_size = voxel_size
        self.out_stride = out_stride

    def absl_to_relative(self, absolute):
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return a1, a2
    
    def center_to_corner_box2d(self, centers, dims, angles, origin=0.5):
        ndim = dims.shape[1]
        corners_norm = dims.new_tensor([[0, 0], [0, 1], [1, 1], [1, 0]])
        corners_norm = corners_norm - dims.new_tensor(origin)
        corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)

        rot_sin = torch.sin(angles)
        rot_cos = torch.cos(angles)
        rot_mat_T = torch.stack([torch.stack([rot_cos, -rot_sin]), torch.stack([rot_sin, rot_cos])])
        corners = torch.einsum("aij,jka->aik", (corners, rot_mat_T))

        corners = corners + centers.view(-1, 1, 2)
        return corners

    def get_box_center(self, boxes):
        center2d = boxes[:, :2] # 获取bbox的中心 eg:(128, 2)
        dim2d = boxes[:, 3:5] # 获取bbox的长和宽 eg:(128, 2)
        rotation_y = boxes[:, 6] # 获取bbox的旋转角 eg:(128,)

        corners = self.center_to_corner_box2d(center2d, dim2d, rotation_y) # 将bbox由中心+长宽-->四个角点形式

        # 求出bev bbox的上下左右四个中间点坐标
        front_middle = (corners[:, 0] + corners[:, 1]) / 2
        back_middle =  (corners[:, 2] + corners[:, 3]) / 2
        left_middle =  (corners[:, 0] + corners[:, 3]) / 2
        right_middle = (corners[:, 1] + corners[:, 2]) / 2

        # 5个点的坐标
        points = torch.cat([boxes[:, :2], front_middle, back_middle, left_middle, right_middle], dim=0) # eg: (128x5, 2)

        return points # 每一帧预测bbox的5个特征点


    def bilinear_interpolate_torch(self, im, x, y):
        """
        Args:
            im: (H, W, C) [y, x]
            x: (N)
            y: (N)
        Returns:
        """
        # 1.计算周围4个点坐标
        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y0 = torch.floor(y).long()
        y1 = y0 + 1
        
        # 2.对四个点坐标进行截断，防止超出边界
        x0 = torch.clamp(x0, 0, im.shape[1] - 1)
        x1 = torch.clamp(x1, 0, im.shape[1] - 1)
        y0 = torch.clamp(y0, 0, im.shape[0] - 1)
        y1 = torch.clamp(y1, 0, im.shape[0] - 1)

        # 3.获取4个点坐标的特征
        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        # 4.计算4个点的权重
        wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
        wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
        wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
        wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
        # 5.相乘在相加
        ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
        return ans

    def forward(self, bev_feats, rois):
        batch_size = len(bev_feats) # 2
        final_feats = bev_feats.permute(0, 2, 3, 1).contiguous()
        pooled_roi_feats = []

        # 逐帧处理
        for batch_idx in range(batch_size):
            roi_inds = (rois[..., 0].int() == batch_idx)
            bboxes = rois[..., 1:][roi_inds]
            batch_centers = self.get_box_center(bboxes).detach()

            xs, ys = self.absl_to_relative(batch_centers) # 计算5个点在特征图上的坐标 (640,)
            
            # N x C 
            feature_map = self.bilinear_interpolate_torch(final_feats[batch_idx], xs, ys) # 在特征图上进行双线性插值获得坐标特征--> (128x5, 64)

            section_size = len(feature_map) // 5 # 128
            feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size] for i in range(5)], dim=1) # (128, 320)

            pooled_roi_feats.append(feature_map) # (128, 320)

        pooled_roi_feats = torch.cat(pooled_roi_feats, 0) # (256, 320)

        return pooled_roi_feats


