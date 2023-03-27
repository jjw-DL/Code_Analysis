# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Sequence
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.core.bbox.box_np_ops import points_in_convex_polygon_torch, center_to_corner_box2d_torch
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class AttentionConstrainedLoss(nn.Module):
    def __init__(self,
                 num_class: int,
                 query_res=[40, 40],
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 loss_weight: int = 1.0):
        super().__init__()
        self.cls_out_channels = num_class # 1
        self.pc_range = np.array(pc_range) # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.dims = self.pc_range[3:] - self.pc_range[:3] # [102.4, 102.4, 8]
        self.query_res = query_res # (40, 40)
        self._loss_weight = loss_weight # 1

        self.h, self.w = self.query_res[0], self.query_res[1] # 40, 40
        ww, hh = np.meshgrid(range(self.w), range(self.h)) # (40, 40)和(40, 40)
        ww = ww.reshape(-1) # (1600,)
        hh = hh.reshape(-1) # (1600,)
        self.current_device = torch.cuda.current_device()
        self.ww_l = torch.LongTensor(ww).to(self.current_device) # (1600,)
        self.hh_l = torch.LongTensor(hh).to(self.current_device) # (1600,)
        ww = torch.FloatTensor(ww).to(self.current_device) + 0.5 # (1600,)
        hh = torch.FloatTensor(hh).to(self.current_device) + 0.5 # (1600,)
        ww = ww / self.w * self.dims[0] + self.pc_range[0] # 计算网格点的(真实)中心坐标 比例*真实宽+偏移量
        hh = hh / self.h * self.dims[1] + self.pc_range[1]
        self.grids_sensor = torch.stack([ww, hh], 1).clone().detach() # (1600, 2)
        self.effective_ratio = [1.0, 6.0]

    def find_grid_in_bbx_single(self, x):
        """
        find the attention grids that are enclosed by a GT bounding box
        Args:
            query_res (Sequence[int]): the size of the query feat map
            gt_bboxes (torch.Tensor, [M, ndim]): a single GT bounding boxes set for a scene
        """
        query_res, gt_bboxes = x # (40, 40)和(17, 9)
        bboxes_grid_ind_list = []
        if len(gt_bboxes > 0):
            temp_grid_flag = -1 * torch.ones(query_res, dtype=torch.long).to(self.current_device) # (40, 40) 全部初始化-1
            effective_boxes = gt_bboxes[:, [0, 1, 3, 4]].clone().detach()  # [17, 4] x y l w
            effective_ratio_l = (self.dims[0] / self.w) / effective_boxes[:, 2]  # [17]
            effective_ratio_w = (self.dims[1] / self.h) / effective_boxes[:, 3]  # [17]
            effective_ratio_l = effective_ratio_l.clamp(min=self.effective_ratio[0],  # [17]
                                                        max=self.effective_ratio[1])  # [17]
            effective_ratio_w = effective_ratio_w.clamp(min=self.effective_ratio[0],  # [17]
                                                        max=self.effective_ratio[1])  # [17]
            effective_boxes[:, 2] *= effective_ratio_l
            effective_boxes[:, 3] *= effective_ratio_w
            angles = gt_bboxes[:, 6]
            effective_boxes = center_to_corner_box2d_torch(
                effective_boxes[:, :2], effective_boxes[:, 2:4], angles) # (17, 4, 2)
            grid_real_centers = self.grids_sensor # (1600, 2)
            w_indices = self.ww_l # (1600,)
            h_indices = self.hh_l # (1600,)
            # 逐个gt box处理
            for i in range(len(gt_bboxes)):
                pos_mask = points_in_convex_polygon_torch(
                    grid_real_centers, effective_boxes[i].unsqueeze(0))  # (1600, 1) # 计算网格点在该bbox内的索引mask
                pos_ind = pos_mask.nonzero()[:, 0] # eg: [821, 861]-->(2,)
                gt_center = gt_bboxes[i: i + 1, :2]  # (1, 2)
                dist_to_grid_center = torch.norm(grid_real_centers - gt_center, dim=1)  # [W * H] (1600,) 计算各网格点到该box中心的距离
                min_ind = torch.argmin(dist_to_grid_center) # 最小值索引
                if min_ind not in pos_ind:
                    pos_ind = torch.cat([pos_ind.reshape(-1, 1), min_ind.reshape(-1, 1)],
                                        dim=0).reshape(-1)
                pos_h_indices = h_indices[pos_ind]  # [num_pos] 取出正例索引对应的高
                pos_w_indices = w_indices[pos_ind]  # [num_pos] 取出正例索引对应的宽
                if len(pos_h_indices):
                    # 如果所有正例点都没有被分配，则直接分配，逐个点处理
                    if not (temp_grid_flag[pos_h_indices, pos_w_indices] == -1).all():
                        unique_pos_h_indices = pos_h_indices.new_zeros((0,)) # 初始化空Tensor
                        unique_pos_w_indices = pos_w_indices.new_zeros((0,))
                        # 逐个正例索引处理
                        for ph, pw in zip(pos_h_indices, pos_w_indices):
                            if temp_grid_flag[ph, pw] == -1:
                                unique_pos_h_indices = torch.cat(
                                    (unique_pos_h_indices, ph.view((1))))
                                unique_pos_w_indices = torch.cat(
                                    (unique_pos_w_indices, pw.view((1))))
                            else:
                                temp_grid_flag[ph, pw] = -1 # 如果该正例同时属于2个bbox，则重新分配
                        pos_h_indices = unique_pos_h_indices
                        pos_w_indices = unique_pos_w_indices
                    temp_grid_flag[pos_h_indices, pos_w_indices] = i # 将该点标记为该box
            temp_grid_flag = temp_grid_flag.view(-1) # （1600,)
            # 逐个bbox计算属于该box的正例点
            for i in range(len(gt_bboxes)):
                bbx_grid_ind = torch.where(temp_grid_flag == i)[0]
                bboxes_grid_ind_list.append(bbx_grid_ind)
        return bboxes_grid_ind_list # 17个list

    def find_grid_in_bbx(self,
                         gt_bboxes: List[torch.Tensor]):
        query_sizes = [self.query_res for i in range(len(gt_bboxes))] # [(40,40),(40,40),(40,40),(40,40)]
        map_results = map(self.find_grid_in_bbx_single, zip(query_sizes, gt_bboxes))
        return map_results

    def compute_var_loss(self,
                         atten_map: torch.Tensor,
                         grid_ind_list: List[torch.Tensor]):
        var_loss = 0.0
        var_pos_num = 0.0 
        # 逐个bbox计算
        for i in range(len(grid_ind_list)):
            grid_ind = grid_ind_list[i]
            temp_var_loss = 0.0
            if len(grid_ind) > 0:
                atten_score = atten_map[grid_ind, :] # eg:(2, 480)
                var = torch.var(atten_score, 1).mean() # (2,)
                temp_var_loss = temp_var_loss + (0.0 - var)
                var_pos_num += 1 # 累加正例的数量 (gt box的数量)
            var_loss = var_loss + temp_var_loss # 累加该帧loss
        return var_loss, var_pos_num

    def forward(self,
                atten_map: torch.Tensor,
                gt_bboxes: List[torch.Tensor],
                **kwargs):
        # List[4] 4:batch size, 每个元素包含n个tensor，表示n个box包含的attn点索引
        # loss的计算也是逐任务处理，任务内按照batch处理
        batch_grid_ind_list = list(self.find_grid_in_bbx(gt_bboxes)) 
        var_loss = torch.tensor(0.0).to(self.current_device) # 初始化0
        var_pos_num = 0.0
        # 逐帧处理
        for i in range(len(gt_bboxes)):
            grid_ind_list = batch_grid_ind_list[i] # 提取该帧gt box的atten点索引
            if len(grid_ind_list) > 0:
                temp_var_loss, temp_var_pos_num = self.compute_var_loss(
                    atten_map[i], grid_ind_list) # 计算atten loss
                var_loss = var_loss + temp_var_loss # 累加个帧的loss
                var_pos_num += temp_var_pos_num # 累加各帧的正例box数量
        var_pos_num = max(var_pos_num, 1) # 正例box的数量最少为1
        norm_var_loss = var_loss * 1.0 / var_pos_num # 计算loss平均值
        return norm_var_loss 
