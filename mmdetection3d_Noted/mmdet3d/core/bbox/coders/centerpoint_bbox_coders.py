# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class CenterPointBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 code_size=9):

        self.pc_range = pc_range # [-51.2, -51.2]
        self.out_size_factor = out_size_factor # 8
        self.voxel_size = voxel_size # [0.1, 0.1]
        self.post_center_range = post_center_range # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.max_num = max_num # 500
        self.score_threshold = score_threshold # 0.1
        self.code_size = code_size # 9

    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.
        和centerpoint_head实现相同
        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor): Mask of the feats. Default: None.

        Returns:
            torch.Tensor: Gathered feats.
        """
        dim = feats.size(2) # 获取特征维度 eg:1 feats:(1, 500, 1)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim) # 对ind进行拓展 --> (1, 500) --> (1, 500, 1)
        feats = feats.gather(1, inds) # 根据ind抽取特征 (1, 500, 1)
        # 如果有mask，根据mask进行filter
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].
            K (int): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K]. eg:(1, 500) 或（2, 500）
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size() # 1, 1, 128, 128

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K) # K=500, (1, 1, 16384) --> (1, 1, 500) 取出前500个最高分和对应索引

        topk_inds = topk_inds % (height * width) # (1, 1, 500) 计算batch内索引
        topk_ys = (topk_inds.float() /
                   torch.tensor(width, dtype=torch.float)).int().float() # 计算y轴坐标 (1, 1, 500)
        topk_xs = (topk_inds % width).int().float() # 计算x轴坐标 (1, 1, 500)

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) # 如果是多分类，单帧点云取出前500个最高分 --> (1, 500)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int() # 计算最高分所属类别 --> (1, 500) eg:0或1
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), # (1, 500, 1) 如果是2分类则-1可能是1000
                                      topk_ind).view(batch, K) # (1, 500) 获取前500个最高分的ind在原scoers中的ind
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                    topk_ind).view(batch, K) # (1, 500) 获取前500个最高分的y轴坐标在原map中的位置
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                    topk_ind).view(batch, K) # (1, 500) 获取前500个最高分的x轴坐标在原map中的位置

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            ind (torch.Tensor): Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous() # (1, 2, 128, 128) --> (1, 128, 128, 2)
        feat = feat.view(feat.size(0), -1, feat.size(3)) # (1, 16384, 2)
        feat = self._gather_feat(feat, ind) # 提取对应的特征：ind:（1, 500） 
        return feat # (1, 500, 2)

    def encode(self):
        pass

    def decode(self,
               heat,
               rot_sine,
               rot_cosine,
               hei,
               dim,
               vel,
               reg=None,
               task_id=-1):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor): Regression value of the boxes in 2D with
                the shape of [B, 2, W, H]. Default: None.
            task_id (int): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size() # 根据heatmap获取batch size和cat数目
        # ---------------------------------
        # 1.获取前500个分数对应的元素 --> (1, 500)
        # ---------------------------------
        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num) 

        # ---------------------------------
        # 2.根据inds提取回归值并恢复坐标值
        # ---------------------------------
        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds) # 取出前500个最高分对应的偏移预测值 (1, 500, 2)
            reg = reg.view(batch, self.max_num, 2) # (1, 500, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1] # 恢复中心点: 中心点 + 偏移值 (1, 500, 1)
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds) # (1, 500, 1)
        rot_sine = rot_sine.view(batch, self.max_num, 1) # (1, 500, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds) # (1, 500, 1)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1) # (1, 500, 1)
        rot = torch.atan2(rot_sine, rot_cosine) # 恢复yaw角 --> (1, 500, 1)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds) # (1, 500, 1)
        hei = hei.view(batch, self.max_num, 1) # # (1, 500, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds) # (1, 500, 3)
        dim = dim.view(batch, self.max_num, 3) # (1, 500, 3)

        # class label
        clses = clses.view(batch, self.max_num).float() # (1, 500)
        scores = scores.view(batch, self.max_num) # (1, 500)

        xs = xs.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0] # 特征图坐标*缩放尺度*voxle_size + 偏移量 --> 恢复真实坐标 (1, 500, 1)
        ys = ys.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds) # (1, 500, 2)
            vel = vel.view(batch, self.max_num, 2) # (1, 500, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2) # (1, 500, 9)

        final_scores = scores # (1, 500)
        final_preds = clses # (1, 500)

        # -------------------------------------------------------
        # 3.根据分数和点云范围计算mask对提取的bbox进行filter，并逐帧处理
        # -------------------------------------------------------
        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold # 0.1 --> (1, 500)

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device) # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
            # 保证预测的最终结果坐标在self.post_center_range范围内
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2) # (1, 500) all中的2指定维度
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2) # (1, 500)

            predictions_dicts = []
            # 逐帧处理(task内)
            for i in range(batch):
                cmask = mask[i, :] # 取出该帧mask (500, )
                if self.score_threshold:
                    cmask &= thresh_mask[i]  # 合并mask (500, )
                # 根据mask提取对应bbox
                boxes3d = final_box_preds[i, cmask] # (8, 9)
                scores = final_scores[i, cmask] # (8, 1)
                labels = final_preds[i, cmask] # (8, 1)
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict) # 逐帧加入
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts
