# Copyright (c) OpenMMLab. All rights reserved.
from black import re_compile_maybe_verbose
import torch
from torch.nn import functional as F

from mmcv.runner import force_fp32
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from ..builder import build_head
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class DynamicCenterPointRCNN(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 pts_roi_head=None,
                 img_rpn_head=None,
                 img_roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DynamicCenterPointRCNN,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

        if pts_roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            rcnn_test_cfg = test_cfg.rcnn if test_cfg is not None else None
            pts_roi_head.update(train_cfg=rcnn_train_cfg)
            pts_roi_head.update(test_cfg=rcnn_test_cfg)
            self.pts_roi_head = build_head(pts_roi_head)

    @property
    def with_pts_roi_head(self):
        """bool: whether the detector has a pts roi head"""
        return hasattr(self, 'pts_roi_head') and self.pts_roi_head is not None

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res) # （225432, 3)
            coors.append(res_coors)
        # 将点拼接
        points = torch.cat(points, dim=0)
        coors_batch = []
        # 添加batch信息
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        # 在batch维度拼接
        coors_batch = torch.cat(coors_batch, dim=0) # (449723, 4)
        return points, coors_batch

    @torch.no_grad()
    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors) # (152933, 5) 每个voxle取平均值
        batch_size = coors[-1, 0] + 1 # 计算batch_size eg:2
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size) # (2, 256, 180, 180)
        x = self.pts_backbone(x) # (2, 128, 180, 180) 和 (2, 256, 90, 90)
        if self.with_pts_neck:
            x = self.pts_neck(x) # (2, 512, 180, 180)
        return x
    
    @torch.no_grad()
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
            List[List[LidarInstance3DBoxes, score(tensor), lable(tensor)],...]: 外层表示帧数，内层的表示一帧的预测结果
            List[Tensor]: shared_feature eg:[(2, 64, 180, 180)] 多尺度特征，这里只有一个元素，即一个特征图
        """
        # 1.在这里进入centerpoint_head的forward函数
        # 返回list[dict]: Output results for tasks. 6个task_head(每个task head又包含6个分支)
        outs, bev_feats = self.pts_bbox_head(pts_feats) 
        # 2.经gt box，gt label和out freature组成list
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs] 
        # 3.进入centerpoint_head的loss函数
        losses = self.pts_bbox_head.loss(*loss_inputs)
        # 4.获得proposal_list
        proposal_list = []
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        for bboxes, scores, labels in bbox_list:
            proposal_list.append(dict(
                boxes_3d=bboxes,
                scores_3d=scores,
                labels_3d=labels))
        return losses, proposal_list, bev_feats

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # 1.提取图像和点云特征
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        losses = dict()
        # 2.rpn计算loss和proposal
        if pts_feats:
                losses_pts, proposal_list, bev_feats = self.forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d,
                                                                              img_metas,gt_bboxes_ignore)
                losses.update(losses_pts)
        
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        
        # 3.roi计算loss
        roi_losses = self.pts_roi_head.forward_train(bev_feats, proposal_list, 
                                                     gt_bboxes_3d, gt_labels_3d, img_metas)
        losses.update(roi_losses)

        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        # tupel(List[dict]) dict中有6个字段‘reg’,'height','dim','rot','vel','heatmap'
        outs, bev_feats = self.pts_bbox_head(x)
        proposal_list = []
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        for bboxes, scores, labels in bbox_list:
            proposal_list.append(dict(
                boxes_3d=bboxes,
                scores_3d=scores,
                labels_3d=labels))
        return self.pts_roi_head.simple_test(bev_feats, proposal_list, img_metas)

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool): Whether to rescale bboxes. Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        # 逐个增强特征处理
        for x, img_meta in zip(feats, img_metas):
            outs, bev_feats = self.pts_bbox_head(x) # 将feat送入head的到输出特征
            # merge augmented outputs before decoding bboxes
            # 逐个任务处理
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']: # 如果存在水平翻转
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2]) # 将该key值对应的任务在对应维度翻转(只是顺序变换)
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...] # 如果是reg，则将在该维度按照进行数值变换
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']: # 如果存在垂直翻转
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                    else:
                        final_feats = bev_feats
            outs_list.append(outs) # 将变换后的结果存入list-->List[Tuple(List(dict))]-->[([dict]), ([dict]), ...]

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor 相同尺度的预测进行相加
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor'] # 提取点云缩放尺度
            if pcd_scale_factor not in preds_dicts.keys(): # 新尺度出现是进入
                preds_dicts[pcd_scale_factor] = outs # 将预测结果放入对应缩放尺度
                scale_img_metas.append(img_meta) # 将图像的元信息放入scale_img_metas
            else:
                # 相同尺度出现时进入
                # 逐任务处理
                for task_id, out in enumerate(outs): 
                    # 逐个预测处理
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key] # 将翻转结果相加 --> eg:{'1.0':([dict])}

        aug_bboxes = []

        # 逐尺度
        for pcd_scale_factor, preds_dict in preds_dicts.items():
            # 逐任务处理
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys()) # 取平均值
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale) # 对合并后的bbox进行NMS等操作
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # 合并多尺度预测
        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes, final_feats
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0], final_feats

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs) # 提取变换后的多个特征图 List[List[Tensor]] (1, 512, 180, 180)
        if pts_feats and self.with_pts_bbox:
            pts_bbox, bev_feats = self.aug_test_pts(pts_feats, img_metas, rescale) # 增强Test
            for key in pts_bbox.keys():
                pts_bbox[key] = pts_bbox[key].to(bev_feats.get_device())
        return self.pts_roi_head.simple_test(bev_feats, [pts_bbox], img_metas[0])
