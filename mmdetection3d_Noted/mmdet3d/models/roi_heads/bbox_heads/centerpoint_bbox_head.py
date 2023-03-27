# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet3d.core.bbox.structures import (LiDARInstance3DBoxes,
                                          rotation_3d_in_axis, xywhr2xyxyr)
from mmdet3d.models.builder import build_loss
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class CenterPointBboxHead(BaseModule):
    """CenterPoint RoI head.

    Args:
        num_classes (int): The number of classes to prediction.
        in_channels (int): Input channels of roi features
        shared_fc_channels (list(int)): Out channels of each shared fc layer.
        cls_channels (list(int)): Out channels of each classification layer.
        reg_channels (list(int)): Out channels of each regression layer.
        dropout_ratio (float): Dropout ratio of classification and
            regression layers.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for box head.
        conv_cfg (dict): Config dict of convolutional layers
        norm_cfg (dict): Config dict of normalization layers
        loss_bbox (dict): Config dict of box regression loss.
        loss_cls (dict): Config dict of classifacation loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 shared_fc_channels=None,
                 cls_channels=None,
                 reg_channels=None,
                 dropout_ratio=0.1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=1.0),
                 init_cfg=None):
        super(CenterPointBboxHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes # 10
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        # shared layers
        shared_fc_list = []
        pre_channel = in_channels
        for k in range(0, len(shared_fc_channels)):
            shared_fc_list.append(
                ConvModule(
                    pre_channel,
                    shared_fc_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    inplace=True)) # 320->256->256
            pre_channel = shared_fc_channels[k]

            if k != len(shared_fc_channels) - 1 and dropout_ratio > 0:
                shared_fc_list.append(nn.Dropout(dropout_ratio))

        self.shared_fc = nn.Sequential(*shared_fc_list)

        # Classification layer
        channel_in = shared_fc_channels[-1]
        cls_channel = 1
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, len(cls_channels)):
            cls_layers.append(
                ConvModule(
                    pre_channel,
                    cls_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    inplace=True)) # 256->256-->256
            pre_channel = cls_channels[k]
        cls_layers.append(
            ConvModule(
                pre_channel,
                cls_channel,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                act_cfg=None)) # 256->1
        if dropout_ratio >= 0:
            cls_layers.insert(1, nn.Dropout(dropout_ratio))

        self.conv_cls = nn.Sequential(*cls_layers)

        # Regression layer
        reg_layers = []
        pre_channel = channel_in
        for k in range(0, len(reg_channels)):
            reg_layers.append(
                ConvModule(
                    pre_channel,
                    reg_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    inplace=True)) # # 256->256-->256
            pre_channel = reg_channels[k]
        reg_layers.append(
            ConvModule(
                pre_channel,
                self.bbox_coder.code_size,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                act_cfg=None)) # 256-->9
        if dropout_ratio >= 0:
            reg_layers.insert(1, nn.Dropout(dropout_ratio))

        self.conv_reg = nn.Sequential(*reg_layers)

        if init_cfg is None:
            self.init_cfg = dict(
                type='Xavier',
                layer=['Conv2d', 'Conv1d'],
                distribution='uniform')

    def init_weights(self):
        super().init_weights()
        normal_init(self.conv_reg[-1].conv, mean=0, std=0.001)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): pool bbox features.

        Returns:
            tuple[torch.Tensor]: Score of class and bbox predictions.
        """
        # (256, 320) --> (256, 1, 320) --> (256, 320, 1)
        pooled_features = x.reshape(-1, 1, x.shape[-1]).permute(0, 2, 1).contiguous()  
        shared_feature = self.shared_fc(pooled_features) # (256, 256, 1)

        cls_score = self.conv_cls(shared_feature).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (256, 1, 1)-->(256, 1)
        bbox_pred = self.conv_reg(shared_feature).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (256, 9, 1)-->(256, 9)

        return cls_score, bbox_pred

    def loss(self, cls_score, bbox_pred, rois, cfg, labels, bbox_targets,
             pos_gt_bboxes, reg_mask, label_weights, bbox_weights):
        """Coumputing losses.

        Args:
            cls_score (torch.Tensor): Scores of each roi.
            bbox_pred (torch.Tensor): Predictions of bboxes.
            rois (torch.Tensor): Roi bboxes.
            labels (torch.Tensor): Labels of class.
            bbox_targets (torch.Tensor): Target of positive bboxes.
            pos_gt_bboxes (torch.Tensor): Ground truths of positive bboxes.
            reg_mask (torch.Tensor): Mask for positive bboxes.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_weights (torch.Tensor): Weights of bbox loss.

        Returns:
            dict: Computed losses.

                - loss_cls (torch.Tensor): Loss of classes.
                - loss_bbox (torch.Tensor): Loss of bboxes.
        """
        losses = dict()
        rcnn_batch_size = cls_score.shape[0] # 获取batch数量 256

        # calculate class loss
        cls_flat = cls_score.view(-1) # (256,)
        loss_cls = self.loss_cls(cls_flat, labels, label_weights) # 计算iou cls损失
        losses['loss_cls_rcnn'] = loss_cls # 记录损失

        # calculate regression loss
        pos_inds = (reg_mask > 0) # 正例索引
        if pos_inds.any() == 0:
            # fake a part loss
            losses['loss_bbox'] = loss_cls.new_tensor(0)
        else:
            pos_bbox_pred = bbox_pred.view(rcnn_batch_size, -1)[pos_inds] # (256,9)-->(128, 9)
            bbox_weights_flat = bbox_weights[pos_inds].view(-1, 1).repeat(
                1, pos_bbox_pred.shape[-1]) # (128, 1)-->(128, 9)
            code_weights = cfg.get('code_weights', None)
            bbox_weights_flat = bbox_weights_flat * bbox_weights_flat.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred.unsqueeze(dim=0), bbox_targets.unsqueeze(dim=0),
                bbox_weights_flat.unsqueeze(dim=0))
            losses['loss_bbox_rcnn'] = loss_bbox

        return losses

    def get_targets(self, sampling_results, rcnn_train_cfg, concat=True):
        """Generate targets.

        Args:
            sampling_results (list[:obj:`SamplingResult`]):
                Sampled results from rois.
            rcnn_train_cfg (:obj:`ConfigDict`): Training config of rcnn.
            concat (bool): Whether to concatenate targets between batches.

        Returns:
            tuple[torch.Tensor]: Targets of boxes and class prediction.
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results] # 提取正例bbox
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results] # 提取正例对应的gt bbox
        iou_list = [res.iou for res in sampling_results] # 提取proposal对应的iou
        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            cfg=rcnn_train_cfg)

        (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
         bbox_weights) = targets

        if concat:
            label = torch.cat(label, 0) # (256,)
            bbox_targets = torch.cat(bbox_targets, 0).detach() # (54, 9)
            pos_gt_bboxes = torch.cat(pos_gt_bboxes, 0) # (54, 9)
            reg_mask = torch.cat(reg_mask, 0) # (256,)

            label_weights = torch.cat(label_weights, 0) # (256,)
            label_weights = label_weights / torch.clamp(label_weights.sum(), min=1.0) # (256,)

            bbox_weights = torch.cat(bbox_weights, 0) # (256,)
            bbox_weights = bbox_weights / torch.clamp(bbox_weights.sum(), min=1.0) # (256,)

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def _get_target_single(self, pos_bboxes, pos_gt_bboxes, ious, cfg):
        """Generate training targets for a single sample.

        Args:
            pos_bboxes (torch.Tensor): Positive boxes with shape
                (N, 7).
            pos_gt_bboxes (torch.Tensor): Ground truth boxes with shape
                (M, 7).
            ious (torch.Tensor): IoU between `proposal_bboxes` and `gt_bboxes`
                in shape (pos + neg,).
            cfg (dict): Training configs.

        Returns:
            tuple[torch.Tensor]: Target for positive boxes.
                (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)
        """
        cls_pos_mask = ious > cfg.cls_pos_thr # 0.75
        cls_neg_mask = ious < cfg.cls_neg_thr # 0.25
        interval_mask = (cls_pos_mask == 0) & (cls_neg_mask == 0) # 0.25 ~ 0.75

        # iou regression target
        label = (cls_pos_mask > 0).float() # 将iou > 0.75的分类label设置为1
        label[interval_mask] = ious[interval_mask] * 2 - 0.5 # 将 0.25 < iou < 0.75的分类lable按照该公式计算
        # label weights
        label_weights = (label >= 0).float() # label权重为全1 (128,)

        # box regression target
        reg_mask = pos_bboxes.new_zeros(ious.size(0)).long() # 初始化reg的mask
        reg_mask[0:pos_gt_bboxes.size(0)] = 1 # 将pos bbox的mask置1 (128,)
        bbox_weights = (reg_mask > 0).float() # 设置bbox的权重 (128,)
        # 对正例的bbox进行encode
        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach() # 提取正例对应的gt bbox
            roi_center = pos_bboxes[..., 0:3] # 正例roi bbox的中心
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi) # 正例roi bbox的yaw(限制在0～2pi)
            
            # canonical transformation(全局到局部的变换)
            pos_gt_bboxes_ct[..., 0:3] = pos_gt_bboxes_ct[..., 0:3] - roi_center # 计算gt与roi中心差
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1),
                -(roi_ry),
                axis=2).squeeze(1) # 将中心坐标绕Z轴旋转yaw的负值

            pos_gt_bboxes_ct[..., 6] = pos_gt_bboxes_ct[..., 6] - roi_ry # 计算gt与roi的偏航角差
            """
            roi_vel = pos_bboxes[:, :, 7:-1] # 正例roi bbox的速度
            pos_gt_bboxes_ct[..., 7:] -= roi_vel # 计算gt与roi的速度差
            pos_gt_vel = torch.cat([pos_gt_bboxes_ct[..., 7:], torch.zeros([roi_vel.shape[0], 1])], dim=-1)
            pos_gt_bboxes_ct[..., 7:] = rotation_3d_in_axis(
                pos_gt_vel.unsqueeze(1), 
                -(roi_ry),
                axis=2)[..., :2].squeeze(1) # 将速度绕Z轴旋转yaw的负值
            rois_anchor[:, 7:] = 0 # 速度为0
            """
            # flip orientation if rois have opposite orientation
            ry_label = pos_gt_bboxes_ct[..., 6] % (2 * np.pi)  # 将正例对应的gt的偏航角限制在0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5) # (pi/2, 3pi/2）
            ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (
                2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = ry_label > np.pi
            ry_label[flag] = ry_label[flag] - np.pi * 2 
            ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2) # 限制在-pi/2至pi/2
            pos_gt_bboxes_ct[..., 6] = ry_label

            rois_anchor = pos_bboxes.clone().detach() # 提取正例的roi坐标
            # 前面都已经计算过了，所以encode的时候设置为0
            rois_anchor[:, 0:3] = 0 # 中心坐标为0
            rois_anchor[:, 6] = 0 # yaw为0
            
            bbox_targets = self.bbox_coder.encode(rois_anchor,
                                                  pos_gt_bboxes_ct) # 计算bbox的target
        else:
            # no fg bbox
            bbox_targets = pos_gt_bboxes.new_empty((0, 7))

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def get_bboxes(self,
                   rois,
                   cls_score, # 预测分数
                   bbox_pred, # 预测bbox
                   class_labels, # bbox类别标签
                   class_pred, # 第一阶段预测分数
                   img_metas,
                   cfg=None):
        """Generate bboxes from bbox head predictions.

        Args:
            rois (torch.Tensor): Roi bounding boxes.
            cls_score (torch.Tensor): Scores of bounding boxes.
            bbox_pred (torch.Tensor): Bounding boxes predictions
            class_labels (torch.Tensor): Label of classes List(Tensor)
            class_pred (torch.Tensor): Score for nms. List(Tensor)
            img_metas (list[dict]): Point cloud and image's meta info.
            cfg (:obj:`ConfigDict`): Testing config.

        Returns:
            list[tuple]: Decoded bbox, scores and labels after nms.
        """
        roi_batch_id = rois[..., 0] # batch 索引
        roi_boxes = rois[..., 1:]  # boxes without batch id
        batch_size = int(roi_batch_id.max().item() + 1) # batch size

        # decode boxes
        roi_ry = roi_boxes[..., 6].view(-1)
        roi_xyz = roi_boxes[..., 0:3].view(-1, 3)
        local_roi_boxes = roi_boxes.clone().detach()
        local_roi_boxes[..., 0:3] = 0
        rcnn_boxes3d = self.bbox_coder.decode(local_roi_boxes, bbox_pred)
        rcnn_boxes3d[..., 0:3] = rotation_3d_in_axis(
            rcnn_boxes3d[..., 0:3].unsqueeze(1), (roi_ry),
            axis=2).squeeze(1)
        rcnn_boxes3d[:, 0:3] = rcnn_boxes3d[:, 0:3] + roi_xyz

        # post processing
        result_list = []
        for batch_id in range(batch_size):
            cur_class_labels = class_labels[batch_id] # 当前帧的类别 eg:(498,)

            cur_cls_score = cls_score[roi_batch_id == batch_id].view(-1)
            cur_cls_score = torch.sigmoid(cur_cls_score) # 当前帧的预测iou分数 eg:(498,)
            cur_box_prob = class_pred[batch_id] # 当前帧的分类分数 eg:(498,)

            cur_rcnn_boxes3d = rcnn_boxes3d[roi_batch_id == batch_id] # 当前帧的预测bbox eg:(498, 9)
            
            result_list.append(
                (img_metas[batch_id]['box_type_3d'](cur_rcnn_boxes3d,
                                                    self.bbox_coder.code_size),
                 cur_box_prob, cur_class_labels))
            
            # selected = self.multi_class_nms(cur_class_labels, scores, cur_rcnn_boxes3d,
            #                                 cfg.score_thr, cfg.nms_thr,
            #                                 img_metas[batch_id],
            #                                 cfg.use_rotate_nms)
            # selected_bboxes = cur_rcnn_boxes3d[selected]
            # selected_label_preds = cur_class_labels[selected]
            # selected_scores = scores[selected]

            # result_list.append(
            #     (img_metas[batch_id]['box_type_3d'](selected_bboxes,
            #                                         self.bbox_coder.code_size),
            #      selected_scores, selected_label_preds))
        return result_list

    def multi_class_nms(self,
                        bbox_labels,
                        box_probs,
                        box_preds,
                        score_thr,
                        nms_thr,
                        input_meta,
                        use_rotate_nms=True):
        """Multi-class NMS for box head.

        Note:
            This function has large overlap with the `box3d_multiclass_nms`
            implemented in `mmdet3d.core.post_processing`. We are considering
            merging these two functions in the future.

        Args:
            bbox_labels (torch.Tensor): Predicted boxes labels in shape(N,)
            box_probs (torch.Tensor): Predicted boxes probabitilies in
                shape (N,).
            box_preds (torch.Tensor): Predicted boxes in shape (N, 7+C).
            score_thr (float): Threshold of scores.
            nms_thr (float): Threshold for NMS.
            input_meta (dict): Meta informations of the current sample.
            use_rotate_nms (bool, optional): Whether to use rotated nms.
                Defaults to True.

        Returns:
            torch.Tensor: Selected indices.
        """
        if use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu

        ont_hot_label = torch.eye(10) # (10, 10)
        device = box_probs.get_device()
        pseudo_labels = ont_hot_label[bbox_labels.long()].to(device) # eg:(498, 10)
        box_probs = pseudo_labels * box_probs.unsqueeze(1) # eg:(498, 10)

        selected_list = []
        # 将bbox转换到BEV视角
        boxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            box_preds, self.bbox_coder.code_size).bev) # eg:(498, 5)

        score_thresh = score_thr if isinstance(
            score_thr, list) else [score_thr for x in range(self.num_classes)] # (10,)
        nms_thresh = nms_thr if isinstance(
            nms_thr, list) else [nms_thr for x in range(self.num_classes)] # (10,)
        
        for k in range(0, self.num_classes):
            class_scores_keep = box_probs[:, k] >= score_thresh[k] # 该类别分数大于阈值的索引

            if class_scores_keep.int().sum() > 0:
                original_idxs = class_scores_keep.nonzero(
                    as_tuple=False).view(-1) # 计算原索引的id
                cur_boxes_for_nms = boxes_for_nms[class_scores_keep] # 提取需要NMS的bbox
                cur_rank_scores = box_probs[class_scores_keep, k] # 提取对应的分数

                cur_selected = nms_func(cur_boxes_for_nms, cur_rank_scores,
                                        nms_thresh[k]) # 进行NMS

                if cur_selected.shape[0] == 0:
                    continue
                selected_list.append(original_idxs[cur_selected])

        selected = torch.cat(
            selected_list, dim=0) if len(selected_list) > 0 else []
        return selected
