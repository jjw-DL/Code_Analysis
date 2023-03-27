# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from torch.nn import functional as F

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi
from mmdet.core import build_assigner, build_sampler
from mmdet.models import HEADS
from ..builder import build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead


@HEADS.register_module()
class CenterPointROIHead(Base3DRoIHead):
    """CenterPoint roi head for CenterPoint.

    Args:
        num_classes (int): The number of classes.
        bbox_roi_extractor (ConfigDict): Config of bbox_roi_extractor.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 num_classes=10,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterPointROIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes

        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor= build_roi_extractor(bbox_roi_extractor)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def init_bbox_head(self, bbox_head):
        """Initialize box head."""
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self):
        """Initialize mask head."""
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
    
    def forward_train(self, bev_feats, proposal_list,
                      gt_bboxes_3d, gt_labels_3d, img_metas):
        """Training forward function of CenterPointROIHead.

        Args:
            bev_feats (tensor): shared feature map (2, 64, 180, 180)
            proposal_list (list[dict]): Proposal information from rpn.
                The dictionary should contain the following keys:
                - boxes_3d (:obj:`BaseInstance3DBoxes`): Proposal bboxes
                - labels_3d (torch.Tensor): Labels of proposals
                - cls_preds (torch.Tensor): Original scores of proposals
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                GT bboxes of each sample. The bboxes are encapsulated
                by 3D box structures.
            gt_labels_3d (list[LongTensor]): GT labels of each sample.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            dict: losses from each head.
                - loss_bbox (torch.Tensor): loss of bboxes
        """
        losses = dict()
        sample_results = self._assign_and_sample(proposal_list, gt_bboxes_3d, gt_labels_3d)
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(bev_feats, sample_results)
            losses.update(bbox_results['loss_rcnn'])
        return losses # 在BaseDetector中会进行_parse_losses，进行loss求和

    def _bbox_forward_train(self, bev_feats, sampling_results):
        """Forward training function of roi_extractor and bbox_head.

        Args:
            bev_feats (torch.Tensor): bev feature map.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """
        rois = bbox3d2roi([res.bboxes for res in sampling_results]) # 对各帧的roi添加batch id并拼接
        bbox_results = self._bbox_forward(bev_feats, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)
        loss_rcnn = self.bbox_head.loss(bbox_results['cls_score'], # (256, 1)
                                        bbox_results['bbox_pred'], # (256, 9)
                                        rois, # (256, 9)
                                        self.train_cfg,
                                        *bbox_targets)

        bbox_results.update(loss_rcnn=loss_rcnn)
        return bbox_results

    def _bbox_forward(self, bev_feats, rois):
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            bev_feats (torch.Tensor): bev feature mao.
            rois (Tensor): Roi boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_bbox_feats = self.bbox_roi_extractor(bev_feats, rois) # (256, 320)
        cls_score, bbox_pred = self.bbox_head(pooled_bbox_feats) # (256, 1)和(256, 9)
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            pooled_bbox_feats=pooled_bbox_feats)
        return bbox_results

    def _assign_and_sample(self, proposal_list, gt_bboxes_3d, gt_labels_3d):
        """Assign and sample proposals for training.

        Args:
            proposal_list (list[dict]): Proposals produced by RPN.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels

        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        sampling_results = []
        # bbox assign and sample 逐帧处理
        for batch_idx in range(len(proposal_list)):
            cur_proposal_list = proposal_list[batch_idx] # 提取proposal list
            cur_boxes = cur_proposal_list['boxes_3d'] # 提取bbox3d
            cur_labels_3d = cur_proposal_list['labels_3d'] # 提取lable3d
            cur_gt_bboxes = gt_bboxes_3d[batch_idx].to(cur_boxes.device) # 提取gt_bboxes
            cur_gt_labels = gt_labels_3d[batch_idx] # 提取gt lables

            batch_num_gts = 0
            # 0 is bg 将proposal的gt_ind初始化为0
            batch_gt_indis = cur_gt_labels.new_full((len(cur_boxes), ), 0) 
            # 将proposal的max iou初始化为0
            batch_max_overlaps = cur_boxes.tensor.new_zeros(len(cur_boxes)) 
            # -1 is bg 将proposal的label初始化为-1
            batch_gt_labels = cur_gt_labels.new_full((len(cur_boxes), ), -1) 

            # each class may have its own assigner
            for i in range(self.num_classes):
                gt_per_cls = (cur_gt_labels == i) # 当前gt类别索引
                pred_per_cls = (cur_labels_3d == i) # 计算当前proposal类别索引
                cur_assign_res = self.bbox_assigner.assign(
                    cur_boxes.tensor[pred_per_cls],
                    cur_gt_bboxes.tensor[gt_per_cls],
                    gt_labels=cur_gt_labels[gt_per_cls])
                # gather assign_results in different class into one result
                batch_num_gts += cur_assign_res.num_gts # 累加gt的数量
                # gt inds (1-based)
                gt_inds_arange_pad = gt_per_cls.nonzero(
                    as_tuple=False).view(-1) + 1 # 将gt ind + 1
                # pad 0 for indice unassigned
                gt_inds_arange_pad = F.pad(
                    gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                # pad -1 for indice ignore
                gt_inds_arange_pad = F.pad(
                    gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                # convert to 0~gt_num+2 for indices
                gt_inds_arange_pad += 1
                # now 0 is bg, >1 is fg in batch_gt_indis 二次索引
                batch_gt_indis[pred_per_cls] = gt_inds_arange_pad[
                    cur_assign_res.gt_inds + 1] - 1
                batch_max_overlaps[
                    pred_per_cls] = cur_assign_res.max_overlaps # 当前类别proposal和gt最大的iou
                batch_gt_labels[pred_per_cls] = cur_assign_res.labels # 当前类别proposal的分配label

            # 构造assign result
            assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                            batch_max_overlaps,
                                            batch_gt_labels)
            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes.tensor,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results


    def simple_test(self, bev_feats, proposal_list, img_metas,
                        **kwargs):
            """Simple testing forward function of CenterPointROIHead.

            Note:
                This function assumes that the batch size is 1

            Args:
                bev_feats (Tensor): shared bev feature map
                img_metas (list[dict]): Meta info of each image.
                proposal_list (list[dict]): Proposal information from rpn.

            Returns:
                dict: Bbox results of one frame.
            """
            assert self.with_bbox, 'Bbox head must be implemented.'
            rois = bbox3d2roi([res['boxes_3d'].tensor for res in proposal_list])
            labels_3d = [res['labels_3d'] for res in proposal_list]
            cls_preds = [res['scores_3d'] for res in proposal_list]
            bbox_results = self._bbox_forward(bev_feats, rois)

            bbox_list = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                labels_3d,
                cls_preds,
                img_metas,
                cfg=self.test_cfg)

            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            return bbox_results