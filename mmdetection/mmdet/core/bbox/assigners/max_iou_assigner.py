# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes. 判断为正例的iou阈值
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes. 判断为负例的iou阈值
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            防止gt没有对应的bbox，设置判断为正例的最小的iou阈值
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
            是否将所有与某个 gt 重叠度最高的 bbox 分配给该 gt
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes. 
            忽略 bbox 的 IoF 阈值（如果指定了 `gt_bboxes_ignore`）负值意味着不忽略任何 bbox。
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
            是否计算 `bboxes` 和 `gt_bboxes_ignore` 之间的 iof，或相反
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
            是否允许低质量匹配。 这对于 RPN 和单级检测器通常是允许的，但在第二级是不允许的，步骤 4 中演示了详细信息
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
        GPU 分配的 GT 数量的上限, 当 gt 的数量高于此阈值时, 将在 CPU 设备上分配, 负值表示不在 CPU 上分配
    """

    def __init__(self,
                 pos_iou_thr, # 0.7
                 neg_iou_thr, # 0.3
                 min_pos_iou=.0, # 0.3
                 gt_max_assign_all=True, # 将所有与某个gt重叠度最高的bbox分配给该gt
                 ignore_iof_thr=-1, # 不忽略任何box
                 ignore_wrt_candidates=True, # 计算bboxes和gt_bboxes_ignore之间的 iof
                 match_low_quality=True, # 允许低质量匹配
                 gpu_assign_thr=-1, # 不在CPU上分配
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr # 0.7
        self.neg_iou_thr = neg_iou_thr # 0.3
        self.min_pos_iou = min_pos_iou # 0.3
        self.gt_max_assign_all = gt_max_assign_all # True 将所有与某个gt重叠度最高的bbox分配给该gt
        self.ignore_iof_thr = ignore_iof_thr # -1 不忽略任何box
        self.ignore_wrt_candidates = ignore_wrt_candidates # True 计算bboxes和gt_bboxes_ignore之间的 iof
        self.gpu_assign_thr = gpu_assign_thr # -1 不在CPU上分配
        self.match_low_quality = match_low_quality # True 允许低质量匹配
        self.iou_calculator = build_iou_calculator(iou_calculator) # 构建iou求解器

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background 将每个proposal分配为背景-1
        2. assign proposals whose iou with all gts < neg_iou_thr to 分配负例
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox 分配正例
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself 为每个gt分配最近的proposal

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        # 1.判断是否在CPU上执行
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False # False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device # 获取bbox的device
            bboxes = bboxes.cpu() # 将bbox转换到cpu
            gt_bboxes = gt_bboxes.cpu() # 将gt box转换到cpu
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu() # 将gt_bboxes_ignore转换到cpu
            if gt_labels is not None:
                gt_labels = gt_labels.cpu() # 将gt label转换到cpu

        overlaps = self.iou_calculator(gt_bboxes, bboxes) # 计算gt与proposal之间的iou--> (k, n)

        # 2.处理igonre的bbox
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof') # 计算bboxes和gt_bboxes_ignore之间的iof
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1) # 计算bboxes和gt_bboxes_ignore之间最大的iof-->(n,)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0) # (n,)
            # 将bboxes和gt_bboxes_ignore之间的最大iof大于阈值的设置为-1
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1 
        # ---------------------------------------------
        # 3.根据iou为proposal分配gt，返回的是AssignResult类
        # ---------------------------------------------
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)

        # 4.如果之前在cpu上计算，这里将AssignResult结果转换回GPU
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1) # gt和bbox的数量 k, n

        # -----------------------------------
        # 1. assign -1 by default
        # -----------------------------------
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        # 处理没有gt或没有proposal的情况
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, )) # 将proposal的max iou全部设置为0
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0 # 将所有proiosal的gt inds设置为背景
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long) # 将proposal的label设置为-1
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels) # 返回AssignResult

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0) # (k, n) --> (n,)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1) # (k, n) --> (k,)

        # -----------------------------------
        # 2. assign negative: below
        # -----------------------------------
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0 # 将负例的gt inds设置为0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # ------------------------------------------------
        # 3. assign positive: above positive IoU threshold
        # ------------------------------------------------
        pos_inds = max_overlaps >= self.pos_iou_thr # 计算proposal正例的索引
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1 # 将正例的gt_inds + 1赋值到对应位置

        if self.match_low_quality:
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.
            # 逐个gt处理
            for i in range(num_gts):
                # 如果该gt和proposal的最大iou > 正例最小iou阈值
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all: # 将所有与该gt重叠度最高的bbox分配给该gt
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i] # proposal索引
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else: # 只将与该gt重叠度最高的bbox分配给该gt
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
        
        # ------------------------------------------------
        # 4.处理gt labels
        # ------------------------------------------------
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1) # 将proposal的labels初始化为-1
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze() # 根据proposal的gt inds计算正例索引
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1] # 为proposal的正例索引赋予label，减一是因为前面加1了
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
