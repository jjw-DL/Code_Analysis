# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from ..bbox import bbox3d2result, bbox3d_mapping_back, xywhr2xyxyr


def merge_aug_bboxes_3d(aug_results, img_metas, test_cfg):
    """Merge augmented detection 3D bboxes and scores.

    Args:
        aug_results (list[dict]): The dict of detection results.
            The dict contains the following keys

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
        img_metas (list[dict]): Meta information of each sample.
        test_cfg (dict): Test config.

    Returns:
        dict: Bounding boxes results in cpu mode, containing merged results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Merged detection bbox.
            - scores_3d (torch.Tensor): Merged detection scores.
            - labels_3d (torch.Tensor): Merged predicted box labels.
    """

    assert len(aug_results) == len(img_metas), \
        '"aug_results" should have the same length as "img_metas", got len(' \
        f'aug_results)={len(aug_results)} and len(img_metas)={len(img_metas)}'
    
    # 初始化
    recovered_bboxes = []
    recovered_scores = []
    recovered_labels = []

    # 逐个增强结果处理
    for bboxes, img_info in zip(aug_results, img_metas):
        scale_factor = img_info[0]['pcd_scale_factor'] # 获取缩放因子
        pcd_horizontal_flip = img_info[0]['pcd_horizontal_flip'] # 获取水平翻转标志 False
        pcd_vertical_flip = img_info[0]['pcd_vertical_flip'] # 获取垂直翻转标志 False
        recovered_scores.append(bboxes['scores_3d']) # 添加预测分数
        recovered_labels.append(bboxes['labels_3d']) # 添加预测标签
        bboxes = bbox3d_mapping_back(bboxes['boxes_3d'], scale_factor,
                                     pcd_horizontal_flip, pcd_vertical_flip) # 将bbox映射会原始bbox
        recovered_bboxes.append(bboxes) # 添加bbox

    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes) # 合并增强后的bbox
    aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev) # 将bbox转化到BEV下，用于NNS
    aug_scores = torch.cat(recovered_scores, dim=0) # 合并分数
    aug_labels = torch.cat(recovered_labels, dim=0) # 合并标签

    # TODO: use a more elegent way to deal with nms
    if test_cfg.use_rotate_nms:
        nms_func = nms_gpu
    else:
        nms_func = nms_normal_gpu

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class nms when merge bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)
    
    # 逐类别处理
    for class_id in range(torch.max(aug_labels).item() + 1):
        class_inds = (aug_labels == class_id) # 获取类比索引
        bboxes_i = aug_bboxes[class_inds] # 根据索引提取bbox
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :] # 提取用于NMS的bbox
        scores_i = aug_scores[class_inds] # 提取对应分数
        labels_i = aug_labels[class_inds] # 提取对应标签
        if len(bboxes_nms_i) == 0:
            continue
        selected = nms_func(bboxes_nms_i, scores_i, test_cfg.nms_thr) # 进行NMS

        merged_bboxes.append(bboxes_i[selected, :]) # 添加被选中的Bbox
        merged_scores.append(scores_i[selected]) # 添加被选中的score
        merged_labels.append(labels_i[selected]) # 添加被选中的label

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes) # 合并bbox
    merged_scores = torch.cat(merged_scores, dim=0) # 合并score
    merged_labels = torch.cat(merged_labels, dim=0) # 合并label

    _, order = merged_scores.sort(0, descending=True) # 按照分数进行排序
    num = min(test_cfg.max_num, len(aug_bboxes)) # 最多保留的bbox数量
    order = order[:num] # 提取前num个索引

    merged_bboxes = merged_bboxes[order] # 提取前num个bbox
    merged_scores = merged_scores[order] # 提取前num个score
    merged_labels = merged_labels[order] # 提取前num个label

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)
