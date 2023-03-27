# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pickle
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from os import path as osp
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle) # 对mask进行解码
    return mask


def _parse_coco_ann_info(ann_info):
    # 初始化list
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []
    
    # 逐个标注处理
    for i, ann in enumerate(ann_info):
        # 如果该ann为忽略，则跳过
        if ann.get('ignore', False):
            continue
        # 获取该ann的bbox
        x1, y1, w, h = ann['bbox']
        # 如果该ann的面积 < 0, 则跳过
        if ann['area'] <= 0:
            continue
        # 将box组织为左上角和右下角的格式
        bbox = [x1, y1, x1 + w, y1 + h]
        # 如果box是crowd的，则加入gt_bboxes_ignore
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            # 否则，在gt bbox中加入该bbox
            gt_bboxes.append(bbox)
            # 加入分割mask
            gt_masks_ann.append(ann['segmentation'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32) # 加入gt boxes
        gt_labels = np.array(gt_labels, dtype=np.int64) # 加入gt label
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
        bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann # 返回gt bbox和gt mask


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair
    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = (
        torch.arange(num_pos,
                     device=device).to(dtype=pos_proposals.dtype)[:, None])
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks).to(device).index_select(
            0, pos_assigned_gt_inds).to(dtype=rois.dtype))
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = (
        roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1))
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0] # 获取正例的数量
    # 初始化mask和img patch的list
    masks = []
    img_patches = []
    # 逐个正例处理
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]] # 获取mask
        bbox = pos_proposals[i, :].astype(np.int32) # 获取gt
        x1, y1, x2, y2 = bbox # 获取该bbox对应的
        w = np.maximum(x2 - x1 + 1, 1) # 计算bbox的宽
        h = np.maximum(y2 - y1 + 1, 1) # 计算bbox的高

        mask_patch = gt_mask[y1:y1 + h, x1:x1 + w] # 获取mask patch
        masked_img = gt_mask[..., None] * org_img # 计算mask后的图像区域
        img_patch = masked_img[y1:y1 + h, x1:x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_mask=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset. --> NuScenesDataset
        data_path (str): Path of the data. --> ../data/nuscenes
        info_prefix (str): Prefix of the info file. --> nuscenes
        info_path (str): Path of the info file. --> nuscenes_infos_train.pkl
            Default: None.
        mask_anno_path (str): Path of the mask_anno.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        # 定义文件客户端
        file_client_args = dict(backend='disk')
        # 更新配置文件信息
        dataset_cfg.update(
            test_mode=False,
            split='training',
            # 模态信息
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=with_mask,
            ),
            pipeline=[
                # 加载点云
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    file_client_args=file_client_args),
                # 加载标注信息
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=file_client_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'WaymoDataset':
        file_client_args = dict(backend='disk')
        dataset_cfg.update(
            test_mode=False,
            split='training',
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=5,
                    file_client_args=file_client_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=file_client_args)
            ])

    # 根据配置文件构建数据集
    dataset = build_dataset(dataset_cfg)
    # 点云数据存储路径（文件夹）
    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database') # nuscenes_gt_database
    # infos存储路径（文件）
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl') # nuscenes_dbinfos_train.pkl
    # 创建文件夹
    mmcv.mkdir_or_exist(database_save_path)
    # 初始db_infos字典
    all_db_infos = dict()
    
    if with_mask:
        # 根据json文件路径构建COCO对象
        coco = COCO(osp.join(data_path, mask_anno_path))
        # 获取图片id
        imgIds = coco.getImgIds() # 共168780帧,168780/6=28130,nusenes训练集的全部关键帧数量
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0] # 根据id读取信息,主要包括filename,id, token,cam2ego,ego2global,cam_intrinsic,width,height
            file2id.update({info['file_name']: i}) # 将img id和filename对应上

    group_counter = 0
    # 逐帧处理
    for j in track_iter_progress(list(range(len(dataset)))):
        # 根据id读取infos信息
        """ For nuscenes
        - sample_idx (str): Sample index.
        - pts_filename (str): Filename of point clouds.
        - sweeps (list[dict]): Infos of sweeps.
        - timestamp (float): Sample timestamp.
        - img_filename (str, optional): Image filename.
        - lidar2img (list[np.ndarray], optional): Transformations \
            from lidar to different cameras.
        - ann_info (dict): Annotation info.
            - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
            - gt_labels_3d (np.ndarray): Labels of ground truths.
            - gt_names 
        """
        input_dict = dataset.get_data_info(j) # 根据索引获取输入
        dataset.pre_pipeline(input_dict) # 在input_dict中加入pre_pipeline的fild信息
        example = dataset.pipeline(input_dict) # 将数据送入pipeline处理(custome_3d中初始化)，处理之后数据格式统一，可能会增加不同字段
        annos = example['ann_info']
        image_idx = example['sample_idx'] # 在kitti中为点云id，在nuscenes为token
        points = example['points'].tensor.numpy() # 在pipeline的loading的LoadPointsFromFile将点云信息加入result的'points'字段
        gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
        names = annos['gt_names']
        # 初始化一帧点云中的group_id,包含全部object
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64) # 如果不存在group_id则按顺序初始化
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32) # 全部初始化为0
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0] # 获取该帧物体数量
        # 获取3D gt box内的点云索引 [N, M]：Indices of points in each box
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if with_mask: # 默认为False
            # prepare masks
            gt_boxes = annos['gt_bboxes'] # 获取该帧点云对应图像的2D box
            img_path = osp.split(example['img_info']['filename'])[-1] # 获取图像路径
            if img_path not in file2id.keys():
                print(f'skip image {img_path} for empty mask')
                continue
            img_id = file2id[img_path] # 根据图片路径获取对应的id
            kins_annIds = coco.getAnnIds(imgIds=img_id) # 获取该帧图片对应的ann id
            kins_raw_info = coco.loadAnns(kins_annIds) # 加载ann
            kins_ann_info = _parse_coco_ann_info(kins_raw_info) # 解析该ann
            h, w = annos['img_shape'][:2] # 获取图片高和宽
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ] # 计算gt的mask
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])
        
        # 逐个box处理
        for i in range(num_obj):
            # 图片id+类名+个数.bin
            filename = f'{image_idx}_{names[i]}_{i}.bin' # kitti:0_Pedestrian_0.bin nuscene:f9878012c..._car_2.bin
            abs_filepath = osp.join(database_save_path, filename) # 绝对路径
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename) # 相对路径 nuscenes_gt_database/f9878012c..._car_2.bin

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]] # 截取gt box内点云
            gt_points[:, :3] -= gt_boxes_3d[i, :3] # 从lidar系转换到local坐标系，所有点云的起点都是0

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f) # 将点云写入文件

            # 将类别信息写入
            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                # local group的id
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id] # 就是该帧中的第几个物体
                if 'score' in annos:
                    db_info['score'] = annos['score'][i] # 一般不含分数
                if with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info) # 根据类别进行添加
                else:
                    all_db_infos[names[i]] = [db_info]
    # 加载database infos信息
    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')
    # 将all_db_infos写入文件
    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
