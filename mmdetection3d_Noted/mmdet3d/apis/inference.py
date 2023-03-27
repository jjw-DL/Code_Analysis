# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import re
import torch
from copy import deepcopy
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from os import path as osp

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result, show_result,
                          show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                # 将norm cfg的naiveSyncBN替换为BN
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config) # 根据配置文件路径读取并生成Config
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None # 将预训练模型设置为None
    convert_SyncBN(config.model) # 设置SyncBN
    config.model.train_cfg = None # 将train_cfg设置为None
    model = build_model(config.model, test_cfg=config.get('test_cfg')) # 构建模型
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu') # 加载权重文件
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names # 获取类别名称
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device) # 将模型放到GPU上
    model.eval() # 将模型设置为评估模式
    return model


def inference_detector(model, pcd):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg # 在init时已经将Config赋值到model中，这里取出来
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline) # 深拷贝test pipeline
    test_pipeline = Compose(test_pipeline) # 初始化Compose类，类内初始化transform sequential的各类
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d) # LiDARInstance3DBoxes和Box3DMode.LIDAR
    # 构造infos信息:input_dict
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[], # 传入的sweeps为空，在LoadPointsFromMultiSweeps时会重复9次
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    # 将input_dict送入pipeline
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def inference_multi_modality_detector(model, pcd, image, ann_file):
    """Inference point cloud with the multi-modality detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file) # 获取data_info
    image_idx = int(re.findall(r'\d+', image)[-1])  # xxx/sunrgbd_000017.jpg
    for x in data_infos:
        if int(x['image']['image_idx']) != image_idx:
            continue
        info = x
        break
    data = dict(
        pts_filename=pcd, # 点云
        img_prefix=osp.dirname(image), # 图片前缀
        img_info=dict(filename=osp.basename(image)), # 图片路径
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)

    # TODO: this code is dataset-specific. Move lidar2img and
    #       depth2img to .pkl annotations in the future.
    # LiDAR to image conversion
    if box_mode_3d == Box3DMode.LIDAR:
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        lidar2img = P2 @ rect @ Trv2c
        data['img_metas'][0].data['lidar2img'] = lidar2img # 计算转换矩阵
    # Depth to image conversion
    elif box_mode_3d == Box3DMode.DEPTH:
        rt_mat = info['calib']['Rt']
        # follow Coord3DMode.convert_point
        rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                           ]) @ rt_mat.transpose(1, 0)
        depth2img = info['calib']['K'] @ rt_mat
        data['img_metas'][0].data['depth2img'] = depth2img

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def inference_mono_3d_detector(model, image, ann_file):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file)
    # find the info corresponding to this image
    for x in data_infos['images']:
        if osp.basename(x['file_name']) != osp.basename(image):
            continue
        img_info = x
        break
    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=img_info['cam_intrinsic']))

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def inference_segmentor(model, pcd):
    """Inference point cloud with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    data = dict(
        pts_filename=pcd,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def show_det_result_meshlab(data,
                            result,
                            out_dir,
                            score_thr=0.0,
                            show=False,
                            snapshot=False):
    """Show 3D detection result by meshlab."""
    points = data['points'][0][0].cpu().numpy() # 将点转换为numpy格式
    pts_filename = data['img_metas'][0][0]['pts_filename'] # 提取点云文件名
    file_name = osp.split(pts_filename)[-1].split('.')[0] # 提取文件名

    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy() # 将bbox转化为预测的numpy格式
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy() # 将score转化为预测的numpy格式
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr # 根据分数过滤
        pred_bboxes = pred_bboxes[inds]

    # for now we convert points into depth mode
    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode != Box3DMode.DEPTH:
        points = points[..., [1, 0, 2]]
        points[..., 0] *= -1
        show_bboxes = Box3DMode.convert(pred_bboxes, box_mode, Box3DMode.DEPTH) # 将box转化为深度模式
    else:
        show_bboxes = deepcopy(pred_bboxes)

    show_result(
        points,
        None,
        show_bboxes,
        out_dir,
        file_name,
        show=show,
        snapshot=snapshot)

    return file_name


def show_seg_result_meshlab(data,
                            result,
                            out_dir,
                            palette,
                            show=False,
                            snapshot=False):
    """Show 3D segmentation result by meshlab."""
    points = data['points'][0][0].cpu().numpy()
    pts_filename = data['img_metas'][0][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    pred_seg = result[0]['semantic_mask'].numpy()

    if palette is None:
        # generate random color map
        max_idx = pred_seg.max()
        palette = np.random.randint(0, 256, size=(max_idx + 1, 3))
    palette = np.array(palette).astype(np.int)

    show_seg_result(
        points,
        None,
        pred_seg,
        out_dir,
        file_name,
        palette=palette,
        show=show,
        snapshot=snapshot)

    return file_name


def show_proj_det_result_meshlab(data,
                                 result,
                                 out_dir,
                                 score_thr=0.0,
                                 show=False,
                                 snapshot=False):
    """Show result of projecting 3D bbox to 2D image by meshlab."""
    assert 'img' in data.keys(), 'image data is not provided for visualization'

    img_filename = data['img_metas'][0][0]['filename']
    file_name = osp.split(img_filename)[-1].split('.')[0]

    # read from file because img in data_dict has undergone pipeline transform
    img = mmcv.imread(img_filename)

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode == Box3DMode.LIDAR:
        if 'lidar2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'LiDAR to image transformation matrix is not provided')

        show_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['lidar2img'],
            out_dir,
            file_name,
            box_mode='lidar',
            show=show)
    elif box_mode == Box3DMode.DEPTH:
        show_bboxes = DepthInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            None,
            out_dir,
            file_name,
            box_mode='depth',
            img_metas=data['img_metas'][0][0],
            show=show)
    elif box_mode == Box3DMode.CAM:
        if 'cam2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'camera intrinsic matrix is not provided')

        show_bboxes = CameraInstance3DBoxes(
            pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['cam2img'],
            out_dir,
            file_name,
            box_mode='camera',
            show=show)
    else:
        raise NotImplementedError(
            f'visualization of {box_mode} bbox is not supported')

    return file_name


def show_result_meshlab(data,
                        result,
                        out_dir,
                        score_thr=0.0,
                        show=False,
                        snapshot=False,
                        task='det',
                        palette=None):
    """Show result by meshlab.

    Args:
        data (dict): Contain data from pipeline.
        result (dict): Predicted result from model.
        out_dir (str): Directory to save visualized result.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.0
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
        task (str): Distinguish which task result to visualize. Currently we
            support 3D detection, multi-modality detection and 3D segmentation.
            Defaults to 'det'.
        palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Defaults to None.
    """
    # 根据不同的任务调用不同的API显示
    assert task in ['det', 'multi_modality-det', 'seg', 'mono-det'], \
        f'unsupported visualization task {task}'
    assert out_dir is not None, 'Expect out_dir, got none.'

    if task in ['det', 'multi_modality-det']:
        file_name = show_det_result_meshlab(data, result, out_dir, score_thr,
                                            show, snapshot)

    if task in ['seg']:
        file_name = show_seg_result_meshlab(data, result, out_dir, palette,
                                            show, snapshot)

    if task in ['multi_modality-det', 'mono-det']:
        file_name = show_proj_det_result_meshlab(data, result, out_dir,
                                                 score_thr, show, snapshot)

    return out_dir, file_name
