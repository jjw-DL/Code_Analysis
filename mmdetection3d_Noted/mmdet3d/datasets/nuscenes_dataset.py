# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False):
        self.load_interval = load_interval # 默认为1(全部加载)，可以设置为8，只使用1/8的数据集
        self.use_valid_flag = use_valid_flag # 默认False
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity = with_velocity # True
        self.eval_version = eval_version # 'detection_cvpr_2019'
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version) # 初始化评估配置
        if self.modality is None: # 设置使用模态
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        # 先根据valid falg过滤部分gt
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])
        # 将剩余gt从name映射为id
        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp'])) # 根据时间戳对data_info进行排序
        data_infos = data_infos[::self.load_interval] # 如果存在load_interval(eg:8)则每隔8帧取一个
        self.metadata = data['metadata'] # 获取data的metadata信息
        self.version = self.metadata['version'] # 获取metadata中的version信息
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.
        1.获取点云路径，标记以及时间戳等信息
        2.获取图像路径和标定信息
        3.获取标注信息
        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index] # 根据索引返回info信息
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'], # 样本标记
            pts_filename=info['lidar_path'], # 点云存储路径
            sweeps=info['sweeps'], # sweep信息
            timestamp=info['timestamp'] / 1e6, # 时间戳
        )
        # 如果使用相机模态
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path']) # 获取图片路径
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation']) # lidar到camera的旋转矩阵
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T # lidar到camera的平移向量
                lidar2cam_rt = np.eye(4) # 初始化4x4的单位矩阵
                lidar2cam_rt[:3, :3] = lidar2cam_r.T # 左上角3x3作为旋转矩阵
                lidar2cam_rt[3, :3] = -lidar2cam_t # 左下角1x3作为平移向量

                intrinsic = cam_info['cam_intrinsic'] # 获取相机内参
                viewpad = np.eye(4) # 初始化4x4的单位矩阵
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic # 左上角3x3作为内参矩阵
                lidar2img_rt = (viewpad @ lidar2cam_rt.T) # lidar到camera图像平面的投影矩阵
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                )) # 在input_dict中加入camera信息

        # 如果不是测试模式，则获取标注信息
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index] # 根据index获取info信息
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        # 根据valid_flag对box进行过滤
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        # 处理lable
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat)) # 将gt_name对应的类别转换为id(数字)
            else:
                gt_labels_3d.append(-1) # 如果不再当前的有效类别，则添加-1
        gt_labels_3d = np.array(gt_labels_3d) # 转换为np.array格式

        # 对速度进行处理
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0] # 将无法估计速度的物体的速度设置为0
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1) # 在gt_box3d的最后一维拼接速度，变为9维

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, # gt box3d
            gt_labels_3d=gt_labels_3d, # lables 数字
            gt_names=gt_names_3d) # gt names 类别名字
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        # 逐帧处理
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det) # [NuScenesBox]
            sample_token = self.data_infos[sample_id]['token'] # token
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version) # 将box从lidar-->ego-->global坐标系
            # 逐个box处理
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label] # 将num映射为str
                # 根据速度进行属性判断
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2: # 如果速度大于0.2m/s
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]: # 如果是车
                        attr = 'vehicle.moving' # 则属性为移动的车
                    elif name in ['bicycle', 'motorcycle']: # 如果是自行车或者摩托车，则属性为骑行者
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name] # 否则根据类别赋值默认属性
                else: # 如果速度小于0.2m/s
                    if name in ['pedestrian']: # 如果是行人
                        attr = 'pedestrian.standing' # 则属性为站着的行人
                    elif name in ['bus']:
                        attr = 'vehicle.stopped' # 如果是bus，则属性为停止的车辆
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name] # 否则根据类别赋值默认属性

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(), # lyft没有速度
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr) # lyft没有属性 --> 构造nusc_anno
                annos.append(nusc_anno) # 将nusc_anno加入annos列表
            nusc_annos[sample_token] = annos # 将该帧的标注按照token加入nusc_annos字典
        nusc_submissions = {
            'meta': self.modality, # 构造meta信息
            'results': nusc_annos, # 将nusc_annos赋值到results字段中
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path) # 写入json文件
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1]) # /tmp/xxx/result/pts_bbox
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False) # 根据version和data_root构造NuScenes对象
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc, # NuScenes对象
            config=self.eval_detection_configs, # 评价配置文件
            result_path=result_path, # json文件路径
            eval_set=eval_set_map[self.version], # v1.0-trainval
            output_dir=output_dir, # 结果输出文件夹
            verbose=False)
        # --------------------------
        # 运行eval的main方法
        # --------------------------
        nusc_eval.main(render_curves=False) # 运行eval

        # record metrics 记录评价
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes' # pts_bbox_NuScenes
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                # pts_bbox_NuScenes/car_AP_dist_0.5
                # pts_bbox_NuScenes/car_AP_dist_1.0
                # pts_bbox_NuScenes/car_AP_dist_2.0
                # pts_bbox_NuScenes/car_AP_dist_4.0
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val 
            for k, v in metrics['label_tp_errors'][name].items():
                # pts_bbox_NuScenes/car_trans_err 平移
                # pts_bbox_NuScenes/car_scale_err 缩放
                # pts_bbox_NuScenes/car_orient_err 方向
                # pts_bbox_NuScenes/car_vel_err 速度
                # pts_bbox_NuScenes/car_attr_err 属性
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                # pts_bbox_NuScenes/mATE
                # pts_bbox_NuScenes/mASE
                # pts_bbox_NuScenes/mAOE
                # pts_bbox_NuScenes/mAVE
                # pts_bbox_NuScenes/mAAE
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score'] # pts_bbox_NuScenes/NDS
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap'] # pts_bbox_NuScenes/mAP
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory() # 构造临时文件夹 /tmp/xxx
            jsonfile_prefix = osp.join(tmp_dir.name, 'results') # 拼接前缀路径+results
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]: # name:pts_bbox
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results] # 将'pts_bbox'去除，内部dict重组
                tmp_file_ = osp.join(jsonfile_prefix, name) # 拼接临时文件路径+pts_bbox
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)}) # {pts_bbox:'/tmp/xxx/results_nusc.json'}
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # Format results到json格式 jsonfile_prefix:None --> {pts_bbox:'/tmp/xxx/results_nusc.json'}和临时文件路径
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names: # ['pts_bbox']
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name]) # name:pts_bbox, 传入json文件的路径
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup() # 清理临时文件夹

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d'] # LidarInstance3DBoxes eg:(300, 9)
    scores = detection['scores_3d'].numpy() # eg:(300,)
    labels = detection['labels_3d'].numpy() # eg:(300,)

    box_gravity_center = box3d.gravity_center.numpy() # 获取box的重力中心 (300, 3)
    box_dims = box3d.dims.numpy() # 获取box的长宽和高 (300, 3)
    box_yaw = box3d.yaw.numpy() # 获取yaw角 (300,)
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2 # 转换yaw角

    box_list = []
    # 逐个bbox处理
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i]) # 将yaw角转换为四元数
        velocity = (*box3d.tensor[i, 7:9], 0.0) # 处理速度 (x, y, z) z的速度始终为0
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i], # 中心点坐标
            box_dims[i], # 长宽高
            quat, # 旋转
            label=labels[i], # 类别
            score=scores[i], # 分数
            velocity=velocity) # 速度-->构造Nuscenes box类
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    # 逐个box处理
    for box in boxes:
        # Move box to ego vehicle coord system 将box从lidar坐标系转换到自车坐标系
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        
        # filter det in ego. 根据距离过滤
        cls_range_map = eval_configs.class_range # 获取配置文件中的类别距离阈值
        radius = np.linalg.norm(box.center[:2], 2) # 计算当前box距离自车的范围
        det_range = cls_range_map[classes[box.label]] # 根据当前box的类别获取距离阈值
        if radius > det_range: # 如果当前距离超过距离阈值，则跳过
            continue
        # Move box to global coord system 将box从自车坐标系转换到全局坐标系
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation'])) 
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
