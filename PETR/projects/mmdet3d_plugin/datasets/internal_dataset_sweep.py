import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import mmcv
import random


@DATASETS.register_module()
class InternalDatasetSweep(Custom3DDataset):
    r"""Internal Dataset.
    """
    CLASSES = ('VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN')
    
    cams = ['center_camera_fov120', 'left_front_camera', 'left_rear_camera',\
            'rear_camera', 'right_rear_camera', "right_front_camera"]

    def __init__(self,
                 data_root,
                 ann_file=None,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 test_mode=False,
                 box_type_3d='LiDAR',
                 shuffle=False):

        self.shuffle = shuffle
        if self.shuffle:
            print("Building a shuffle dataset")

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            test_mode=test_mode,
            box_type_3d=box_type_3d)

        self.data_root = data_root
    
    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        if self.shuffle:
            random.seed(0)
            random.shuffle(data_infos)
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.
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
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        center2lidar = np.matrix(info['center2lidar'])
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6) # 时间戳
                image_paths.append(cam_info['data_path']) # 图片路径
                intrinsic = np.matrix(np.array(cam_info['cam_intrinsic']).reshape(3,3)) # 相机内参
                extrinsic = np.matrix(np.array(cam_info['extrinsic']).reshape(4,4)) # lidar2cam
                extrinsic = extrinsic @ center2lidar # center2cam
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = np.array((viewpad @ extrinsic)) # center2img

                intrinsics.append(viewpad) # 相机内参4x4
                extrinsics.append(extrinsic.T) # center2cam
                lidar2img_rts.append(lidar2img_rt) # center2img

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = np.array(info['gt_boxes'])
        gt_velocity = np.array([[0, 0]] * gt_bboxes_3d.shape[0])
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity] , axis=-1)
        gt_names_3d = info['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5,0.5,0.5)).convert_to(self.box_mode_3d)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d
        )
        
        return anns_results

