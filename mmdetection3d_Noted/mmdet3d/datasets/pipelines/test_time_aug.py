# Copyright (c) OpenMMLab. All rights reserved.
from cv2 import transform
import mmcv
import warnings
from copy import deepcopy

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to False. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to False. Note that it works only when
            'flip' is turned on.
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        self.transforms = Compose(transforms) # 将变换组合
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale] # img_scale必须为list
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else[float(pts_scale_ratio)] # 如果点云缩放尺度只有一个值，则转换为float

        assert mmcv.is_list_of(self.img_scale, tuple) # 断言当前的img_scale是否是一个tuple的seq
        assert mmcv.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip # True
        self.pcd_horizontal_flip = pcd_horizontal_flip # True
        self.pcd_vertical_flip = pcd_vertical_flip # True

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction] # 将翻转方向变为list --> ['horizontal']
        assert mmcv.is_list_of(self.flip_direction, str) # 断言翻转方向
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with \
                different scales and flips.  
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False] # [True]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False] # [False, True]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False] # [False, True]
        for scale in self.img_scale: # [(800, 1440),]
            for pts_scale_ratio in self.pts_scale_ratio: # 1.0
                for flip in flip_aug: # [True]
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug: # [False, True]
                        for pcd_vertical_flip in pcd_vertical_flip_aug: # [False, True]
                            for direction in self.flip_direction: # ['horizontal']
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results) # 对result进行深拷贝
                                _results['scale'] = scale # (800, 1440)
                                _results['flip'] = flip # True
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio # 1.0
                                _results['flip_direction'] = direction # 'Horizontal'
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip # False和True
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip # false和True
                                data = self.transforms(_results) # 对results进行数据增强变换，主要是flip和scale
                                aug_data.append(data) # 将增强后的数据加入aug_data中-->增强过程中会有4个翻转变换后的值（FF，FT，TF和TT）
        # list of dict to dict of list
        # 将data中的key初始化为空字典，只取第一个result就可以，因为其他的相同
        aug_data_dict = {key: [] for key in aug_data[0]}
        # 逐个增强数据处理，将相同的字段添加到一起
        for data in aug_data: # 经过Collect3D后，data中只有points和img_metas字段
            for key, val in data.items():
                aug_data_dict[key].append(val) 
        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
