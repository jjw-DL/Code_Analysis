# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                          points_cam2img)
from ..builder import FUSION_LAYERS
from . import apply_3d_transformation


def point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True):
    """Obtain image features using points.
    1.根据点云变换，恢复原始点云
    2.根据外参将点云投影到原始图像上
    3.根据img_meta信息，对投影点进行数据增强变换（平移，缩放，旋转，翻转）
    4.计算采样比例，在[-1,1]之间
    5.调用F.grid_sample进行采样

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    """
        "T" stands for translation;
        "S" stands for scale;
        "R" stands for rotation;
        "HF" stands for horizontal flip;
        "VF" stands for vertical flip.
    """
    # 1.进行点云的逆变换,还原初始点云 [T, S, R]
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True) 

    # 2.将点云坐标映射为图像坐标
    pts_2d = points_cam2img(points, proj_mat)

    # 3.进行图像的逆变换 img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    # 3.1 图像缩放变换
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2 eg:(19221, 2)
    # 3.2 图像平移变换
    img_coors -= img_crop_offset
    # 3.3 图像翻转变换
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1 eg:(19221, 1)
    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    # 4.计算网格采样点坐标，必须在[-1, 1]之间
    # grid sample, the valid grid range should be in [-1,1]
    h, w = img_pad_shape # 448, 1440
    coor_y = coor_y / h * 2 - 1 # [0, 2] --> [-1, 1]
    coor_x = coor_x / w * 2 - 1
    
    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2 eg:[1, 1, 19221, 2]
    
    # 5.根据坐标进行采样
    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest' # ‘bilinear’
    point_features = F.grid_sample(
        img_features, # eg:(1, 128, 56, 180)
        grid, # eg:[1, 1, 19221, 2]
        mode=mode, # ‘bilinear’
        padding_mode=padding_mode, # zeros
        align_corners=align_corners)  # 1xCx1xN feats --> (1, 128, 1, 19221)

    return point_features.squeeze().t() # (19221, 128)


@FUSION_LAYERS.register_module()
class PointFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Defaults to 'LIDAR'.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(self,
                 img_channels, # 256
                 pts_channels, # 64
                 mid_channels, # 128
                 out_channels, # 128
                 img_levels=3, # [0, 1, 2, 3, 4]
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True, # True
                 fuse_out=False, # False
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True, # False
                 padding_mode='zeros',
                 lateral_conv=True):
        super(PointFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels] # [0, 1, 2, 3, 4]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels) # [256, 256, 256, 256, 256]
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels # [0, 1, 2, 3, 4]
        self.coord_type = coord_type # Lidar
        self.act_cfg = act_cfg # None
        self.activate_out = activate_out # True
        self.fuse_out = fuse_out # False
        self.dropout_ratio = dropout_ratio # 0
        self.img_channels = img_channels # 256
        self.aligned = aligned # True
        self.align_corners = align_corners # False
        self.padding_mode = padding_mode # 'zeros'

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i], # 256
                    mid_channels, # 128
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv) # 5层
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels), # 640-->128
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            ) # 640 --> 128
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        ) # 64 --> 128

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels), # 128-->128
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats, pts, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas) # 获取点云投影点图像多尺度拼接特征 --> (19221, 640)
        img_pre_fuse = self.img_transform(img_pts) # 线性层，将640维特征降为128维度，获取前融合图像特征 --> (19221, 128)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats) # 线性层，将64维特征升为128维度，获取前融合点云特征 --> (19221, 128)

        fuse_out = img_pre_fuse + pts_pre_fuse # 将点云特征和对应的图像特征融合 --> (19221, 128)
        if self.activate_out:
            fuse_out = F.relu(fuse_out) # 激活函数
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out # (19221, 128)

    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ] # 将特征图通道从256，降低为128 --> 一共5个通道
        else:
            img_ins = img_feats
        # 初始化每个点的特征
        img_feats_per_point = []
        # Sample multi-level features
        # 逐图片
        for i in range(len(img_metas)):
            mlvl_img_feats = []
            # 逐尺度
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
                                       img_metas[i])) # 获取点云投影点的多尺度特征，共5层 每层的特征都是（19221，128）
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1) # （19221， 640）
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0) # 将batch内的所有点云投影点对应的图像特征进行拼接，只有一帧点云-->（19221， 640）
        return img_pts

    def sample_single(self, img_feats, pts, img_meta):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (1, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1) # 图片缩放尺度
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False # 是否翻转
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0) # 图片偏移
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type) # 获取lidar2img的变换矩阵
        img_pts = point_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][:2],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        ) # 单尺度特征图采样
        return img_pts
