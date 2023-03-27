import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import DynamicScatter
from .. import builder
from ..builder import VOXEL_ENCODERS
from .utils import VFELayer, get_paddings_indicator


@VOXEL_ENCODERS.register_module()
class DynamicVoxelEncoder(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 15.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[], # [64, 64]
                 with_distance=False, # False
                 with_cluster_center=False, # True
                 with_voxel_center=False, # True
                 voxel_size=(0.2, 0.2, 4), # [0.05, 0.05, 0.1]
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1), # [0, -40, -3, 70.4, 40, 1]
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max'):
        super(DynamicVoxelEncoder, self).__init__()
        assert mode in ['avg', 'max'] # max
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels # 15 + 3 + 3 = 21
        self._with_distance = with_distance # False
        self._with_cluster_center = with_cluster_center # True
        self._with_voxel_center = with_voxel_center # True
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0] # 0.075
        self.vy = voxel_size[1] # 0.075
        self.vz = voxel_size[2] # 0.2
        self.x_offset = self.vx / 2 + point_cloud_range[0] # -53.9625
        self.y_offset = self.vy / 2 + point_cloud_range[1] # -53.9625
        self.z_offset = self.vz / 2 + point_cloud_range[2] # -4.9
        self.point_cloud_range = point_cloud_range # [-54, -54, -5.0, 54, 54, 3.0]
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True) # 初始化DynamicScatter并返回该类
        
        feat_channels = [self.in_channels] + list(feat_channels) # [21, 64, 64]
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i] # 21-->64
            out_filters = feat_channels[i + 1] # 64-->64
            if i > 0:
                in_filters *= 2 # 128
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True))) # 21-->64和128-->64
        self.vfe_layers = nn.ModuleList(vfe_layers)

        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max')) # 初始化vfe的Scatter
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True) # 初始化cluster的Scatter

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point. eg:(19221, 4)
            voxel_mean (torch.Tensor): Voxel features to be mapped. eg:(16161, 4)
            voxel_coors (torch.Tensor): Coordinates of valid voxels eg:(16161, 4)

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz) # 40
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy) # 1600
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx) # 1408
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1 # 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size # 90112000
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long) #(90112000,)
        # Only include non-empty pillars 计算非空voxle的索引
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3]) # (16161,) 有效体素索引
        # Scatter the blob back to the canvas 在对应体素索引处设置索引，类似hash过程
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device) # 在canvas的有效体素索引处记录线性索引

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3]) # (19221,) 这里的pts_coors在voxle过程中存在重复
        # 根据voxle_index利用canvas做桥梁，建立pts_coors和voxel_coors之间的关系
        voxel_inds = canvas[voxel_index.long()] # 在canvas对应位置取voxel_coors的索引
        center_per_point = voxel_mean[voxel_inds, ...] # (19221, 4) 提取每个点对应的voxle mean特征
        return center_per_point

    @force_fp32(out_fp16=True)
    def forward(self, features, coors):
        """Forward functions.
           
        Args: 
            features (torch.Tensor): Features of voxels, shape is NxC. (19221, 15)
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim). (19221, 4)

        Returns:
            tuple: return voxel features and its coordinates.
        """
        features_ls = [features[:, :5]] # (19221, 4)
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors) # 计算包含点的有效voxel的平均值及其坐标（16160， 4）
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors) # 计算每个点所属的voxel_mean值(这里第一个参数是coors，表示每个点的voxle坐标)-->(19221, 4)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3] # 将每个点的特征减去所属voxle的mean值
            features_ls.append(f_cluster) # (19221, 3)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3)) # (19221, 3) 这里的features是点的实际坐标，不是体素坐标
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset) # 体素坐标coors乘以voxle大小+偏移
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center) # (19221, 3)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features_ls.append(features[:, 5:])
        features = torch.cat(features_ls, dim=-1) # (19221, 21)

        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features) # 21-->64 128-->64 # (19221, 64)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors) # (16160, 64), (16160, 4)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1) # (19221, 128)

        return voxel_feats, voxel_coors