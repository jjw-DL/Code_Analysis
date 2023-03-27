"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap


def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP'] # 获取sample data的token
    ref_sd_rec = nusc.get('sample_data', ref_sd_token) # 获取sample data的record
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token']) # 获取ego pose的record
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token']) # 获取calibrated_sensor的record
    ref_time = 1e-6 * ref_sd_rec['timestamp'] # 获取时间戳

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True) # 构造ego到global的变换矩阵

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP'] # 获取sample data的token
    current_sd_rec = nusc.get('sample_data', sample_data_token) # 获取当前sample data的record
    # 逐个sweep处理
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename'])) # 读取该帧点云
        current_pc.remove_close(min_distance) # 移除近点

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token']) # 获取该帧的位姿
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False) # 构造ego到global的变换矩阵

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token']) # 获取calibrated_sensor的record
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False) # 构造从lidar到ego的变化矩阵

        # Fuse four transformation matrices into one and perform transform.
        # lidar-->ego-->global-->ego
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix) # 将当点云变换到当前自车坐标系下

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp'] # 计算时间延迟
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0) # 为每个点添加时间延时
        points = np.concatenate((points, new_points), 1) # 将点云拼接

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev']) # 更新当前sample的record为前一帧

    return points


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
       rot: from cam to ego car
       trans: from cam to ego car
       P_ego = rot * P_cam + trans
       P_cam = rot^(-1) * (P_ego - trans)
    """
    points = points - trans.unsqueeze(1) # (3, N) - (3, 1) 
    points = rot.permute(1, 0).matmul(points) # rot的转置就是逆

    points = intrins.matmul(points) # 将cam系下的点转换到相机坐标系下
    points[:2] /= points[2:3] # 从相机坐标系转换到像素坐标系

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1) # 根据图像深度和图像宽高过滤点云-->mask


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image (1600, 900)
    img = img.resize(resize_dims) # 图像缩放 (330, 185)
    img = img.crop(crop) # 图像裁剪 (352, 128)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT) # (352, 128)
    img = img.rotate(rotate) # 图像旋转 (352, 128)

    # post-homography transformation
    post_rot *= resize # 2D旋转矩阵缩放 eg:[[1, 0], [0, 1]] --> [[0.2172, 0], [0, 0.2172]]
    post_tran -= torch.Tensor(crop[:2]) # 2D平移向量 [0, 0] --> [0, -45]
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0]) # (352, 0) --> 这里并不是旋转+平移而是将翻转变换转换成这种形式，方便处理
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi) # 根据旋转角计算对应的旋转矩阵 --> [[0.9966, 0.0824], [-0.0824, 0.9966]]
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2 # eg:(176, 64) 裁剪宽度的一半，旋转中心点
    b = A.matmul(-b) + b # 相当于先将坐标原点从左上角移动到图像中心，然后绕中心旋转
    post_rot = A.matmul(post_rot) # 旋转矩阵(连乘即可)
    post_tran = A.matmul(post_tran) + b # R1*T2+T1

    return img, post_rot, post_tran # 获取经过缩放，裁剪，翻转和旋转等变换后对应的旋转矩阵和平移向量


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0) # (135577, 64) 沿着行累加
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool) # 135577
        kept[:-1] = (ranks[1:] != ranks[:-1]) # trick:坐标在同一个pillar内的点只保留一个

        x, geom_feats = x[kept], geom_feats[kept] # 提取被保留的voxel特征和geo特征 (35769, 64)和(35769, 4)
        x = torch.cat((x[:1], x[1:] - x[:-1])) # (35769, 64) 计算每个pillar内的特征，并与第一行拼接

        # save kept for backward
        ctx.save_for_backward(kept)# 几何特征不求梯度

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0) # 对于一维特征，沿着列累加
        back[kept] -= 1 # 减一变为索引

        val = gradx[back] # 提取上下文特征对应位置的梯度，几何特征不求梯度

        return val, None, None


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt) # (4, 1, 200, 200) --> eg:0.8427
        return loss


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'iou': total_intersect / total_union,
            }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85 # 车宽, 车长4.084
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]] # x和y交换，因为y方向为车长
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    """
    rec: 该sample的record
    nusc_maps(dict): 4个NuScenesMap类
    nusc: Nuscenes类
    scene2map(dict): 每个scene对应的地图位置
    dx: (0.5, 0.5)
    bx: (-49.75, -49.75)
    """
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token']) # 获取该sample自车位姿record
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']] # 获取该sample对应的map

    rot = Quaternion(egopose['rotation']).rotation_matrix # 将自车位姿从四元数转换为旋转矩阵
    rot = np.arctan2(rot[1, 0], rot[0, 0]) # 绕z轴旋转，根据旋转矩阵计算yaw角
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)]) # xy中心坐标和yaw角

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    # 返回ploy和line在自车坐标系下的坐标dict{'road_segment':[], 'lane':[], road_divider':[], 'lane_divider':[]}
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    # 逐个layer处理
    for name in poly_names:
        # 逐个poly处理
        for la in lmap[name]:
            pts = (la - bx) / dx # 计算该poly的图像坐标
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2) # 绘制该poly
    # 逐线绘制
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)


def get_local_map(nmap, center, stretch, layer_names, line_names):
    """
    nmap: NuScenesMap类
    center: xy中心坐标和yaw角
    stretch: 周围距离
    layer_names: 多变形层的名字 ['road_segment', 'lane']
    line_names: 线的名字 ['road_divider', 'lane_divider']
    """
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    ) # 局部地图的范围坐标(左上角和右下角)

    polys = {}

    # polygons 获取与特定矩形块相交或位于特定矩形块内的所有record token
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    # 逐层处理
    for layer_name in layer_names: # ['road_segment', 'lane']
        polys[layer_name] = []
        # 逐个token处理
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token) # 获取该polygon的record
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']] # 获取该record内所有polygon的token
            # 逐个polygon处理
            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token) # 构造shapley ploygon对象
                polys[layer_name].append(np.array(polygon.exterior.xy).T) # 获取该polygon的坐标，并添加对应的层

    # lines
    for layer_name in line_names: # ['road_divider', 'lane_divider']
        polys[layer_name] = []
        for record in getattr(nmap, layer_name): # 获取该layer所有的record
            token = record['token'] # 获取该record的token

            line = nmap.extract_line(record['line_token']) # 根据token获取shapely LineString对象
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                ) # 在polys对应层中添加该polygon的坐标

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T # 计算yaw角度
    # 逐层处理
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            # 将polygon的坐标转换到自车坐标系下
            polys[layer_name][rowi] -= center[:2] # 减平移
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot) # 乘旋转矩阵

    return polys
