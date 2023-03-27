# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import torch

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')


def _draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyz'):
    """Draw points on visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.

    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size 设置点的大小
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy() # 将点从Tensor转化为numpy

    points = points.copy() # 复制点
    pcd = geometry.PointCloud() # 创建open3d的点云
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3]) # 为open3d的点云添加Vector3dVector的点
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1)) # 构造点云颜色
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors) # 为点云添加颜色
    vis.add_geometry(pcd) # 在visualizer中添加点云

    return pcd, points_colors # 返回点云和对应的颜色信息


def _draw_bboxes(bbox3d,
                 vis,
                 points_colors,
                 pcd=None,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
    """Draw bbox on visualizer and change the color of points inside bbox3d.

    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`): point cloud. Default: None.
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points inside bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy() # 将box从Tensor形式转换为numpy
    bbox3d = bbox3d.copy() # 复制bbox

    in_box_color = np.array(points_in_box_color) # bbox内点云的颜色
    # 逐个bbox绘制
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3] # 提取bbox的中心
        dim = bbox3d[i, 3:6] # 提取bbox的长宽高

        yaw = np.zeros(3)
        yaw[rot_axis] = -bbox3d[i, 6]  # 提取bbox的yaw角
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw) # 计算旋转矩阵

        # 根据不同模式,调整点云中心z坐标
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim) # 旋转bbox

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d) # 根据bbox计算边线
        line_set.paint_uniform_color(bbox_color) # 为边线上颜色
        # draw bboxes on visualizer
        vis.add_geometry(line_set) # 在visualizer中添加边线

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points) # 获取在该bbox内的点云索引
            points_colors[indices] = in_box_color # 为bbox内的点云添加颜色

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors) # 更新点云颜色
        vis.update_geometry(pcd)


def show_pts_boxes(points,
                   bbox3d=None,
                   show=True,
                   save_path=None,
                   points_size=2,
                   point_color=(0.5, 0.5, 0.5),
                   bbox_color=(0, 1, 0),
                   points_in_box_color=(1, 0, 0),
                   rot_axis=2,
                   center_mode='lidar_bottom',
                   mode='xyz'):
    """Draw bbox and points on visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize. Default: None.
        show (bool): whether to show the visualization results. Default: True.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # TODO: support score and class info
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer() # 创建visualizer
    vis.create_window() # 创建窗口
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame 创建坐标系
    vis.add_geometry(mesh_frame) # 添加坐标系

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode) # 绘制点云

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes(bbox3d, vis, points_colors, pcd, bbox_color,
                     points_in_box_color, rot_axis, center_mode, mode) # 绘制bbox

    if show:
        vis.run() # 显示点云

    if save_path is not None:
        vis.capture_screen_image(save_path) # 保存截图

    vis.destroy_window() # 销毁窗口


def _draw_bboxes_ind(bbox3d,
                     vis,
                     indices,
                     points_colors,
                     pcd=None,
                     bbox_color=(0, 1, 0),
                     points_in_box_color=(1, 0, 0),
                     rot_axis=2,
                     center_mode='lidar_bottom',
                     mode='xyz'):
    """Draw bbox on visualizer and change the color or points inside bbox3d
    with indices.

    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        indices (numpy.array | torch.tensor, shape=[N, M]):
            indicate which bbox3d that each point lies in.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`): point cloud. Default: None.
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy() # 将bbox转换为numpy格式
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy() # 将indice转换为numpy格式
    bbox3d = bbox3d.copy() # 复制bbox3d

    in_box_color = np.array(points_in_box_color) # 将box内的点云颜色转换为numpy格式
    # 逐个bbox处理
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3] # 提起bbox的中心
        dim = bbox3d[i, 3:6] # 提取bbox的长宽和高
        yaw = np.zeros(3)
        # TODO: fix problem of current coordinate system
        # dim[0], dim[1] = dim[1], dim[0]  # for current coordinate
        # yaw[rot_axis] = -(bbox3d[i, 6] - 0.5 * np.pi)
        yaw[rot_axis] = -bbox3d[i, 6] # 提取yaw角
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw) # 根据yaw角计算旋转矩阵
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d) # 计算Line Set
        line_set.paint_uniform_color(bbox_color) # 为Line Set添加颜色
        # draw bboxes on visualizer
        vis.add_geometry(line_set) # 在vis中添加Lise Set

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            points_colors[indices[:, i].astype(np.bool)] = in_box_color # 为indice中的点云赋予对应颜色

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors) # 更新点云颜色
        vis.update_geometry(pcd) # 更新点云


def show_pts_index_boxes(points,
                         bbox3d=None,
                         show=True,
                         indices=None,
                         save_path=None,
                         points_size=2,
                         point_color=(0.5, 0.5, 0.5),
                         bbox_color=(0, 1, 0),
                         points_in_box_color=(1, 0, 0),
                         rot_axis=2,
                         center_mode='lidar_bottom',
                         mode='xyz'):
    """Draw bbox and points on visualizer with indices that indicate which
    bbox3d that each point lies in.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize. Default: None.
        show (bool): whether to show the visualization results. Default: True.
        indices (numpy.array | torch.tensor, shape=[N, M]):
            indicate which bbox3d that each point lies in. Default: None.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # TODO: support score and class info
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes 根据indices绘制bbox
    if bbox3d is not None:
        _draw_bboxes_ind(bbox3d, vis, indices, points_colors, pcd, bbox_color,
                         points_in_box_color, rot_axis, center_mode, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


class Visualizer(object):
    r"""Online visualizer implemented with Open3d.

    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7]): 3d bbox (x, y, z, dx, dy, dz, yaw)
            to visualize. The 3d bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 save_path=None,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer() # 创建Visualizer句柄
        self.o3d_visualizer.create_window() # 创建窗口
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame # 创建坐标系
        self.o3d_visualizer.add_geometry(mesh_frame) # 在visualizer中添加坐标系

        self.points_size = points_size # 2
        self.point_color = point_color # (0.5, 0.5, 0.5) 点云颜色
        self.bbox_color = bbox_color # (0, 1, 0) box的颜色
        self.points_in_box_color = points_in_box_color # (1, 0, 0) box中的点云颜色
        self.rot_axis = rot_axis # 2
        self.center_mode = center_mode # 'lidar_bottom'
        self.mode = mode # 'xyz'
        self.seg_num = 0

        # draw points 绘制点云
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        # draw boxes 
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        """Add bounding box to visualizer.

        Args:
            bbox3d (numpy.array, shape=[M, 7]):
                3D bbox (x, y, z, dx, dy, dz, yaw) to be visualized.
                The 3d bbox is in mode of Box3DMode.DEPTH with
                gravity_center (please refer to core.structures.box_3d_mode).
            bbox_color (tuple[float]): the color of bbox. Defaule: None.
            points_in_box_color (tuple[float]): the color of points which
                are in bbox3d. Defaule: None.
        """
        # 设置添加的bbox的颜色
        if bbox_color is None:
            bbox_color = self.bbox_color
        # 设置bbox内点的颜色
        if points_in_box_color is None:
            points_in_box_color = self.points_in_box_color
        # 绘制该bbox
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
                     bbox_color, points_in_box_color, self.rot_axis,
                     self.center_mode, self.mode)

    def add_seg_mask(self, seg_mask_colors):
        """Add segmentation mask to visualizer via per-point colorization.

        Args:
            seg_mask_colors (numpy.array, shape=[N, 6]):
                The segmentation mask whose first 3 dims are point coordinates
                and last 3 dims are converted colors.
        """
        # we can't draw the colors on existing points
        # in case gt and pred mask would overlap
        # instead we set a large offset along x-axis for each seg mask
        self.seg_num += 1
        offset = (np.array(self.pcd.points).max(0) -
                  np.array(self.pcd.points).min(0))[0] * 1.2 * self.seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        self.o3d_visualizer.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        _draw_points(
            seg_points, self.o3d_visualizer, self.points_size, mode='xyzrgb')

    def show(self, save_path=None):
        """Visualize the points cloud.

        Args:
            save_path (str): path to save image. Default: None.
        """

        self.o3d_visualizer.run() # 显示点云

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path) # 保存截图

        self.o3d_visualizer.destroy_window() # 销毁窗口
        return
