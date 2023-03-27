import os, numpy as np, nuscenes, argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
import matplotlib.pyplot as plt
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../raw/nuscenes/data/sets/nuscenes/')
parser.add_argument('--data_folder', type=str, default='../../../datasets/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()


def load_pc(path):
    pc = np.fromfile(path, dtype=np.float32) # 读取点云
    pc = pc.reshape((-1, 5))[:, :4] # 获取点云的前四维
    return pc


def main(nusc, scene_names, root_path, pc_folder, mode, pid=0, process=1):
    # 逐个scene处理
    for scene_index, scene_info in enumerate(nusc.scene):
        if scene_index % process != pid:
            continue
        scene_name = scene_info['name'] # 获取scene名称
        if scene_name not in scene_names: # 如果该scene名称不在val中，则跳过
            continue
        print('PROCESSING {:} / {:}'.format(scene_index + 1, len(nusc.scene))) # eg: 3 / 850

        first_sample_token = scene_info['first_sample_token'] # 获取该scene的第一帧的token
        frame_data = nusc.get('sample', first_sample_token) # 获取该scene的最后一帧的token
        if mode == '20hz':
            cur_sample_token = frame_data['data']['LIDAR_TOP'] # 取该seq第一帧中sample data的lidar token
        elif mode == '2hz':
            cur_sample_token = deepcopy(first_sample_token)
        frame_index = 0
        pc_data = dict() #  初始化点云dict
        while True:
            # find the path to lidar data
            if mode == '2hz':
                lidar_data = nusc.get('sample', cur_sample_token) # 获取该sample的token
                # 根据lidar token获取点云保存路径
                lidar_path = nusc.get_sample_data_path(lidar_data['data']['LIDAR_TOP'])
            elif args.mode == '20hz':
                lidar_data = nusc.get('sample_data', cur_sample_token)
                lidar_path = lidar_data['filename']

            # load and store the data
            point_cloud = np.fromfile(os.path.join(root_path, lidar_path), dtype=np.float32) # 加载点云
            point_cloud = np.reshape(point_cloud, (-1, 5))[:, :4] # 提取点云前4维
            pc_data[str(frame_index)] = point_cloud # 保存点云

            # clean up and prepare for the next
            cur_sample_token = lidar_data['next'] # 更新下一帧
            if cur_sample_token == '':
                break
            frame_index += 1 # 更新帧索引

            if frame_index % 10 == 0:
                print('PROCESSING ', scene_index, ' , ', frame_index)
        # 保存该scene全部点云
        np.savez_compressed(os.path.join(pc_folder, '{:}.npz'.format(scene_name)), **pc_data)
    return


if __name__ == '__main__':
    print('point cloud')
    pc_folder = os.path.join(args.data_folder, 'pc', 'raw_pc')
    os.makedirs(pc_folder, exist_ok=True)
    
    val_scene_names = splits.create_splits_scenes()['val']
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, val_scene_names, args.raw_data_folder, pc_folder, args.mode, 0, 1)
