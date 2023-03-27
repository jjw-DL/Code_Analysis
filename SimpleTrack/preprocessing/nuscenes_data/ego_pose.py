import os, numpy as np, nuscenes, argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../raw/nuscenes/data/sets/nuscenes/')
parser.add_argument('--data_folder', type=str, default='../../../datasets/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
args = parser.parse_args()


def main(nusc, scene_names, root_path, ego_folder, mode):
    pbar = tqdm(total=len(scene_names)) # 创建进度条
    # 逐个scene处理
    for scene_index, scene_info in enumerate(nusc.scene):
        scene_name = scene_info['name'] # 获取scene名称
        if scene_name not in scene_names: # 如果该scene名称不在val中，则跳过
            continue
        first_sample_token = scene_info['first_sample_token'] # 获取该scene的第一帧的token
        last_sample_token = scene_info['last_sample_token'] # 获取该scene的最后一帧的token
        frame_data = nusc.get('sample', first_sample_token) # 获取第一帧的sample record
        if args.mode == '20hz':
            cur_sample_token = frame_data['data']['LIDAR_TOP'] # 取该seq第一帧中sample data的lidar token
        elif args.mode == '2hz':
            cur_sample_token = deepcopy(first_sample_token)
        frame_index = 0
        ego_data = dict() # 初始化ego_data的list
        while True:
            if mode == '2hz':
                frame_data = nusc.get('sample', cur_sample_token) # 获取该sample的record
                lidar_token = frame_data['data']['LIDAR_TOP'] # 获取该sample的lidar token
                lidar_data = nusc.get('sample_data', lidar_token) # 获取lidar data
                ego_token = lidar_data['ego_pose_token'] # 在lidar data中获取calib_token
                ego_pose = nusc.get('ego_pose', ego_token) # 根据calib_token获取标定信息
            elif mode == '20hz':
                frame_data = nusc.get('sample_data', cur_sample_token) # 这里已经是lidar data
                ego_token = frame_data['ego_pose_token'] # 获取calib_token
                ego_pose = nusc.get('ego_pose', ego_token) # 根据calib_token获取标定信息

            # translation + rotation 记录帧索引和对应的平移和旋转信息
            ego_data[str(frame_index)] = ego_pose['translation'] + ego_pose['rotation']

            # clean up and prepare for the next
            cur_sample_token = frame_data['next'] # 更新下一帧
            if cur_sample_token == '':
                break
            frame_index += 1 # 更新帧索引
        
        # 保存一个scene的全部自车信息
        np.savez_compressed(os.path.join(ego_folder, '{:}.npz'.format(scene_name)), **ego_data)
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    print('ego info')
    ego_folder = os.path.join(args.data_folder, 'ego_info')
    os.makedirs(ego_folder, exist_ok=True)

    val_scene_names = splits.create_splits_scenes()['val']
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, val_scene_names, args.raw_data_folder, ego_folder, args.mode)
