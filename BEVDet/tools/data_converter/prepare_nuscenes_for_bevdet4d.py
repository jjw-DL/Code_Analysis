# Copyright (c) Phigent Robotics. All rights reserved.

import pickle
import json
from nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion


def add_adj_info():
    interval = 3 # 间隔
    max_adj = 60 # 最大允许相邻帧
    for set in ['train', 'val']:
        dataset = pickle.load(open('./data/nuscenes/nuscenes_infos_%s.pkl' % set, 'rb')) # 读取infos文件
        nuscenes_version = 'v1.0-trainval'
        dataroot = './data/nuscenes/'
        nuscenes = NuScenes(nuscenes_version, dataroot) # 构建nuscenes类对象
        map_token_to_id = dict() # 初始化token到id的映射
        # 逐帧映射, 为每一个关键帧的token赋予id
        for id in range(len(dataset['infos'])): # (28130,)
            map_token_to_id[dataset['infos'][id]['token']] = id
        # 逐帧处理
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos']))) # 每隔10帧输出一次
            info = dataset['infos'][id] # 获取第id帧的info
            sample = nuscenes.get('sample', info['token']) # 通过sample的token获取sample的record
            
            # 处理相邻帧，分前向和后向单独处理
            for adj in ['next', 'prev']:
                sweeps = []
                adj_list = dict() # 初始化相邻帧dict
                # 逐个相机处理，找到每个相机的60个相邻帧
                for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                    adj_list[cam] = [] # 初始化单相机相邻帧list
                    # sample['data'][sensor_name]-->token, sample_data的record
                    sample_data = nuscenes.get('sample_data', sample['data'][cam]) 
                    count = 0
                    # 逐个相邻帧处理
                    while count < max_adj:
                        if sample_data[adj] == '': # 如果相邻帧的token为空，则beark
                            break
                        sd_adj = nuscenes.get('sample_data', sample_data[adj]) # 获取相邻帧的sample data
                        sample_data = sd_adj
                        # 在相邻帧list中加入相邻帧路径，时间戳和自车位姿token
                        adj_list[cam].append(dict(data_path='./data/nuscenes/' + sd_adj['filename'], # 图片路径
                                                  timestamp=sd_adj['timestamp'], # 时间戳
                                                  ego_pose_token=sd_adj['ego_pose_token'])) # 自车位姿token(重要, 后面要做位姿补偿)
                        count += 1 # 相邻帧+1
                
                # 至此，6个相机的相邻帧已经处理完毕，间隔处理相邻帧
                for count in range(interval - 1, min(max_adj, len(adj_list['CAM_FRONT'])), interval):
                    timestamp_front = adj_list['CAM_FRONT'][count]['timestamp'] # 获取前视相机时间戳
                    # get ego pose 获取前视相机第count相邻帧的自车位姿
                    pose_record = nuscenes.get('ego_pose', adj_list['CAM_FRONT'][count]['ego_pose_token'])

                    # get cam infos 初始化时加入相邻帧路径
                    cam_infos = dict(CAM_FRONT=dict(data_path=adj_list['CAM_FRONT'][count]['data_path']))
                    # 逐个相机处理，找到与前视相机最近的时间戳
                    for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                                'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                        # 将该相机对应的时间戳加入list
                        timestamp_curr_list = np.array([t['timestamp'] for t in adj_list[cam]], dtype=np.long) # (60,)
                        diff = np.abs(timestamp_curr_list - timestamp_front) # 计算和前视相机的时间戳差异 (60,)
                        selected_idx = np.argmin(diff) # 选最小时间间隔，表示该相机当前选中的帧 2
                        cam_infos[cam] = dict(data_path=adj_list[cam][int(selected_idx)]['data_path'])
                        # print('%02d-%s'%(selected_idx, cam))
                    # 在sweep中记录前视相机时间戳和对应相邻帧的6个相机的路径以及自车到全局的旋转和平移
                    sweeps.append(dict(timestamp=timestamp_front, cams=cam_infos,
                                       ego2global_translation=pose_record['translation'],
                                       ego2global_rotation=pose_record['rotation'])) # 正常有20帧
                dataset['infos'][id][adj] = sweeps if len(sweeps) > 0 else None # 将sweep加入dataset的infos中

            # get ego speed and transform the targets velocity from global frame into ego-relative mode
            previous_id = id # 初始化previous_id
            if not sample['prev'] == '':
                sample_tmp = nuscenes.get('sample', sample['prev']) # 获取该sample前一帧的token
                previous_id = map_token_to_id[sample_tmp['token']] # 根据token获取id
            next_id = id # 初始化next_id
            if not sample['next'] == '':
                sample_tmp = nuscenes.get('sample', sample['next']) # 获取该sample下一帧的token
                next_id = map_token_to_id[sample_tmp['token']] # 根据token获取id
            time_pre = 1e-6 * dataset['infos'][previous_id]['timestamp'] # 获取前一帧的时间戳
            time_next = 1e-6 * dataset['infos'][next_id]['timestamp'] # 获取下一帧的时间戳
            time_diff = time_next - time_pre # 计算时间差
            posi_pre = np.array(dataset['infos'][previous_id]['ego2global_translation'], dtype=np.float32) # 前一帧在全局坐标系下的位置
            posi_next = np.array(dataset['infos'][next_id]['ego2global_translation'], dtype=np.float32) # 下一帧在全局坐标系下的位置
            velocity_global = (posi_next - posi_pre) / time_diff # 位置差 / 时间差 = 速度

            l2e_r = info['lidar2ego_rotation'] # 获取lidar到自车的旋转
            l2e_t = info['lidar2ego_translation'] # 获取lidar到自车的平移
            e2g_r = info['ego2global_rotation'] # 获取自车到全局的旋转
            e2g_t = info['ego2global_translation'] # 获取自车到全局的平移
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix # 将旋转四元数转换为旋转矩阵
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            velocity_global = np.array([*velocity_global[:2], 0.0]) # 构造自车在全局系下的速度
            # V_global : global-->ego-->lidar
            velocity_lidar = velocity_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T # 自车在lidar系下的速度
            velocity_lidar = velocity_lidar[:2]

            dataset['infos'][id]['velo'] = velocity_lidar # 记录自车在lidar系下的速度
            # 计算box相对于自车的速度(都在lidar系下)
            dataset['infos'][id]['gt_velocity'] = dataset['infos'][id]['gt_velocity'] - velocity_lidar.reshape(1, 2) # eg:(11, 2)

        with open('./data/nuscenes/nuscenes_infos_%s_4d_interval%d_max%d.pkl' % (set, interval, max_adj), 'wb') as fid:
            pickle.dump(dataset, fid)

if __name__=='__main__':
    add_adj_info()