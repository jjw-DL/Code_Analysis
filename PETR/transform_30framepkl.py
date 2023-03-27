import pickle
from refile import smart_open
from pyquaternion import Quaternion
import numpy as np
import os
import mmcv
import tqdm
sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
# info_prefix = 'train'
# info_prefix = 'val'
info_prefix = 'test'
info_path = os.path.join("/data/Dataset/nuScenes/",'mmdet3d_nuscenes_30f_infos_{}.pkl'.format(info_prefix)) # 构造输出文件路径
with smart_open('/data/Dataset/nuScenes/nuscenes_infos_{}.pkl'.format(info_prefix), "rb") as f:
    key_infos = pickle.load(f) ####nuscenes pkl 读取nuscenes的info
with smart_open('/data/Dataset/nuScenes/mmdet3d_key_nuscenes_12hz_infos_{}.pkl'.format(info_prefix), "rb") as f:
    key_sweep_infos = pickle.load(f) #### pkl contains previous key frames as sweep data, previous key frames has already aligned with current frame
with smart_open('/data/Dataset/nuScenes/nuscenes_12hz_infos_{}.pkl'.format(info_prefix), "rb") as f:
    sweep_infos = pickle.load(f) #### pkl contains origin sweep frames as sweep data, sweep frames has not aligned with current frame


num_prev = 5  ### previous key frame 
# 逐个关键帧处理
for current_id in tqdm.tqdm(range(len(sweep_infos))):
    ####current frame parameters 获取当前帧的参数
    e2g_t = key_infos['infos'][current_id]['ego2global_translation']
    e2g_r = key_infos['infos'][current_id]['ego2global_rotation']
    l2e_t = key_infos['infos'][current_id]['lidar2ego_translation']
    l2e_r = key_infos['infos'][current_id]['lidar2ego_rotation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    sweep_lists = [] # 逐个处理当前帧前一帧的参数
    for i in range(num_prev):  #### previous key frame
        sample_id = current_id - i # 前一帧sample id
        if sample_id < 0 or len(sweep_infos[sample_id]['sweeps']) == 0 or i >= len(key_sweep_infos['infos'][current_id]['sweeps']):
            continue
        for sweep_id in range(5): ###sweep frame for each previous key frame
            if len(sweep_infos[sample_id]['sweeps'][sweep_id].keys()) != 6:
                print(sample_id, sweep_id, sweep_infos[sample_id]['sweeps'][sweep_id].keys())
                temp = sweep_lists[-1]
                sweep_lists.append(temp)
                continue
            else:
                sweep_cams = dict()
                for view in sweep_infos[sample_id]['sweeps'][sweep_id].keys(): # view表示相机名称
                    sweep_cam = dict()
                    sweep_cam['data_path'] = '/data/Dataset/nuScenes/'+ sweep_infos[sample_id]['sweeps'][sweep_id][view]['filename']
                    sweep_cam['type'] = 'camera'
                    sweep_cam['timestamp'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['timestamp']
                    sweep_cam['is_key_frame'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['is_key_frame']
                    sweep_cam['nori_id'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['nori_id']
                    sweep_cam['sample_data_token'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['sample_token']
                    sweep_cam['ego2global_translation']  = sweep_infos[sample_id]['sweeps'][sweep_id][view]['ego_pose']['translation']
                    sweep_cam['ego2global_rotation']  = sweep_infos[sample_id]['sweeps'][sweep_id][view]['ego_pose']['rotation']
                    sweep_cam['sensor2ego_translation']  = sweep_infos[sample_id]['sweeps'][sweep_id][view]['calibrated_sensor']['translation']
                    sweep_cam['sensor2ego_rotation']  = sweep_infos[sample_id]['sweeps'][sweep_id][view]['calibrated_sensor']['rotation']
                    sweep_cam['cam_intrinsic'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['calibrated_sensor']['camera_intrinsic']

                    l2e_r_s = sweep_cam['sensor2ego_rotation']
                    l2e_t_s = sweep_cam['sensor2ego_translation'] 
                    e2g_r_s = sweep_cam['ego2global_rotation']
                    e2g_t_s = sweep_cam['ego2global_translation'] 

                    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
                    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                    # T-1帧的sensor到T帧的lidar的旋转和平移
                    sweep_cam['sensor2lidar_rotation'] = R.T  # points @ R.T + T
                    sweep_cam['sensor2lidar_translation'] = T

                    lidar2cam_r = np.linalg.inv(sweep_cam['sensor2lidar_rotation'])
                    lidar2cam_t = sweep_cam['sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = np.array(sweep_cam['cam_intrinsic'])
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    sweep_cam['intrinsics'] = viewpad # 该帧相机内参
                    sweep_cam['extrinsics'] = lidar2cam_rt # lidar2cam，右乘
                    sweep_cam['lidar2img'] = lidar2img_rt # lidar2img，左乘

                    pop_keys = ['ego2global_translation', 'ego2global_rotation', 'sensor2ego_translation', 'sensor2ego_rotation', 'cam_intrinsic']
                    [sweep_cam.pop(k) for k in pop_keys] # 弹出无用字段
                    # sweep_cam= sweep_cam.pop(pop_keys)
                    sweep_cams[view] = sweep_cam # 记录该帧信息，dict
                sweep_lists.append(sweep_cams) # 将该帧信息加入list
        ##key frame
        sweep_lists.append(key_sweep_infos['infos'][current_id]['sweeps'][i])
        ####suppose that previous key frame has aligned with current frame. The process of previous key frame is similar to the sweep frame before.
    key_infos['infos'][current_id]['sweeps'] = sweep_lists # 记录当前帧的全部之前帧
mmcv.dump(key_infos, info_path)
    






