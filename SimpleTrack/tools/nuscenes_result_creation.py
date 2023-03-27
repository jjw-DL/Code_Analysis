import os, argparse, json, numpy as np
from pyquaternion import Quaternion
from mot_3d.data_protos import BBox, Validity
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--obj_types', type=str, default='car,bus,trailer,truck,pedestrian,bicycle,motorcycle')
parser.add_argument('--result_folder', type=str, default='../nu_mot_results/')
parser.add_argument('--data_folder', type=str, default='../datasets/nuscenes/')
args = parser.parse_args()


def bbox_array2nuscenes_format(bbox_array):
    translation = bbox_array[:3].tolist() # 获取中心点
    size = bbox_array[4:7].tolist() # 获取bbox的大小
    size = [size[1], size[0], size[2]]
    velocity = [0.0, 0.0] # 将速度设置为0
    score = bbox_array[-1] # 获取Bbox分数

    yaw = bbox_array[3] # 获取bbox的yaw角
    rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                           [np.sin(yaw),  np.cos(yaw), 0, 0],
                           [0,            0,           1, 0],
                           [0,            1,           0, 1]]) # 组装旋转矩阵
    q = Quaternion(matrix=rot_matrix) # 将旋转矩阵转化为四元数
    rotation = q.q.tolist()

    sample_result = {
        'translation':    translation, # (3,)
        'size':           size, # (3,)
        'velocity':       velocity, # (2,)
        'rotation':       rotation, # (4,)
        'tracking_score': score # eg:0.865
    }
    return sample_result


def main(name, obj_types, data_folder, result_folder, output_folder):
    """
    name:SimpleTrack2Hz
    obj_types:[car,bus,trailer,truck,pedestrian,bicycle,motorcycle]
    data_folder:datasets/nuscenes/2hz
    result_folder:nuscenes_results/SimpleTrack2Hz
    output_folder:nuscenes_results/SimpleTrack2Hz/results
    """
    # 逐个类别处理
    for obj_type in obj_types:
        print('CONVERTING {:}'.format(obj_type))
        summary_folder = os.path.join(result_folder, 'summary', obj_type) # nuscenes_results/SimpleTrack2Hz/summary/car
        file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info'))) # 获取scene名称排序
        token_info_folder = os.path.join(data_folder, 'token_info') # datasets/nuscenes/2hz/token_info
    
        results = dict()
        pbar = tqdm(total=len(file_names)) # 创建进度条
        # 逐个scene处理
        for file_index, file_name in enumerate(file_names):
            segment_name = file_name.split('.')[0] # 获取scene名称 eg：scene-0003
            # 加载该scene的token info --> datasets/nuscenes/2hz/token_info/scene-0003.json
            token_info = json.load(open(os.path.join(token_info_folder, '{:}.json'.format(segment_name)), 'r'))
            # 加载该scene的跟踪结果 --> nuscenes_results/SimpleTrack2Hz/summary/car/scene-0003.npz
            mot_results = np.load(os.path.join(summary_folder, '{:}.npz'.format(segment_name)), allow_pickle=True)
            # 提取跟踪结果
            ids, bboxes, states, types = \
                mot_results['ids'], mot_results['bboxes'], mot_results['states'], mot_results['types']
            frame_num = len(ids) # 获取帧数量
            # 逐帧处理
            for frame_index in range(frame_num):
                sample_token = token_info[frame_index] # 获取该帧的token
                results[sample_token] = list() # 初始化results字典
                # 获取该帧的跟踪结果
                frame_bboxes, frame_ids, frame_types, frame_states = \
                    bboxes[frame_index], ids[frame_index], types[frame_index], states[frame_index]
                # 逐个bbox处理
                frame_obj_num = len(frame_ids)
                for i in range(frame_obj_num):
                    # 将bbox转换为nuscenes格式
                    sample_result = bbox_array2nuscenes_format(frame_bboxes[i])
                    sample_result['sample_token'] = sample_token # 记录sample token
                    sample_result['tracking_id'] = frame_types[i] + '_' + str(frame_ids[i]) # eg：car_0_0
                    sample_result['tracking_name'] = frame_types[i] # car
                    results[sample_token].append(sample_result) # 将跟踪结果加入results
            pbar.update(1)
        pbar.close()
        submission_file = {
            'meta': {
                'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False
            },
            'results': results
        } # 组织提交文件
    
        f = open(os.path.join(output_folder, obj_type, 'results.json'), 'w') # 保存结果
        json.dump(submission_file, f)
        f.close()
    return 


if __name__ == '__main__':
    result_folder = os.path.join(args.result_folder, args.name) # 'nuscenes_results/SimpleTrack2Hz'
    obj_types = args.obj_types.split(',') # [car,bus,trailer,truck,pedestrian,bicycle,motorcycle]
    output_folder = os.path.join(result_folder, 'results') # nuscenes_results/SimpleTrack2Hz/results
    for obj_type in obj_types:
        tmp_output_folder = os.path.join(result_folder, 'results', obj_type)
        os.makedirs(tmp_output_folder, exist_ok=True) # 创建结果文件夹
    
    main(args.name, obj_types, args.data_folder, result_folder, output_folder)
