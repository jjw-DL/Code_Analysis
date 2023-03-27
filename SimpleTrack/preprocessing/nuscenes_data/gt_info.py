import os, numpy as np, nuscenes, argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../raw/nuscenes/data/sets/nuscenes/')
parser.add_argument('--data_folder', type=str, default='../../../datasets/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
args = parser.parse_args()


def instance_info2bbox_array(info):
    translation = info.center.tolist() # 中心点坐标 (3,)
    size = info.wlh.tolist() # 长宽高 (3,)
    rotation = info.orientation.q.tolist() # 旋转角 四元数 (4,)
    return translation + size + rotation # (10,)


def main(nusc, scene_names, root_path, gt_folder):
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
        # 初始化bbox的token，类别和bbox几何的list
        IDS, inst_types, bboxes = list(), list(), list()
        while True:
            frame_ids, frame_types, frame_bboxes = list(), list(), list()
            if args.mode == '2hz':
                frame_data = nusc.get('sample', cur_sample_token) # 获取该sample的record
                lidar_token = frame_data['data']['LIDAR_TOP'] # 获取该sample的lidar token
                instances = nusc.get_boxes(lidar_token) # 根据lidar token获取对应的bbox
                # 逐个bbox处理
                for inst in instances:
                    frame_ids.append(inst.token) # 获取该bbox的token
                    frame_types.append(inst.name) # 获取该bbox的类别
                    frame_bboxes.append(instance_info2bbox_array(inst)) # 将bbox转换为10维

            elif args.mode == '20hz':
                frame_data = nusc.get('sample_data', cur_sample_token)
                lidar_data = nusc.get('sample_data', cur_sample_token) # 这里已经是lidar data
                instances = nusc.get_boxes(lidar_data['token']) # 根据lidar token获取对应的bbox
                # 逐个bbox处理
                for inst in instances:
                    frame_ids.append(inst.token) # 获取该bbox的token
                    frame_types.append(inst.name) # 获取该bbox的类别
                    frame_bboxes.append(instance_info2bbox_array(inst)) # 将bbox转换为10维
            
            IDS.append(frame_ids) # 添加该帧的ann token
            inst_types.append(frame_types) # 添加该帧的ann 类别
            bboxes.append(frame_bboxes) # 添加该帧的全部bbox

            # clean up and prepare for the next
            cur_sample_token = frame_data['next'] # 更新下一帧
            if cur_sample_token == '':
                break

        # 保存一个scene的全部标注信息
        np.savez_compressed(os.path.join(gt_folder, '{:}.npz'.format(scene_name)), 
            ids=IDS, types=inst_types, bboxes=bboxes)
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    print('gt info')
    gt_folder = os.path.join(args.data_folder, 'gt_info')
    os.makedirs(gt_folder, exist_ok=True)

    val_scene_names = splits.create_splits_scenes()['val']
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, val_scene_names, args.raw_data_folder, gt_folder)
