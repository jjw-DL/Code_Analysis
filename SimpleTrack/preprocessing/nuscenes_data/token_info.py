import os, numpy as np, nuscenes, argparse, json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../raw/nuscenes/data/sets/nuscenes/')
parser.add_argument('--data_folder', type=str, default='../../../datasets/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
args = parser.parse_args()


def set_selected_or_not(frame_tokens):
    """ under the 20hz setting, 
        we have to set whether to use a certain frame
        1. select at the interval of 1 frames
        2. if meet key frame, reset the counter
    """
    counter = -1
    selected = list()
    frame_num = len(frame_tokens) # eg:384
    # 逐帧处理
    for _, tokens in enumerate(frame_tokens):
        is_key_frame = tokens[1] # 判断是否是关键帧
        counter += 1 # 计数器+1
        if is_key_frame: # 如果是关键帧,将select标记设置为True
            selected.append(True) # 关键帧必选
            counter = 0 # 将计数器归零
            continue
        else:
            if counter % 2 == 0: # 如果不是关键帧则间隔2帧取一帧
                selected.append(True)
            else:
                selected.append(False)
    # 在token标记中加入是否被选择的标记
    result_tokens = [(list(frame_tokens[i]) + [selected[i]]) for i in range(frame_num)]
    return result_tokens

# ----------------------------------
# 关键帧直接通过sample的next获取，过渡帧则需要sample data的next获取
# ----------------------------------

def main(nusc, scene_names, root_path, token_folder, mode):
    pbar = tqdm(total=len(scene_names)) # 创建进度条
    # 逐个scene处理
    for scene_index, scene_info in enumerate(nusc.scene):
        scene_name = scene_info['name'] # 获取scene名称
        if scene_name not in scene_names: # 如果该scene名称不在val中，则跳过
            continue

        first_sample_token = scene_info['first_sample_token'] # 获取该scene的第一帧的token
        last_sample_token = scene_info['last_sample_token'] # 获取该scene的最后一帧的token
        frame_data = nusc.get('sample', first_sample_token) # 获取第一帧的sample record
        if mode == '20hz':
            cur_sample_token = frame_data['data']['LIDAR_TOP'] # 取该seq第一帧中sample data的lidar token
        elif mode == '2hz':
            cur_sample_token = deepcopy(first_sample_token)
        frame_tokens = list() # 初始化frame token的list

        while True:
            # find the path to lidar data
            if mode == '2hz':
                frame_data = nusc.get('sample', cur_sample_token) # 获取该帧的sample record
                frame_tokens.append(cur_sample_token) # 将该token加入frame token
            elif mode == '20hz':
                frame_data = nusc.get('sample_data', cur_sample_token) # 取该seq第一帧中sample data的record
                # 加入当前帧sample data token，是否是关键帧，以及sample token
                frame_tokens.append((cur_sample_token, frame_data['is_key_frame'], frame_data['sample_token']))

            # clean up and prepare for the next
            cur_sample_token = frame_data['next'] # 更新下一帧
            if cur_sample_token == '':
                break
        
        if mode == '20hz':
            frame_tokens = set_selected_or_not(frame_tokens)
        f = open(os.path.join(token_folder, '{:}.json'.format(scene_name)), 'w') # 拼接文件名称并写入
        json.dump(frame_tokens, f)
        f.close()

        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    print('token info')
    os.makedirs(args.data_folder, exist_ok=True) # 创建输出文件夹

    token_folder = os.path.join(args.data_folder, 'token_info')
    os.makedirs(token_folder, exist_ok=True) # 创建token_info文件夹

    val_scene_names = splits.create_splits_scenes()['val'] # 获取验证集的scene name
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True) # 创建NuScenes对象
    main(nusc, val_scene_names, args.raw_data_folder, token_folder, args.mode)
