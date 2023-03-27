import os, argparse, numpy as np, json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='raw/nuscenes')
parser.add_argument('--data_folder', type=str, default='datasets/nuscenes/2hz')
parser.add_argument('--det_name', type=str, default='cp') # centerpoint
parser.add_argument('--file_path', type=str, default='val.json')
parser.add_argument('--velo', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
args = parser.parse_args()


def get_sample_tokens(data_folder, mode):
    token_folder = os.path.join(data_folder, 'token_info') # 拼接token所在文件夹 datasets/nuscenes/2hz/token_info
    file_names = sorted(os.listdir(token_folder)) # 对scene名称排序
    result = dict()
    # 逐个场景处理
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(token_folder, file_name)
        scene_name = file_name.split('.')[0] # 获取场景名称 ge：scene-0003
        tokens = json.load(open(file_path, 'r')) # 打开token文件

        if mode == '2hz':
            result[scene_name] = tokens # dict-->记录scene的token (40,)
        elif mode == '20hz':
            result[scene_name] = [t[0] for t in tokens] # 记录该scene的token
    return result


def sample_result2bbox_array(sample):
    trans, size, rot, score = \
        sample['translation'], sample['size'],sample['rotation'], sample['detection_score']
    return trans + size + rot + [score] # 取出对应的结果并拼接


def main(det_name, file_path, detection_folder, data_folder, mode):
    # 1.dealing with the paths
    detection_folder = os.path.join(detection_folder, det_name) # datasets/nuscenes/2hz/detection/cp
    output_folder = os.path.join(detection_folder, 'dets') # datasets/nuscenes/2hz/detection/cp/dets
    os.makedirs(output_folder, exist_ok=True) # 构造输出文件夹
    
    # 2.load the detection file
    print('LOADING RAW FILE')
    f = open(file_path, 'r')
    det_data = json.load(f)['results'] # 加载检测结果
    f.close()

    # 3.prepare the scene names and all the related tokens
    tokens = get_sample_tokens(data_folder, mode) # datasets/nuscenes/2hz 2hz --> 150个scene
    scene_names = sorted(list(tokens.keys())) # 对scene名称对排序
    bboxes, inst_types, velos = dict(), dict(), dict() # 初始化--> scene_name为first key，frame id为second key
    # 逐个scene处理, 初始化bboxes，类别和速度为[]
    for scene_name in scene_names:
        frame_num = len(tokens[scene_name]) # eg:40
        bboxes[scene_name], inst_types[scene_name] = \
            [[] for i in range(frame_num)], [[] for i in range(frame_num)]
        if args.velo:
            velos[scene_name] = [[] for i in range(frame_num)]

    # 4.enumerate through all the frames
    sample_keys = list(det_data.keys()) # 每一帧的token
    print('PROCESSING...')
    pbar = tqdm(total=len(sample_keys)) # 创建进度条
    # 逐帧处理
    for sample_index, sample_key in enumerate(sample_keys):
        # find the corresponding scene and frame index
        scene_name, frame_index = None, None # 初始化scene name和frame index
        # 逐个scene处理
        for scene_name in scene_names:
            # 如果当前帧在该scene的token中
            if sample_key in tokens[scene_name]:
                frame_index = tokens[scene_name].index(sample_key) # 获取该帧在该scene中的帧索引
                break
        
        # extract the bboxes and types
        sample_results = det_data[sample_key] # 取出该帧的全部检测结果 eg：(200,)
        # 逐个bbox处理
        for sample in sample_results:
            # 取出对应bbox并转换成array和类别
            bbox, inst_type = sample_result2bbox_array(sample), sample['detection_name']
            inst_velo = sample['velocity'] # 取出速度预测
            bboxes[scene_name][frame_index] += [bbox] # bbox拼接
            inst_types[scene_name][frame_index] += [inst_type] # 类别拼接

            if args.velo:
                velos[scene_name][frame_index] += [inst_velo] # 速度拼接
        pbar.update(1)
    pbar.close()

    # save the results
    print('SAVING...')
    pbar = tqdm(total=len(scene_names)) # 创建进度条
    # 逐个scene处理
    for scene_name in scene_names:
        if args.velo:
            # 逐个scene保存对应的检测结果，包括bboxes，类别和速度
            np.savez_compressed(os.path.join(output_folder, '{:}.npz'.format(scene_name)),
                bboxes=bboxes[scene_name], types=inst_types[scene_name], velos=velos[scene_name]) 
        else:
            np.savez_compressed(os.path.join(output_folder, '{:}.npz'.format(scene_name)),
                bboxes=bboxes[scene_name], types=inst_types[scene_name])
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    detection_folder = os.path.join(args.data_folder, 'detection')
    os.makedirs(detection_folder, exist_ok=True)

    main(args.det_name, args.file_path, detection_folder, args.data_folder, args.mode)