import os, argparse, json, numpy as np
from pyquaternion import Quaternion
from mot_3d.data_protos import BBox, Validity

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--obj_types', default='car,bus,trailer,truck,pedestrian,bicycle,motorcycle')
parser.add_argument('--result_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/nu_mot_results/')
args = parser.parse_args()


def main(name, obj_types, result_folder):
    raw_results = list() # 初始化检测结果dict
    # 逐类处理
    for type_name in obj_types:
        # nuscenes_results/SimpleTrack2Hz/results/car/results.json
        path = os.path.join(result_folder, type_name, 'results.json')
        f = open(path, 'r') # 打开该类的跟踪结果文件
        raw_results.append(json.load(f)['results']) # 加载结果文件，并加入检测结果di
        f.close()
    
    results = raw_results[0]
    sample_tokens = list(results.keys()) # 获取全部sample token eg:(6019,)
    # 逐个sample处理
    for token in sample_tokens:
        for i in range(1, len(obj_types)): # 7类
            results[token] += raw_results[i][token] # 将同一帧中不同类别的跟踪结果组合
    
    submission_file = {
        'meta': {
            'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False
        },
        'results': results
    }

    f = open(os.path.join(result_folder, 'results.json'), 'w') # 保存最终结果
    json.dump(submission_file, f)
    f.close()
    return 


if __name__ == '__main__':
    # nuscenes_results/SimpleTrack2Hz/results
    result_folder = os.path.join(args.result_folder, args.name, 'results')
    obj_types = args.obj_types.split(',') # [car,bus,trailer,truck,pedestrian,bicycle,motorcycle]
    main(args.name, obj_types, result_folder)