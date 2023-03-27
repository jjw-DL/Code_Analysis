from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from pub_tracker import PubTracker as Tracker
from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--max_age", type=int, default=3)

    args = parser.parse_args()

    return args


def save_first_frame():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=True) # 构造nuscenes对象
    if args.version == 'v1.0-trainval':
        scenes = splits.val # 获取验证集的scenes id(150)
    elif args.version == 'v1.0-test':
        scenes = splits.test 
    else:
        raise ValueError("unknown")

    # 逐帧处理
    frames = []
    for sample in nusc.sample: # (34149,)
        scene_name = nusc.get("scene", sample['scene_token'])['name'] # 获取sample对应scene的名称
        if scene_name not in scenes: # 如果不是验证集的scene则跳过
            continue 

        timestamp = sample["timestamp"] * 1e-6 # 获取时间戳
        token = sample["token"] # 获取该sample的token
        frame = {}
        frame['token'] = token # 记录该帧的token
        frame['timestamp'] = timestamp  # 记录该帧的timestamp

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True # 如果没有之前帧，则作为起始帧
        else:
            frame['first'] = False 
        frames.append(frame) # 将该帧加入list

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)


def main():
    args = parse_args()
    print('Deploy OK')

    tracker = Tracker(max_age=args.max_age, hungarian=args.hungarian) # 初始化tracker

    with open(args.checkpoint, 'rb') as f:
        predictions=json.load(f)['results'] # 加载预测信息

    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames'] # 加载帧信息

    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(frames) # 6019

    print("Begin Tracking\n")
    start = time.time()
    for i in range(size):
        token = frames[i]['token'] # 获取该帧的token

        # reset tracking after one video sequence
        if frames[i]['first']: # 在一个序列跟踪完毕后，重置tracker
            # use this for sanity check to ensure your token order is correct
            # print("reset ", i)
            tracker.reset()
            last_time_stamp = frames[i]['timestamp'] # 更新上一帧的时间戳

        time_lag = (frames[i]['timestamp'] - last_time_stamp) # 计算和前一帧的时间差，用于和速度计算位移
        last_time_stamp = frames[i]['timestamp'] # 更新上一帧的时间戳

        preds = predictions[token] # 获取该帧的所有标注

        outputs = tracker.step_centertrack(preds, time_lag) # 计算该帧的跟踪结果
        annos = []

        for item in outputs: # eg:209
            if item['active'] == 0:
                continue 
            nusc_anno = {
                "sample_token": token, # 记录该跟踪框的sample token
                "translation": item['translation'], # 中心位移+大小+旋转+速度
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']), # 记录跟踪的id
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos}) # 根据token记录annos

    
    end = time.time() # 结束时间

    second = (end-start) # 计算跟踪时间 

    speed = size / second # 计算跟踪速度
    print("The speed is {} FPS".format(speed))

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    } # 更新标注meta信息

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir) # 创建输出文件夹

    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)
    return speed

def eval_tracking():
    args = parse_args()
    eval(os.path.join(args.work_dir, 'tracking_result.json'),
        "val",
        args.work_dir,
        args.root
    )

def eval(res_path, eval_set="val", output_dir=None, root_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval 
    from nuscenes.eval.common.config import config_factory as track_configs

    
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()


def test_time():
    speeds = []
    for i in range(3):
        speeds.append(main())

    print("Speed is {} FPS".format( max(speeds)  ))

if __name__ == '__main__':
    save_first_frame()
    main()
    # test_time()
    eval_tracking()
