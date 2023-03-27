""" inference on the nuscenes dataset
"""
import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from data_loader import NuScenesLoader
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box


parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--det_name', type=str, default='cp') # centerpoint
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--start_frame', type=int, default=0, help='start at a middle frame for debug')
parser.add_argument('--obj_types', default='car,bus,trailer,truck,pedestrian,bicycle,motorcycle')
# paths
parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='config file path, follow the path in the documentation')
parser.add_argument('--result_folder', type=str, default='nu_mot_results/')
parser.add_argument('--data_folder', type=str, default='datasets/nuscenes/')
args = parser.parse_args()


def nu_array2mot_bbox(b):
    nu_box = Box(b[:3], b[3:6], Quaternion(b[6:10])) # Nuscene的bbox类
    mot_bbox = BBox(
        x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2],
        w=nu_box.wlh[0], l=nu_box.wlh[1], h=nu_box.wlh[2],
        o=nu_box.orientation.yaw_pitch_roll[0]
    ) # 自定义的跟踪bbox
    if len(b) == 11:
        mot_bbox.s = b[-1]
    return mot_bbox


def load_gt_bboxes(data_folder, type_token, segment_name):
    # datasets/nuscenes/2hz/gt_info/scene-0003.npz
    gt_info = np.load(os.path.join(data_folder, 'gt_info', '{:}.npz'.format(segment_name)), allow_pickle=True)
    # 获取该scene中bbox的token，类别和bboxes
    ids, inst_types, bboxes = gt_info['ids'], gt_info['types'], gt_info['bboxes']
    
    mot_bboxes = list()
    # 逐帧处理
    for _, frame_bboxes in enumerate(bboxes):
        mot_bboxes.append([])
        # 逐个bbox处理
        for _, b in enumerate(frame_bboxes):
            mot_bboxes[-1].append(BBox.bbox2array(nu_array2mot_bbox(b)))
    # 根据类别过滤并且返回重新映射id
    gt_ids, gt_bboxes = utils.inst_filter(ids, mot_bboxes, inst_types, 
        type_field=type_token, id_trans=True)
    return gt_bboxes, gt_ids


def frame_visualization(bboxes, ids, states, gt_bboxes=None, gt_ids=None, pc=None, dets=None, name=''):
    # 创建visualizer对象
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))
    # 绘制点云
    if pc is not None:
        visualizer.handler_pc(pc)
    # 绘制GT bbox
    if gt_bboxes is not None:
        for _, bbox in enumerate(gt_bboxes):
            visualizer.handler_box(bbox, message='', color='black')
    # 绘制检测框
    dets = [d for d in dets if d.s >= 0.01]
    for det in dets:
        visualizer.handler_box(det, message='%.2f' % det.s, color='gray', linestyle='dashed')
    # 绘制跟踪框
    for _, (bbox, id, state_string) in enumerate(zip(bboxes, ids, states)):
        if Validity.valid(state_string):
            visualizer.handler_box(bbox, message='%.2f %s'%(bbox.s, id), color='red')
        else:
            visualizer.handler_box(bbox, message='%.2f %s'%(bbox.s, id), color='light_blue')
    visualizer.show()
    visualizer.close()


def sequence_mot(configs, data_loader, obj_type, sequence_id, gt_bboxes=None, gt_ids=None, visualize=False):
    tracker = MOTModel(configs) # 构造MOTModel对象
    frame_num = len(data_loader) # 计算帧数
    IDs, bboxes, states, types = list(), list(), list(), list() # 初始化跟踪结果

    # 逐帧处理
    for frame_index in range(data_loader.cur_frame, frame_num): # 0, 40
        if frame_index % 10 == 0:
            # 每10帧打印一次 TYPE car SEQ 0 FRAME 1 / 40
            print('TYPE {:} SEQ {:} Frame {:} / {:}'.format(obj_type, sequence_id, frame_index + 1, frame_num))
        
        # input data
        frame_data = next(data_loader) # 40
        # 构造FrameData对象
        frame_data = FrameData(dets=frame_data['dets'], ego=frame_data['ego'], pc=frame_data['pc'], 
            det_types=frame_data['det_types'], aux_info=frame_data['aux_info'], time_stamp=frame_data['time_stamp'])

        # ------------------
        # mot 执行跟踪
        # ------------------
        results = tracker.frame_mot(frame_data)
        # 跟踪结果按照类型整理
        result_pred_bboxes = [trk[0] for trk in results]
        result_pred_ids = [trk[1] for trk in results]
        result_pred_states = [trk[2] for trk in results]
        result_types = [trk[3] for trk in results]

        # visualization 可视化
        if visualize:
            frame_visualization(result_pred_bboxes, result_pred_ids, result_pred_states,
                gt_bboxes[frame_index], gt_ids[frame_index], frame_data.pc, dets=frame_data.dets, name='{:}_{:}'.format(args.name, frame_index))
                    
        # wrap for output 整理输出
        IDs.append(result_pred_ids)
        result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
        bboxes.append(result_pred_bboxes)
        states.append(result_pred_states)
        types.append(result_types)

    return IDs, bboxes, states, types


def main(name, obj_types, config_path, data_folder, det_data_folder, result_folder, start_frame=0, token=0, process=1):
    """
    name:SimpleTrack2Hz
    obj_types:检测类别 [car,bus,trailer,truck,pedestrian,bicycle,motorcycle]
    config_path:配置文件路径
    data_folder:'datasets/nuscenes/2hz'
    det_data_folder: 'datasets/nuscenes/2hz/detection/cp'
    result_folder: 'nuscenes_results/SimpleTrack2Hz'
    start_frame:0
    token:0
    process:1
    """
    # 逐类处理
    for obj_type in obj_types:
        summary_folder = os.path.join(result_folder, 'summary', obj_type) # 拼接结果输出文件夹 nuscenes_results/SimpleTrack2Hz/summary/car
        # simply knowing about all the segments
        file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info'))) # datasets/nuscenes/2hz/ego_info 对scenes name进行排序
        
        # load model configs
        configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader) # 加载config文件

        # 逐个scene处理
        for file_index, file_name in enumerate(file_names[:]):
            if file_index % process != token: # 分线程处理
                continue
            # eg：START TYPE car SEQ 1 / 150
            print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, file_index + 1, len(file_names)))
            segment_name = file_name.split('.')[0] # eg: scene-0003
            
            # 构造NuScenesLoader对象
            data_loader = NuScenesLoader(configs, [obj_type], segment_name, data_folder, det_data_folder, start_frame)

            # 加载gt和对应的id
            gt_bboxes, gt_ids = load_gt_bboxes(data_folder, [obj_type], segment_name)
            # ---------------------------
            # 对该scene的bbox进行跟踪
            # ---------------------------
            ids, bboxes, states, types = sequence_mot(configs, data_loader, obj_type, file_index, gt_bboxes, gt_ids, args.visualize)
    
            frame_num = len(ids)
            # 逐帧处理
            for frame_index in range(frame_num):
                id_num = len(ids[frame_index]) # 获取该帧的id数量
                for i in range(id_num):
                    ids[frame_index][i] = '{:}_{:}'.format(file_index, ids[frame_index][i])

            # 保存一个scene的跟踪信息，按照帧组织，帧内按照bbox组织
            np.savez_compressed(os.path.join(summary_folder, '{}.npz'.format(segment_name)),
                ids=ids, bboxes=bboxes, states=states, types=types)


if __name__ == '__main__':
    result_folder = os.path.join(args.result_folder, args.name) # 'nuscenes_results/SimpleTrack2Hz'
    os.makedirs(result_folder, exist_ok=True)
    summary_folder = os.path.join(result_folder, 'summary') # 'nuscenes_results/SimpleTrack2Hz/summary'
    os.makedirs(summary_folder, exist_ok=True)
    det_data_folder = os.path.join(args.data_folder, 'detection', args.det_name) # 'datasets/nuscenes/2hz/detection/cp'

    obj_types = args.obj_types.split(',') # [car,bus,trailer,truck,pedestrian,bicycle,motorcycle]
    # 逐类处理
    for obj_type in obj_types:
        tmp_summary_folder = os.path.join(summary_folder, obj_type)
        os.makedirs(tmp_summary_folder, exist_ok=True) # 每个类别单独构建文件夹

    if args.process > 1: # 
        pool = multiprocessing.Pool(args.process) # 多线程处理
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.name, obj_types, args.config_path, args.data_folder, det_data_folder, 
                result_folder, 0, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.name, obj_types, args.config_path, args.data_folder, det_data_folder, 
            result_folder, args.start_frame, 0, 1)