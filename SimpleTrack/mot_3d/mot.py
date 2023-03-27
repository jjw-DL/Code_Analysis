from copy import deepcopy
import numpy as np, mot_3d.tracklet as tracklet, mot_3d.utils as utils
from .redundancy import RedundancyModule
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .data_protos import BBox, Validity
from .association import associate_dets_to_tracks
from . import visualization
from mot_3d import redundancy
import pdb, os


class MOTModel:
    def __init__(self, configs):
        self.trackers = list()         # tracker for each single tracklet 每个tracklet的跟踪器
        self.frame_count = 0           # record for the frames 帧记录
        self.count = 0                 # record the obj number to assign ids 记录obj编号以分配id
        self.time_stamp = None         # the previous time stamp 上一个时间戳
        self.redundancy = RedundancyModule(configs) # module for no detection cases 无检测case模块

        non_key_redundancy_config = deepcopy(configs)
        non_key_redundancy_config['redundancy'] = {
            'mode': 'mm',
            'det_score_threshold': {'giou': 0.1, 'iou': 0.1, 'euler': 0.1},
            'det_dist_threshold': {'giou': -0.5, 'iou': 0.1, 'euler': 4}
        } # 更新redundancy字段
        self.non_key_redundancy = RedundancyModule(non_key_redundancy_config)

        self.configs = configs
        self.match_type = configs['running']['match_type'] # bipartite
        self.score_threshold = configs['running']['score_threshold'] # 0.01
        self.asso = configs['running']['asso'] # giou
        self.asso_thres = configs['running']['asso_thres'][self.asso] # 1.5
        self.motion_model = configs['running']['motion_model'] # kf

        self.max_age = configs['running']['max_age_since_update'] # 2
        self.min_hits = configs['running']['min_hits_to_birth'] # 0

    @property
    def has_velo(self):
        return not (self.motion_model == 'kf' or self.motion_model == 'fbkf' or self.motion_model == 'ma')
    
    def frame_mot(self, input_data: FrameData):
        """ For each frame input, generate the latest mot results
        Args:
            input_data (FrameData): input data, including detection bboxes and ego information
        Returns:
            tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
        """
        self.frame_count += 1 # 帧计数器

        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp # 帧时间戳

        if not input_data.aux_info['is_key_frame']:
            result = self.non_key_frame_mot(input_data)
            return result
    
        if 'kf' in self.motion_model:
            # -------------------------
            # 执行跟踪的关键步骤，计算匹配的检测和跟踪框，未匹配的检测框和未匹配的跟踪框
            # -------------------------
            matched, unmatched_dets, unmatched_trks = self.forward_step_trk(input_data)
        
        time_lag = input_data.time_stamp - self.time_stamp # 计算时间延迟 0.5s左右
        # update the matched tracks 更新匹配的跟踪框
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks: # 针对匹配上的跟踪框，更新
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0] # 找到匹配的检测框
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]), 
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                    frame_index=self.frame_count, pc=input_data.pc, 
                    dets=input_data.dets, aux_info=aux_info) # 创建UpdateInfoData类
                trk.update(update_info) # 更新轨迹信息
            else:
                # 针对未匹配的跟踪框，更新
                # 返回假定的状态、关联字符串和辅助信息 result_bbox, 0, None
                result_bbox, update_mode, aux_info = self.redundancy.infer(trk, input_data, time_lag)
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']} # 获取是否是关键帧
                update_info = UpdateInfoData(mode=update_mode, bbox=result_bbox, 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info) # 由于没有匹配，因此不更新运动状态，但更新生命周期
        
        # create new tracks for unmatched detections 对未匹配的检测框创建新的tracklet
        for index in unmatched_dets:
            if self.has_velo:
                aux_info = {
                    'velo': list(input_data.aux_info['velos'][index]), 
                    'is_key_frame': input_data.aux_info['is_key_frame']}
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}

            track = tracklet.Tracklet(self.configs, self.count, input_data.dets[index], input_data.det_types[index], 
                self.frame_count, aux_info=aux_info, time_stamp=input_data.time_stamp) # 初始化tracklet
            self.trackers.append(track) # 将tracklet加入tracker
            self.count += 1 # 计数器加1
        
        # remove dead tracks
        track_num = len(self.trackers)
        for index, trk in enumerate(reversed(self.trackers)):
            if trk.death(self.frame_count):
                self.trackers.pop(track_num - 1 - index) # 如果该跟踪已经是一个dead的状态，则将其pop出来
        
        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count) # eg：alive_1_0
            # 当前的box，id，状态和类别
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))
        
        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp) # 同步时间戳

        return result
    
    def forward_step_trk(self, input_data: FrameData):
        dets = input_data.dets # (39,)
        det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold] # 根据分数阈值过滤
        dets = [dets[i] for i in det_indexes] # 重新提取bbox

        # prediction and association 预测和关联
        trk_preds = list()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame'])) # 预测
        
        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

        # 计算匹配的检测和跟踪框，未匹配的检测框和未匹配的跟踪框
        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix) # 关联
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]] # 将检测结果重新映射
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks
    
    def non_key_forward_step_trk(self, input_data: FrameData):
        """ tracking on non-key frames (for nuScenes)
        """
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= 0.5]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        
        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks
    
    def non_key_frame_mot(self, input_data: FrameData):
        """ tracking on non-key frames (for nuScenes)
        """
        self.frame_count += 1
        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp
        
        if 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.non_key_forward_step_trk(input_data)
        time_lag = input_data.time_stamp - self.time_stamp

        redundancy_bboxes, update_modes = self.non_key_redundancy.bipartite_infer(input_data, self.trackers)
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]), 
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                    frame_index=self.frame_count, pc=input_data.pc, 
                    dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_modes[t], bbox=redundancy_bboxes[t], 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
        
        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))

        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        return result