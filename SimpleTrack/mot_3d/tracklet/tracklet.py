import numpy as np
from .. import motion_model
from .. import life as life_manager
from ..update_info_data import UpdateInfoData
from ..frame_data import FrameData
from ..data_protos import BBox


class Tracklet:
    def __init__(self, configs, id, bbox: BBox, det_type, frame_index, time_stamp=None, aux_info=None):
        self.id = id # 跟踪id
        self.time_stamp = time_stamp # 时间戳
        self.asso = configs['running']['asso'] # giou
        
        self.configs = configs # 匹配文件
        self.det_type = det_type # 检测类别
        self.aux_info = aux_info # 附加信息
        
        # initialize different types of motion model
        self.motion_model_type = configs['running']['motion_model'] # kf
        # simple kalman filter 简单的kalman滤波
        if self.motion_model_type == 'kf':
            self.motion_model = motion_model.KalmanFilterMotionModel(
                bbox=bbox, inst_type=self.det_type, time_stamp=time_stamp, covariance=configs['running']['covariance'])

        # life and death management 生命周期管理
        self.life_manager = life_manager.HitManager(configs, frame_index)
        # store the score for the latest bbox # 存储最新bbox的分数
        self.latest_score = bbox.s
    
    def predict(self, time_stamp=None, is_key_frame=True):
        """ in the prediction step, the motion model predicts the state of bbox
            the other components have to be synced
            the result is a BBox

            the ussage of time_stamp is optional, only if you use velocities
        """
        result = self.motion_model.get_prediction(time_stamp=time_stamp) # 对bbox进行预测
        self.life_manager.predict(is_key_frame=is_key_frame) # 对生命周期进行预测
        self.latest_score = self.latest_score * 0.01 # 最新的分数*0.01
        result.s = self.latest_score # 计算最新的分数
        return result

    def update(self, update_info: UpdateInfoData):
        """ update the state of the tracklet
        """
        self.latest_score = update_info.bbox.s # 获取最新分数
        is_key_frame = update_info.aux_info['is_key_frame'] # 标记是否是关键帧
        
        # only the direct association update the motion model
        if update_info.mode == 1 or update_info.mode == 3:
            self.motion_model.update(update_info.bbox, update_info.aux_info) # 运动状态更新
        else:
            pass
        self.life_manager.update(update_info, is_key_frame) # 生命周期更新
        return

    def get_state(self):
        """ current state of the tracklet
        """
        result = self.motion_model.get_state()
        result.s = self.latest_score
        return result
    
    def valid_output(self, frame_index):
        return self.life_manager.valid_output(frame_index)
    
    def death(self, frame_index):
        return self.life_manager.death(frame_index)
    
    def state_string(self, frame_index):
        """ the string describes how we get the bbox (e.g. by detection or motion model prediction)
        """
        return self.life_manager.state_string(frame_index)
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return self.motion_model.compute_innovation_matrix()
    
    def sync_time_stamp(self, time_stamp):
        """ sync the time stamp for motion model
        """
        self.motion_model.sync_time_stamp(time_stamp)
        return
