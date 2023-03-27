""" input form of the data in each frame
"""
from .data_protos import BBox
import numpy as np, mot_3d.utils as utils


class FrameData:
    def __init__(self, dets, ego, time_stamp=None, pc=None, det_types=None, aux_info=None):
        self.dets = dets         # detections for each frame 每帧的检测
        self.ego = ego           # ego matrix information 自车的位姿 4x4
        self.pc = pc # 点云(n, 3) global系下
        self.det_types = det_types # 检测的类别
        self.time_stamp = time_stamp # 时间戳
        self.aux_info = aux_info # 附加信息

        for i, det in enumerate(self.dets):
            self.dets[i] = BBox.array2bbox(det) # 将检测bbox转化为自定义bbox类
        
        # if not aux_info['is_key_frame']:
        #     self.dets = [d for d in self.dets if d.s >= 0.5]