from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import torch
from . import box_np_ops, box_torch_ops
import pdb
import copy
class BoxCoder(object):
    """Abstract base class for box coder."""

    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass

class GroundBox3dCoderAF(BoxCoder):
    def __init__(self, velocity=True, center='direct', height='direct', dim='log', rotation='direct', pc_range=[-50, -51.2, -5, 50, 51,2, 3],kwargs=None):
        super().__init__()
        self.velocity = velocity #  True
        self.center = center # soft_argmin
        self.height = height # bottom_soft_argmin
        self.dim = dim 
        self.rotation = rotation # 'vector'
        self.pc_range = np.array(pc_range) # [-51.2, -51.2, -5, 51.2, 51.2, 3]
        self.dims = self.pc_range[3:] - self.pc_range[:3] # [102.4, 102.4, 8]
        self.n_dim = 7
        # x_range=[-4.5, 4.5], y_range=[-4.5, 4.5], z_range=[-5, 3], 
        # xy_bin_num=16, z_bin_num=12, r_bin_num=12, dim_bin_num=12, dim_range=[-3, 3]
        self.kwargs = kwargs
        if velocity: self.n_dim += 2
        if rotation == 'vector': self.n_dim += 1
        self.grids_sensor = None # (25600, 2)
        self.ww_l = None # (25600,)
        self.hh_l = None # (25600,)

    @property
    def code_size(self):
        return self.n_dim # 20

    def layout(self, w, h):
        if self.grids_sensor is None:
            mode = self.kwargs.get('mode', None)

            ww, hh = np.meshgrid(range(w), range(h)) # (160, 160)
            ww = ww.reshape(-1) # (25600,)
            hh = hh.reshape(-1) # (25600,)
            self.ww_l = torch.LongTensor(ww).to(torch.cuda.current_device())
            self.hh_l = torch.LongTensor(hh).to(torch.cuda.current_device())

            ww = torch.FloatTensor(ww).to(torch.cuda.current_device()) + 0.5
            hh = torch.FloatTensor(hh).to(torch.cuda.current_device()) + 0.5
            ww = ww / w * self.dims[0] + self.pc_range[0]
            hh = hh / h * self.dims[1] + self.pc_range[1]

            self.grids_sensor = torch.stack([ww, hh], 1).clone().detach()

    def _encode(self, centers, shifts, gt_bbox):
        shifts[:, 0:2] = gt_bbox[0:2] - centers # (19, 2)
        if 'bottom' in self.height:
            shifts[:, 2] = gt_bbox[2] - 0.5 * gt_bbox[5] # (19, 1) 计算gt与center的差
        else:
            shifts[:, 2] = gt_bbox[2]
        if self.rotation == 'vector':
            shifts[:, -2] = torch.cos(gt_bbox[-1]) # (19, 1)
            shifts[:, -1] = torch.sin(gt_bbox[-1]) # (19, 1)
        else:
            shifts[:, -1] = gt_bbox[-1]

        if 'log' in self.dim:
            shifts[:, 3:6] = torch.log(gt_bbox[3:6]) # (19, 3) 计算gt的log值
        else:
            shifts[:, 3:6] = gt_bbox[3:6] 
        if self.velocity:
            shifts[:, 6:8] = gt_bbox[6:8] # (19, 2)
        return shifts

    def _decode(self, shifts, w, h):
        self.layout(w, h) # 生成坐标
        shifts[:, :, 0] += self.grids_sensor[:, 0] # (4, 25600) 在真实网格坐标基础上加预测值
        shifts[:, :, 1] += self.grids_sensor[:, 1] # (4, 25600)
        if self.rotation == 'vector':
            shifts[:, :, -2] = torch.atan2(shifts[:, :, -1], shifts[:, :, -2]) # 恢复yaw角 (4, 25600)
            shifts = shifts[:, :, :-1]
        if 'log' in self.dim:
            shifts[:, :, 3:6] = torch.exp(shifts[:, :, 3:6]) # (4, 25600, 3) 编码的时候取log，这里取exp
        if 'bottom' in self.height:
            shifts[:, :, 2] += 0.5 * shifts[:, :, 5] # z + h/2 恢复中心 (4, 25600)
        return shifts


