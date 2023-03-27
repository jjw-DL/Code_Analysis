""" Split the area into grid boxes
    BBoxes in different grid boxes without overlap cannot have overlap
"""
import numpy as np
from ..data_protos import BBox


class BBoxCoarseFilter:
    def __init__(self, grid_size, scaler=100):
        self.gsize = grid_size # 100
        self.scaler = 100
        self.bbox_dict = dict()
    
    def bboxes2dict(self, bboxes):
        # 将bboxes转换为dict
        # 逐个bbox处理
        for i, bbox in enumerate(bboxes):
            grid_keys = self.compute_bbox_key(bbox) # # [209, 209, 209, 209]
            for key in grid_keys:
                if key not in self.bbox_dict.keys():
                    self.bbox_dict[key] = set([i])
                else:
                    self.bbox_dict[key].add(i)
        return
        
    def compute_bbox_key(self, bbox):
        corners = np.asarray(BBox.box2corners2d(bbox)) # (4, 3) 4个角点坐标
        min_keys = np.floor(np.min(corners, axis=0) / self.gsize).astype(np.int) # 最小key eg：[2, 9, 0]
        max_keys = np.floor(np.max(corners, axis=0) / self.gsize).astype(np.int) # 最大key eg：[2, 9, 0]
        
        # enumerate all the corners
        grid_keys = [
            self.scaler * min_keys[0] + min_keys[1],
            self.scaler * min_keys[0] + max_keys[1],
            self.scaler * max_keys[0] + min_keys[1],
            self.scaler * max_keys[0] + max_keys[1]
        ] # [209, 209, 209, 209]
        return grid_keys
    
    def related_bboxes(self, bbox):
        """ return the list of related bboxes
        """ 
        result = set()
        grid_keys = self.compute_bbox_key(bbox) 
        for key in grid_keys:
            if key in self.bbox_dict.keys():
                result.update(self.bbox_dict[key])
        return list(result) # 相关Bbox索引
    
    def clear(self):
        self.bbox_dict = dict()