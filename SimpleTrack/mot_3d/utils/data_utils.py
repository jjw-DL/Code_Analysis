# Selecting the sequences according to types
# Transfer the ID from string into int if needed
from ..data_protos import BBox
import numpy as np


__all__ = ['inst_filter', 'str2int', 'box_wrapper', 'type_filter', 'id_transform']


def str2int(strs):
    result = [int(s) for s in strs]
    return result


def box_wrapper(bboxes, ids):
    frame_num = len(ids)
    result = list()
    for _i in range(frame_num):
        frame_result = list()
        num = len(ids[_i])
        for _j in range(num):
            frame_result.append((ids[_i][_j], bboxes[_i][_j]))
        result.append(frame_result)
    return result


def id_transform(ids):
    frame_num = len(ids) # 帧数

    id_list = list()
    for _i in range(frame_num):
        id_list += ids[_i] # 拼接该scene内所有帧的id
    id_list = sorted(list(set(id_list))) # 去重后排序 eg:(1779)
    
    id_mapping = dict()
    for _i, id in enumerate(id_list):
        id_mapping[id] = _i # id重新映射
    
    result = list()
    # 逐帧重新处理
    for _i in range(frame_num):
        frame_ids = list()
        frame_id_num = len(ids[_i]) # eg：(37,) 该帧的id数
        # 逐个id处理
        for _j in range(frame_id_num):
            frame_ids.append(id_mapping[ids[_i][_j]]) # 将id重映射
        result.append(frame_ids) # 将重映射后的id加入result
    return result    


def inst_filter(ids, bboxes, types, type_field=[1], id_trans=False):
    """ filter the bboxes according to types 根据类型过滤 bbox
    """
    frame_num = len(ids) # 帧数
    if id_trans:
        ids = id_transform(ids) # 转换id
    id_result, bbox_result = [], []
    # 逐帧处理
    for _i in range(frame_num):
        frame_ids = list()
        frame_bboxes = list()
        frame_id_num = len(ids[_i]) # eg：(37,) 该帧的id数
        # 逐个id(bbox)处理
        for _j in range(frame_id_num):
            obj_type = types[_i][_j] # gt的类别 vehicle.car
            matched = False
            for type_name in type_field:
                if str(type_name) in str(obj_type):
                    matched = True # 过滤类别
            if matched:
                frame_ids.append(ids[_i][_j]) # 将bbox的id加入frame_ids
                frame_bboxes.append(BBox.array2bbox(bboxes[_i][_j])) # 将bbox加入frame_bboxes
        id_result.append(frame_ids)
        bbox_result.append(frame_bboxes)
    return id_result, bbox_result


def type_filter(contents, types, type_field=[1]):
    frame_num = len(types)
    content_result = [list() for i in range(len(type_field))]
    for _k, inst_type in enumerate(type_field):
        for _i in range(frame_num):
            frame_contents = list()
            frame_id_num = len(contents[_i])
            for _j in range(frame_id_num):
                if types[_i][_j] != inst_type:
                    continue
                frame_contents.append(contents[_i][_j])
            content_result[_k].append(frame_contents)
    return content_result