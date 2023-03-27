import numpy as np
import copy
from track_utils import greedy_assignment
import copy 
import importlib
import sys 

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
  'car':4,
  'truck':4,
  'bus':5.5,
  'trailer':3,
  'pedestrian':1,
  'motorcycle':13,
  'bicycle':3,  
}



class PubTracker(object):
  def __init__(self,  hungarian=False, max_age=0):
    self.hungarian = hungarian # False
    self.max_age = max_age # 3

    print("Use hungarian: {}".format(hungarian))

    self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR # l2速度误差分布的99.9个百分点（每clss/0.5秒）

    self.reset() # 重置id和tracks
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, results, time_lag):
    if len(results) == 0:
      self.tracks = []
      return []
    else:
      temp = []
      # 逐个预测结果处理
      for det in results:
        # filter out classes not evaluated for tracking
        if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
          continue

        det['ct'] = np.array(det['translation'][:2]) # 获取物体的中心
        det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag # 根据预测速度算位移
        det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name']) # 2
        temp.append(det)

      results = temp

    N = len(results) # 111
    M = len(self.tracks) # 167

    # N X 2 
    if 'tracking' in results[0]:
      dets = np.array(
      [ det['ct'] + det['tracking'].astype(np.float32) # 物体中心位置+负位移-->预测上一帧的位置
       for det in results], np.float32)
    else:
      dets = np.array(
        [det['ct'] for det in results], np.float32) 

    item_cat = np.array([item['label_preds'] for item in results], np.int32) # N 当前框的类别
    track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M 跟踪框的类别

    # 计算每一类的最大速度误差
    max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)

    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2 提取跟踪框的物体中心

    if len(tracks) > 0:  # NOT FIRST FRAME
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M --> eg:(111, 167) 计算距离误差
      dist = np.sqrt(dist) # absolute distance in meter

      # 如果距离过大和类别不匹配的直接设置为无效
      invalid = ((dist > max_diff.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

      dist = dist  + invalid * 1e18 # 将无效位置的距离设置为无穷大
      if self.hungarian:
        dist[dist > 1e18] = 1e18
        matched_indices = linear_assignment(copy.deepcopy(dist))
      else:
        matched_indices = greedy_assignment(copy.deepcopy(dist)) # 利用贪心策略计算匹配索引 eg:(69, 2)-->(det, gt)
    else:  # first few frame 初始帧
      assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2) # 空匹配索引

    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])] # 计算未匹配检测框的索引 eg:(42,)

    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])] # 计算未匹配跟踪框的索引 eg:(98,)
    
    if self.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices # 不用hungainan

    ret = []
    # 逐个处理匹配的检测和跟踪框
    for m in matches:
      track = results[m[0]] # m[0]表示det索引
      track['tracking_id'] = self.tracks[m[1]]['tracking_id'] # m[1]表示track索引,记录id
      track['age'] = 1 # age设置为1
      track['active'] = self.tracks[m[1]]['active'] + 1 # 激活次数加1
      ret.append(track)

    # 逐个处理未匹配检测框，新增跟踪框
    for i in unmatched_dets:
      track = results[i] # 获取检测结果
      self.id_count += 1 # 更新track id
      track['tracking_id'] = self.id_count # 记录track id
      track['age'] = 1 # 设置age为1
      track['active'] =  1 # 设置激活数为1
      ret.append(track)

    # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
    # the object in current frame 
    # 逐个处理未匹配的跟踪框
    for i in unmatched_tracks:
      track = self.tracks[i] # 获取跟踪框
      # 如果age小于跟踪数量
      if track['age'] < self.max_age:
        track['age'] += 1 # 增加age
        track['active'] = 0 # 将激活数设置为0
        ct = track['ct'] # 将该跟踪框的中心设置为原始中心

        # movement in the last second-->如果该track有位移
        if 'tracking' in track:
            offset = track['tracking'] * -1 # move forward
            track['ct'] = ct + offset # 将跟踪框中心位置+前移-->更新该track的中心
        ret.append(track)

    self.tracks = ret # 更新self.tracks
    return ret
