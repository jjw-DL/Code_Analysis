import numpy as np

def greedy_assignment(dist):
  matched_indices = [] # 初始化匹配索引
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  # 逐个检测框处理
  for i in range(dist.shape[0]):
    j = dist[i].argmin() # 计算最小距离的索引
    if dist[i][j] < 1e16: # 如果距离有效
      dist[:, j] = 1e18 # 将已经匹配完的框的距离设置为更大的值，避免二次匹配
      matched_indices.append([i, j]) # 记录匹配索引
  return np.array(matched_indices, np.int32).reshape(-1, 2)
