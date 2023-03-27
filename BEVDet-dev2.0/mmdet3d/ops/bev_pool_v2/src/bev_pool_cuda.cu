// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111

#include <stdio.h>
#include <stdlib.h>

/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w] (8, 6, 59, 16, 44)
    feat             : input feat, FloatTensor[b,n,h,w,c] (8, 6, 16, 44, 80)
    ranks_depth      : input index of depth, IntTensor[n] (1508751,)
    ranks_feat       : input index of feat, IntTensor[n] (1508751,)
    ranks_bev        : output index, IntTensor[n] (1508751,)
    interval_lengths : starting position for pooled point, IntTensor[n_intervals] (109981,) 
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals] (109981,) 
    out              : output features, FloatTensor[b, d, h, w, c] (8, 1, 128, 128, 80)
*/
// bev_pool_v2与bev_pool_v1的区别是没有提前计算volume，将深度与特征计算推迟到CUDA中
// bev_pool_v1只有取值累加的操作，而bev_pool_v2还多了一次乘法操作
// 只记录索引，也较少了内存的需求
// bev_pool_v1在计算输出内存位置的时候是根据coor坐标累加计算（存在重复计算）
// 而bev_pool_v2则是利用提前计算的索引计算输出特征位置
__global__ void bev_pool_v2_kernel(int c, int n_intervals,
                                  const float *__restrict__ depth,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程id-->真实索引
  int index = idx / c; // pillar索引-->逻辑索引
  int cur_c = idx % c; // 特征索引
  if (index >= n_intervals) return; // 如果pillar索引 >= pillar数量，则直接返回
  // 根据逻辑索引计算线性内存的对应索引
  int interval_start = interval_starts[index]; // 获取pillar索引的起始位置
  int interval_length = interval_lengths[index]; // 获取pillar包含的点数
  float psum = 0;
  const float* cur_depth;
  const float* cur_feat;
  // 一个线程负责取一个pillar内不同点第cur_c维特征
  for(int i = 0; i < interval_length; i++){
    cur_depth = depth + ranks_depth[interval_start+i]; // 获取该点的深度
    // ranks_depth与ranks_feat的维度一致，但是内容不同，一个feat对应D个depth
    // 分清索引和内容的区别
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c; // 获取该点的cur_c维特征
    psum += *cur_feat * *cur_depth; // 累加pillar内所有点的第cur_c维特征
  }
  // 计算输出内存的位置
  const int* cur_rank = ranks_bev + interval_start; // 计算输出pillar的起始位置
  float* cur_out = out + *cur_rank * c + cur_c; // 对应到特征维度-->(cur_c维)
  *cur_out = psum; // 将计算结果赋值
}


/*
  Function: pillar pooling backward
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    out_grad         : gradient of the BEV fmap from top, FloatTensor[b, d, h, w, c]
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    depth_grad       : gradient of the depth fmap, FloatTensor
    feat_grad        : gradient of the feature fmap, FloatTensor
*/
__global__ void bev_pool_grad_kernel(int c, int n_intervals,
                                  const float *__restrict__ out_grad,
                                  const float *__restrict__ depth,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ depth_grad,
                                  float* __restrict__ feat_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程id-->真实索引
  if (idx >= n_intervals) return;
  int interval_start = interval_starts[idx]; // 提取特征起始索引
  int interval_length = interval_lengths[idx]; // 提取该特征包含的点云数量

  // 初始化指针
  const int* cur_rank;
  const float* cur_out_grad;
  const float* cur_out_grad_start;

  const float* cur_feat;
  const float* cur_feat_start;
  float* cur_depth_grad;
  float grad_sum;
  // 计算depth的梯度
  // 逐个点处理
  for(int i = 0; i < interval_length; i++){
    cur_rank = ranks_bev + interval_start + i; // 计算对应的pillar索引
    cur_out_grad_start = out_grad +  * cur_rank * c; // 计算pillar梯度的起始索引
    cur_feat_start = feat + ranks_feat[interval_start+i] * c; // 计算特征的起始索引

    grad_sum = 0;
    // 逐个特征处理
    for(int cur_c = 0; cur_c < c; cur_c++){
      cur_out_grad = cur_out_grad_start + cur_c; // 提取当前pillar的cur_c维特征对应的梯度
      cur_feat = cur_feat_start + cur_c; // 提取当前特征的cur_c维特征
      // 一个深度要和c维度特征相乘--> z = xy --> dx = dz*y
      // 累加所有维度的特征 z = ax + bx + cx --> dx = a + b + c
      grad_sum += *cur_out_grad * *cur_feat; 
    }

    cur_depth_grad = depth_grad + ranks_depth[interval_start+i]; // 计算该点的梯度输出索引
    *cur_depth_grad = grad_sum; // 赋值梯度
  }

  // 计算feat的梯度
  float* cur_feat_grad;
  const float* cur_depth;
  // 逐个特征处理
  for(int cur_c = 0; cur_c < c; cur_c++){
    grad_sum = 0;
    // 逐个点处理
    for(int i = 0; i < interval_length; i++){
      cur_rank = ranks_bev + interval_start + i;  // 计算对应的pillar索引
      cur_out_grad = out_grad + *cur_rank * c + cur_c; // 计算pillar特征梯度的起始索引

      cur_depth = depth + ranks_depth[interval_start+i]; // 提取该特征对应的点
      grad_sum += *cur_out_grad * *cur_depth; // 累加
    }
    cur_feat_grad = feat_grad + ranks_feat[interval_start] * c + cur_c ; // 计算该特征cur_c维度的梯度索引
    * cur_feat_grad = grad_sum; // 赋值梯度
  }
}



void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out) {
  bev_pool_v2_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}

void bev_pool_v2_grad(int c, int n_intervals, const float* out_grad,
  const float* depth, const float* feat, const int* ranks_depth, const int* ranks_feat,
  const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* depth_grad, float* feat_grad) {
  bev_pool_grad_kernel<<<(int)ceil(((double)n_intervals / 256)), 256>>>(
     c, n_intervals, out_grad, depth, feat, ranks_depth, ranks_feat,
     ranks_bev, interval_starts, interval_lengths, depth_grad, feat_grad
  );
}
