#include <stdio.h>
#include <stdlib.h>
#include "trt_bev_pool_kernel.hpp"
/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n_points]
    ranks_feat       : input index of feat, IntTensor[n_points]
    ranks_bev        : output index, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, z, h, w, c]
*/
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

__global__ void bev_pool_v2_set_zero_kernel(int n_points, float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  float* cur_out = out + idx;
  *cur_out = 0.0; // 清空所有点的值
}

void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out,
  cudaStream_t stream) {
  // kernel<<<Dg, Db, Ns, S>>>(param list);
  // Dg：int型或者dim3类型(x,y,z)，用于定义一个Grid中Block是如何组织的，如果是int型，则表示一维组织结构
  // Db：int型或者dim3类型(x,y,z)，用于定义一个Block中Thread是如何组织的，如果是int型，则表示一维组织结构
  // Ns：size_t类型，可缺省，默认为0； 
  // 用于设置每个block除了静态分配的共享内存外，最多能动态分配的共享内存大小，单位为byte, 0表示不需要动态分配
  // S：cudaStream_t类型，可缺省，默认为0, 表示该核函数位于哪个流
  bev_pool_v2_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256, 0, stream>>>(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}


void bev_pool_v2_set_zero(int n_points, float* out) {
  bev_pool_v2_set_zero_kernel<<<(int)ceil(((double)n_points / 256)), 256>>>(n_points, out);
}