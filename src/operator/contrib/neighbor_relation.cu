/*!
 * Copyright (c) 2018 Microsoft
 * \file neighbor_relation-inl.h
 * \brief neighbor_relation Operator
 * \author Han Hu
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./neighbor_relation-inl.h"

namespace mshadow {
namespace cuda {
namespace neighbor_relation {

/*
# [batch_size, key_channels, height * 49, width]
warp_key_data = mx.sym.BilinearSampler(data=key_data, grid=offset_grids_repeat)
# [batch_size, key_channels, 49, height, width]
warp_key_data_reshape = mx.sym.reshape(warp_key_data, shape=(0, 0, -4, kernel_size ** 2, -1, -2))
# [batch_size, value_channels, height * 49, width]
warp_value_data = mx.sym.BilinearSampler(data=value_data, grid=offset_grids_repeat)
# [batch_size, num_group, value_channels/num_group, 49, height, width]
warp_value_data_reshape = mx.sym.reshape(warp_value_data, shape=(0, -4, num_group, -1, -4, kernel_size ** 2, -1, -2))
# [batch_size, key_channels, 1, height, width]
query_data_reshape = mx.sym.expand_dims(query_data, axis=2)
# [batch_size, key_channels, 49, height, width]
app_dot = mx.sym.broadcast_mul(query_data_reshape, warp_key_data_reshape)
# [batch_size, num_group, key_channels/num_group, 49, height, width]
app_dot_reshape = mx.sym.reshape(app_dot, shape=(0, -4, num_group, -1, -2))
# [batch_size, num_group, 49, height, width]
app_sim = mx.sym.sum(app_dot_reshape, axis=2) * scale
# [batch_size, num_group, 49, height, width]
app_geo_sim = mx.sym.broadcast_add(app_sim, mx.sym.reshape(pos_weight, (1, num_group, -1, 1, 1)))
# [batch_size, num_group, 49, height, width]
app_geo_sim = mx.sym.softmax(app_geo_sim, axis=2)
# [batch_size, num_group, 1, 49, height, width]
app_geo_sim = mx.sym.expand_dims(app_geo_sim, axis=2)
output_value = mx.sym.reshape(mx.sym.sum(mx.sym.broadcast_mul(app_geo_sim, warp_value_data_reshape), axis=3), shape=(0, -3, -2))
*/

// query: [batch_size, key_channels, height, width]
// key: [batch_size, key_channels, height, width]
// scale:
// num_group:
// pos_weight: has transformed using log operation [num_group, kernel_height, kernel_width]
// output: [batch_size, num_group * kernel_height * kernel_width, height, width]

template <typename DType>
__global__ void SimilarityComputeForwardKernel(const int n,
                                      const DType* key, 
                                      const DType* query, 
                                      const DType* pos_weight,
                                      const int batch_size, 
                                      const int key_channels,
                                      const int height, 
                                      const int width,
                                      const int kernel_height,
                                      const int kernel_width,
                                      const int num_group,
                                      const DType scale,
                                      const DType no_define_value,
                                      const int dilate,
                                      const int stride,
                                      const int in_height,
                                      const int in_width,
                                      const int sim_method,
                                      DType* output) {
  // n = batch_size * num_group * kernel_height * kernel_width * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int kw = h / height;
    h = h % height;
    int kh = kw / kernel_width;
    kw = kw % kernel_width;
    int g = kh / kernel_height;
    kh = kh % kernel_height;
    const int b = g / num_group;
    g = g % num_group;

    const int key_per_group = key_channels / num_group;
    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;
    
    const int spatial_dim = height * width;
    const int in_spatial_dim = in_height * in_width;
    DType sum_sim = 0;
    int query_inds = 0;
    if (sim_method == 0) {
      query_inds = ((b * num_group + g) * key_per_group * height + h) * width + w;
    }
    if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
      int key_inds = ((b * num_group + g) * key_per_group * in_height + h * stride + dilate * (kh - half_kh)) * in_width + w * stride + dilate * (kw - half_kw); 
      for (int i = 0; i < key_per_group; ++i) {
        if (sim_method == 0) {
          sum_sim += query[query_inds + i * spatial_dim] * key[key_inds + i * in_spatial_dim];
        }
        else {
          sum_sim += key[key_inds + i * in_spatial_dim];
        }
      }
      sum_sim *= scale;
    }
    else{
      sum_sim = no_define_value;
    }

    int pos_inds = (g * kernel_height + kh) * kernel_width + kw;
    sum_sim += pos_weight[pos_inds];

    output[index] = sum_sim;
  }
}

template <typename DType>
__global__ void SimilarityComputeBackwardKernel(const int n,
                                      const DType* key, 
                                      const DType* query,
                                      const DType* output_grad,
                                      const int batch_size, 
                                      const int key_channels,
                                      const int height, 
                                      const int width,
                                      const int kernel_height,
                                      const int kernel_width,
                                      const int num_group,
                                      const int key_per_group,
                                      const DType scale,
                                      const int dilate,
                                      const int stride,
                                      const int in_height,
                                      const int in_width,
                                      const int sim_method,
                                      DType* key_grad,
                                      DType* query_grad) {
  // n = batch_size * num_group * key_per_group * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int kpg = h / height;
    h = h % height;
    int g = kpg / key_per_group;
    kpg = kpg % key_per_group;
    const int b = g / num_group;
    g = g % num_group;

    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;
    
    const int spatial_dim = height * width;

    //int query_inds = ((b * num_group + g) * key_per_group * height + h) * width + w;
    int output_inds = ((b * num_group + g) * kernel_height * kernel_width * height + h) * width + w; 
    DType sum_query_grad = 0;

    int key_inds = (((b * num_group + g) * key_per_group + kpg) * in_height + h * stride) * in_width + w * stride; 
    if (sim_method == 0) {
      for (int kh = 0; kh < kernel_height; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
          if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width 
            && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
          //if (kw + w - half_kw >= 0 && kw + w - half_kw < width && kh + h - half_kh >= 0 && kh + h - half_kh < height) {
            sum_query_grad += output_grad[output_inds + (kh * kernel_width + kw) * spatial_dim] * key[key_inds + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)];
          }
        }
      }
      sum_query_grad *= scale;
      query_grad[index] += sum_query_grad;
    }

    DType sum_key_grad = 0;
    int start_kh = -half_kh / stride;
    int end_kh = half_kh / stride;
    int start_kw = -half_kw / stride;
    int end_kw = half_kw / stride;
    key_inds = (((b * num_group + g) * key_per_group + kpg) * in_height + h * stride) * in_width + w * stride; 
    for (int kh = start_kh; kh <= end_kh; ++kh) {
      for (int kw = start_kw; kw <= end_kw; ++kw) {
        if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
          int spatial_offset = dilate * kh * width + dilate * kw;
          if (sim_method == 0) {
            sum_key_grad += output_grad[output_inds + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride) * spatial_dim + spatial_offset]
                     * query[index + spatial_offset];
          }
          else {
            sum_key_grad += output_grad[output_inds + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride) * spatial_dim + spatial_offset];
          }
        }
      }
    }
    sum_key_grad *= scale;
    key_grad[key_inds] += sum_key_grad;

    if (stride == 2){
      if (h * stride + 1 < in_height) {
        sum_key_grad = 0;
        start_kh = (1 - half_kh) / stride;
        end_kh = (half_kh + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              if (sim_method == 0) {
                sum_key_grad += output_grad[output_inds + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride) * spatial_dim + spatial_offset]
                         * query[index + spatial_offset];
              }
              else {
                sum_key_grad += output_grad[output_inds + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride) * spatial_dim + spatial_offset];
              }
            }
          }
        }
        sum_key_grad *= scale;
        key_grad[key_inds + in_width] += sum_key_grad;
      }
      if (w * stride + 1 < in_width) {
        sum_key_grad = 0;
        start_kh = -half_kh / stride;
        end_kh = half_kh / stride;
        start_kw = (1 - half_kw) / stride;
        end_kw = (half_kw + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              if (sim_method == 0) {
                sum_key_grad += output_grad[output_inds + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride + 1) * spatial_dim + spatial_offset]
                         * query[index + spatial_offset];
              }
              else {
                sum_key_grad += output_grad[output_inds + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride + 1) * spatial_dim + spatial_offset];
              }
            }
          }
        }
        sum_key_grad *= scale;
        key_grad[key_inds + 1] += sum_key_grad;
      }
      if (h * stride + 1 < in_height && w * stride + 1 < in_width) {
        sum_key_grad = 0;
        start_kh = (1 - half_kh) / stride;
        end_kh = (half_kh + 1) / stride;
        start_kw = (1 - half_kw) / stride;
        end_kw = (half_kw + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              if (sim_method == 0) {
                sum_key_grad += output_grad[output_inds + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride + 1) * spatial_dim + spatial_offset]
                         * query[index + spatial_offset];
              }
              else {
                sum_key_grad += output_grad[output_inds + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride + 1) * spatial_dim + spatial_offset];
              }
            }
          }
        }
        sum_key_grad *= scale;
        key_grad[key_inds + in_width + 1] += sum_key_grad;
      }
    }
  }
}

template <typename DType, int BLOCK_SIZE>
__global__ void SimilarityComputePositionWeightBackwardKernel(const int n,
                                      const DType* output_grad,
                                      const int batch_size, 
                                      const int spatial_dim, 
                                      const int kernel_height,
                                      const int kernel_width,
                                      const int num_group,
                                      DType* pos_weight_grad) {
  // threadIdx = BLOCK_SIZE / 2
  //
  // output_grad: [batch_size, num_group, kernel_height, kernel_width, height, width]
  // n = num_group * kernel_height * kernel_width * BLOCK_SIZE * (int((height * width - 1) / BLOCK_SIZE) + 1)
  CUDA_KERNEL_LOOP(index, n) {

    __shared__ DType partial_sum[BLOCK_SIZE * 2];

    int spatial_block_num = 1;
    int batch_block_num = 1;
    if (BLOCK_SIZE * 2 < spatial_dim) {
      spatial_block_num = (int)((spatial_dim - 1) / (BLOCK_SIZE * 2)) + 1;
    }
    else {
      batch_block_num = (int)(BLOCK_SIZE * 2 / spatial_dim);
    }
    
    const int tx = threadIdx.x;
    int kw = index / BLOCK_SIZE;
    int kh = kw / kernel_width;
    kw = kw % kernel_width;
    const int g = kh / kernel_height;
    kh = kh % kernel_height;

    int out_inds = ((g * kernel_height + kh) * kernel_width + kw) * spatial_dim;
    const int batch_stride = num_group * kernel_height * kernel_width * spatial_dim;

    int spatial_id = 0;
    int batch_id = 0;
    int sum_val = 0;

    for (int i_batch = 0; i_batch < batch_size; i_batch += batch_block_num) {
      for (int i_spatial = 0; i_spatial < spatial_block_num; ++i_spatial) {
        for (int i_cnt = tx; i_cnt < 2 * BLOCK_SIZE; i_cnt += BLOCK_SIZE) {
          if (spatial_block_num > 1) {
            spatial_id = i_spatial * (BLOCK_SIZE * 2) + i_cnt;
            if (spatial_id < spatial_dim) {
              partial_sum[i_cnt] = output_grad[out_inds + i_batch * batch_stride + spatial_id]; 
            }
            else{
              partial_sum[i_cnt] = 0;
            }
          }
          else {
            spatial_id = i_cnt % spatial_dim;
            batch_id = i_cnt / spatial_dim;
            if (batch_id < batch_block_num && i_batch + batch_id < batch_size) {
              partial_sum[i_cnt] = output_grad[out_inds + (i_batch + batch_id) * batch_stride + spatial_id];
            }
            else {
              partial_sum[i_cnt] = 0;
            }
          }
        }

        for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
          __syncthreads();
          if (tx < stride)
            partial_sum[tx] += partial_sum[tx + stride];
        }

        if (tx == 0) {
          sum_val += partial_sum[tx];
        }
      }
    }

    if (tx == 0) {
      pos_weight_grad[(g * kernel_height + kh) * kernel_width + kw] += sum_val;
    }
  }
}
/*
# [batch_size, num_group, 49, height, width]
app_geo_sim = mx.sym.softmax(app_geo_sim, axis=2)
# [batch_size, num_group, 1, 49, height, width]
app_geo_sim = mx.sym.expand_dims(app_geo_sim, axis=2)
output_value = mx.sym.reshape(mx.sym.sum(mx.sym.broadcast_mul(app_geo_sim, warp_value_data_reshape), axis=3), shape=(0, -3, -2))
*/
// value: [batch_size, value_channels, height, width]
// softmax_data: [batch_size, num_group * kernel_height * kernel_width, height, width]
// num_group:
// output: [batch_size, value_channels, height, width]

template <typename DType>
__global__ void AggregationForwardKernel(const int n,
                                      const DType* value, 
                                      const DType* softmax_data,
                                      const int batch_size, 
                                      const int value_channels,
                                      const int height, 
                                      const int width,
                                      const int kernel_height,
                                      const int kernel_width,
                                      const int num_group,
                                      const int dilate,
                                      const int stride,
                                      const int in_height,
                                      const int in_width,
                                      DType* output) {
  // n = batch_size * value_channels * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int c = h / height;
    h = h % height;
    const int b = c / value_channels;
    c = c % value_channels;

    const int value_per_group = value_channels / num_group;

    const int g = c / value_per_group;
    const int g_in_group = c % value_per_group;

    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;
    
    const int spatial_dim = height * width;
    DType sum_val = 0;

    int value_inds = (((b * num_group + g) * value_per_group + g_in_group) * in_height + h * stride) * in_width + w * stride;
    int softmax_inds = ((b * num_group + g) * kernel_height * kernel_width * height + h) * width + w;
    for (int kh = 0; kh < kernel_height; ++kh) {
      for (int kw = 0; kw < kernel_width; ++kw) {
        if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width 
            && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
          sum_val += value[value_inds + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)] * softmax_data[softmax_inds + kh * kernel_width * spatial_dim + kw * spatial_dim];
           //if ((value_inds) == 10001) {
           //     printf("b: %d, g: %d, h: %d, w: %d, softmax_inds: %d, value_inds: %d, sum_val: %.4f, k:%d, w:%d, softmax: %.4f, val: %.4f\n", 
           //         b, g, h, w, softmax_inds, value_inds, sum_val, kh, kw,
           //         softmax_data[softmax_inds + kh * kernel_width * spatial_dim + kw * spatial_dim],
           //         value[value_inds + dilate * (kh - half_kh) * width + dilate * (kw - half_kw)]);
           //   }
        }
      }
    }
    //if (value_inds % 10000 == 1) {
    //  printf("b: %d, g: %d, h: %d, w: %d, softmax_inds: %d, value_inds: %d, sum_val: %.4f\n", 
    //                b, g, h, w, softmax_inds, value_inds, sum_val);
    //}
    output[index] = sum_val;
  }
}

template <typename DType>
__global__ void AggregationValueBackwardKernel(const int n,
                                      const DType* softmax_data,
                                      const DType* output_grad,
                                      const int batch_size, 
                                      const int value_channels,
                                      const int height, 
                                      const int width,
                                      const int kernel_height,
                                      const int kernel_width,
                                      const int num_group,
                                      const int dilate,
                                      const int stride,
                                      const int in_height,
                                      const int in_width,
                                      DType* value_grad) {
  // n = batch_size * value_channels * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int c = h / height;
    h = h % height;
    const int b = c / value_channels;
    c = c % value_channels;

    const int value_per_group = value_channels / num_group;

    const int g = c / value_per_group;
    const int g_in_group = c % value_per_group;

    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;
    
    const int spatial_dim = height * width;
    DType sum_val = 0;

    int value_inds = (((b * num_group + g) * value_per_group + g_in_group) * in_height + h * stride) * in_width + w * stride;
    int softmax_inds = ((b * num_group + g) * kernel_height * kernel_width * height + h) * width + w;

    int start_kh = -half_kh / stride;
    int end_kh = half_kh / stride;
    int start_kw = -half_kw / stride;
    int end_kw = half_kw / stride;
    for (int kh = start_kh; kh <= end_kh; ++kh) {
      for (int kw = start_kw; kw <= end_kw; ++kw) {
        if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
          int spatial_offset = dilate * kh * width + dilate * kw;
          sum_val += output_grad[index + spatial_offset] 
                  * softmax_data[softmax_inds + spatial_offset + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride) * spatial_dim];
        }
      }
    }
    value_grad[value_inds] += sum_val;

    if (stride == 2){
      if (h * stride + 1 < in_height) {
        sum_val = 0;
        start_kh = (1 - half_kh) / stride;
        end_kh = (half_kh + 1) / stride;
        start_kw = -half_kw / stride;
        end_kw = half_kw / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              sum_val += output_grad[index + spatial_offset] 
                  * softmax_data[softmax_inds + spatial_offset + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride) * spatial_dim];
            }
          }
        }
        value_grad[value_inds + in_width] += sum_val;
      }
      if (w * stride + 1 < in_width) {
        sum_val = 0;
        start_kh = -half_kh / stride;
        end_kh = half_kh / stride;
        start_kw = (1 - half_kw) / stride;
        end_kw = (half_kw + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              sum_val += output_grad[index + spatial_offset] 
                  * softmax_data[softmax_inds + spatial_offset + ((half_kh - kh * stride) * kernel_width + half_kw - kw * stride + 1) * spatial_dim];
            }
          }
        }
        value_grad[value_inds + 1] += sum_val;
      }
      if (h * stride + 1 < in_height && w * stride + 1 < in_width) {
        sum_val = 0;
        start_kh = (1 - half_kh) / stride;
        end_kh = (half_kh + 1) / stride;
        start_kw = (1 - half_kw) / stride;
        end_kw = (half_kw + 1) / stride;
        for (int kh = start_kh; kh <= end_kh; ++kh) {
          for (int kw = start_kw; kw <= end_kw; ++kw) {
            if (dilate * kh + h >= 0 && dilate * kh + h < height && dilate * kw + w >= 0 && dilate * kw + w < width) {
              int spatial_offset = dilate * kh * width + dilate * kw;
              sum_val += output_grad[index + spatial_offset] 
                  * softmax_data[softmax_inds + spatial_offset + ((half_kh - kh * stride + 1) * kernel_width + half_kw - kw * stride + 1) * spatial_dim];
            }
          }
        }
        value_grad[value_inds + in_width + 1] += sum_val;
      }
    }
  }
}

template <typename DType>
__global__ void AggregationSoftmaxBackwardKernel(const int n,
                                      const DType* value,
                                      const DType* output_grad,
                                      const int batch_size, 
                                      const int value_channels,
                                      const int height, 
                                      const int width,
                                      const int kernel_height,
                                      const int kernel_width,
                                      const int num_group,
                                      const int dilate,
                                      const int stride,
                                      const int in_height,
                                      const int in_width,
                                      DType* softmax_grad) {
  // n = batch_size * num_group * kernel_height * kernel_width * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int kw = h / height;
    h = h % height;
    int kh = kw / kernel_width;
    kw = kw % kernel_width;
    int g = kh / kernel_height;
    kh = kh % kernel_height;
    const int b = g / num_group;
    g = g % num_group;
    
    const int half_kh = kernel_height / 2;
    const int half_kw = kernel_width / 2;

    const int value_per_group = value_channels / num_group;
    
    const int spatial_dim = height * width;
    const int in_spatial_dim = in_height * in_width;
    DType sum_val = 0;

    int value_inds = ((b * num_group + g) * value_per_group * in_height + h * stride) * in_width + w * stride;
    int output_inds = ((b * num_group + g) * value_per_group * height + h) * width + w;
    
    if (w * stride + dilate * (kw - half_kw) >= 0 && w * stride + dilate * (kw - half_kw) < in_width && h * stride + dilate * (kh - half_kh) >= 0 && h * stride + dilate * (kh - half_kh) < in_height) {
      for (int iv = 0; iv < value_per_group; ++iv) {
        sum_val += output_grad[output_inds + iv * spatial_dim] * value[value_inds + iv * in_spatial_dim + dilate * (kh - half_kh) * in_width + dilate * (kw - half_kw)];
      }
    }
    softmax_grad[index] = sum_val;
  }
}

}  // namespace neighbor_relation
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class NeighborRelationGPUOp : public Operator{
 public:
  explicit NeighborRelationGPUOp(NeighborRelationParam param) {
    this->param_ = param;
  }
  /*!
   * \brief Dropout kernel, compute dropout tensor
   */
  struct CDropoutKernel {
    /*!
     * \brief Dropout kernel function
     * \param id Thread number (0-based representing count)
     * \param gen Random number generator
     * \param N Total number of items in the output
     * \param step Step between items, related to parallelism
     * \param dropout_out Output dropout values
     * \param mask_out  Output mask (is multiplied to create dropout output, may be 0)
     * \param input_data Input data to perform the dropout on
     * \param pkeep Dropout rate (keep when the generated random number is less than this value)
     */
    MSHADOW_XINLINE static void Map(int id,
                                    RandGenerator<xpu, DType> gen,
                                    const int N,
                                    const int step,
                                    DType *dropout_out,
                                    DType *mask_out,
                                    const DType *input_data,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        mask_out[i] = mshadow_op::threshold::Map<real_t>(rand_num, pkeep) * (1.0f / pkeep);
        dropout_out[i] = input_data[i] * mask_out[i];
      });
    }
  };

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::neighbor_relation;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> key  = in_data[neighborRelation::kKey].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> query = in_data[neighborRelation::kQuery].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> value = in_data[neighborRelation::kValue].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> pos_weight = in_data[neighborRelation::kPos].get<xpu, 3, DType>(s);
    Tensor<xpu, 4, DType> output = out_data[neighborRelation::kOutput].get<xpu, 4, DType>(s);

    if (req[neighborRelation::kOutput] == kWriteTo)
      output = 0;

    int key_channels = key.shape_[1];
    int height   = query.shape_[2];
    int width    = query.shape_[3];
    int batch_size  = key.shape_[0];
    int value_channels = value.shape_[1];
    int in_height   = key.shape_[2];
    int in_width    = key.shape_[3];
    
    batch_step_ = std::min(param_.batch_step, batch_size);
    int sim_size = batch_step_ * param_.num_group * param_.kernel_size * param_.kernel_size * height * width;

    int key_step = batch_step_ * key_channels * in_height * in_width;
    int query_step = batch_step_ * key_channels * height * width;
    int value_step = batch_step_ * value_channels * in_height * in_width;
    int output_step = batch_step_ * value_channels * height * width;

    Tensor<xpu, 1, DType> workspace = ctx.requested[neighborRelation::kTempSpace]
            .get_space_typed<xpu, 1, DType>(Shape1(2 * sim_size + batch_step_ * param_.num_group * height * width), s);
    
    TShape sim_buffer_shape(3);
    sim_buffer_shape[0] = batch_step_ * param_.num_group;
    sim_buffer_shape[1] = param_.kernel_size * param_.kernel_size;
    sim_buffer_shape[2] = height * width;

    TBlob sim_buffer(workspace.dptr_, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> sim_buffer_tensor = sim_buffer.get<xpu, 3, DType>(s);
    
    TBlob softmax_buffer(workspace.dptr_ + sim_size, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> softmax_buffer_tensor = softmax_buffer.get<xpu, 3, DType>(s);

    TShape sum_softmax_buffer_shape(2);
    sum_softmax_buffer_shape[0] = batch_step_ * param_.num_group;
    sum_softmax_buffer_shape[1] = height * width;

    TBlob sum_softmax_buffer(workspace.dptr_ + sim_size * 2, sum_softmax_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 2, DType> sum_softmax_buffer_tensor = sum_softmax_buffer.get<xpu, 2, DType>(s);
   
    Tensor<xpu, 3, DType> mask;
    RandGenerator<xpu, DType> *pgen;
    if (ctx.is_train && param_.dropout_ratio > 0) {
        mask = out_data[neighborRelation::kMask].get<xpu, 3, DType>(s);
        pgen = ctx.requested[neighborRelation::kRandomGen].get_parallel_random<xpu, DType>();
    }
    int M = (batch_size - 1) / batch_step_ + 1;
    for (int i = 0; i < M; ++i) {
        int cur_batch_step = batch_step_;
        if (i == M - 1) {
            cur_batch_step = batch_size - (M - 1) * batch_step_;
            CHECK_EQ(cur_batch_step, batch_step_) << "batch_step must be divided by batch_size";
        }
        
        SimilarityComputeForwardKernel<DType>
            <<<cuda_get_num_blocks(sim_size), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      sim_size, 
                                      key.dptr_ + key_step * i, 
                                      query.dptr_ + query_step * i, 
                                      pos_weight.dptr_,
                                      cur_batch_step, 
                                      key_channels,
                                      height, 
                                      width,
                                      param_.kernel_size,
                                      param_.kernel_size,
                                      param_.num_group,
                                      param_.scale,
                                      param_.no_define_value,
                                      param_.dilate,
                                      param_.stride,
                                      in_height,
                                      in_width,
                                      param_.sim_method,
                                      sim_buffer_tensor.dptr_);
        MSHADOW_CUDA_POST_KERNEL_CHECK(SimilarityComputeForwardKernel);

        // softmax forward
        if (param_.norm_method == 0) {
          Softmax(softmax_buffer_tensor, sim_buffer_tensor);
        }
        else if (param_.norm_method == 1) {
          Assign(sim_buffer_tensor, kWriteTo, F<mshadow_op::relu>(sim_buffer_tensor));
          sum_softmax_buffer_tensor = reduce_with_axis<red::sum, false>(sim_buffer_tensor, 1) + scalar<DType>(1e-6);
          Assign(softmax_buffer_tensor, kWriteTo, sim_buffer_tensor / broadcast_with_axis(sum_softmax_buffer_tensor, 0, param_.kernel_size * param_.kernel_size));
        }

        if (ctx.is_train && param_.dropout_ratio > 0) {
          LaunchRNG<CDropoutKernel, xpu>(s, pgen, sim_size,
                                        softmax_buffer_tensor.dptr_,
                                        mask.dptr_ + sim_size * i,
                                        softmax_buffer_tensor.dptr_,
                                        1.0 - param_.dropout_ratio);
        
        }
       
        //
        AggregationForwardKernel
            <<<cuda_get_num_blocks(output_step), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      output_step,
                                      value.dptr_ + value_step * i, 
                                      softmax_buffer_tensor.dptr_,
                                      cur_batch_step, 
                                      value_channels,
                                      height, 
                                      width,
                                      param_.kernel_size,
                                      param_.kernel_size,
                                      param_.num_group,
                                      param_.dilate,
                                      param_.stride,
                                      in_height,
                                      in_width,
                                      output.dptr_ + i * output_step);
        MSHADOW_CUDA_POST_KERNEL_CHECK(AggregationForwardKernel);
    }

  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::neighbor_relation;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    
    Tensor<xpu, 4, DType> output_grad = out_grad[neighborRelation::kOutput].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> key  = in_data[neighborRelation::kKey].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> query = in_data[neighborRelation::kQuery].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> value = in_data[neighborRelation::kValue].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> pos_weight = in_data[neighborRelation::kPos].get<xpu, 3, DType>(s);

    Tensor<xpu, 4, DType> key_grad  = in_grad[neighborRelation::kKey].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> query_grad = in_grad[neighborRelation::kQuery].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> value_grad = in_grad[neighborRelation::kValue].get<xpu, 4, DType>(s);
    //Tensor<xpu, 3, DType> pos_weight_grad = in_grad[neighborRelation::kPos].get<xpu, 3, DType>(s);
    Tensor<xpu, 1, DType> pos_weight_grad_1d = in_grad[neighborRelation::kPos].get_with_shape<xpu, 1, DType>(Shape1(param_.num_group * param_.kernel_size* param_.kernel_size), s);
    
    Tensor<xpu, 3, DType> mask;
    if (ctx.is_train && param_.dropout_ratio > 0) {
      mask = out_data[neighborRelation::kMask].get<xpu, 3, DType>(s);
    }

    if (req[neighborRelation::kKey] == kWriteTo) {
      key_grad = 0;
    }
    if (req[neighborRelation::kQuery] == kWriteTo) {
      query_grad = 0;
    }
    if (req[neighborRelation::kValue] == kWriteTo) {
      value_grad = 0;
    }
    if (req[neighborRelation::kPos] == kWriteTo) {
      pos_weight_grad_1d = 0;
    }

    int key_channels = key.shape_[1];
    int height   = query.shape_[2];
    int width    = query.shape_[3];
    int batch_size  = key.shape_[0];
    int value_channels = value.shape_[1];
    int in_height   = key.shape_[2];
    int in_width    = key.shape_[3];

    int key_per_group = key_channels / param_.num_group;
    
    batch_step_ = std::min(param_.batch_step, batch_size);
    int sim_size = batch_step_ * param_.num_group * param_.kernel_size * param_.kernel_size * height * width;

    int key_step = batch_step_ * key_channels * in_height * in_width;
    int query_step = batch_step_ * key_channels * height * width;
    int value_step = batch_step_ * value_channels * in_height * in_width;
    int output_step = batch_step_ * value_channels * height * width;

    Tensor<xpu, 1, DType> workspace = ctx.requested[neighborRelation::kTempSpace]
            .get_space_typed<xpu, 1, DType>(Shape1(3 * sim_size + batch_step_ * param_.num_group * height * width), s);
    
    TShape sim_buffer_shape(3);
    sim_buffer_shape[0] = batch_step_ * param_.num_group;
    sim_buffer_shape[1] = param_.kernel_size * param_.kernel_size;
    sim_buffer_shape[2] = height * width;

    TShape sum_softmax_buffer_shape(2);
    sum_softmax_buffer_shape[0] = batch_step_ * param_.num_group;
    sum_softmax_buffer_shape[1] = height * width;

    TBlob sim_buffer(workspace.dptr_, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> sim_buffer_tensor = sim_buffer.get<xpu, 3, DType>(s);
    
    TBlob softmax_buffer(workspace.dptr_ + sim_size, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> softmax_buffer_tensor = softmax_buffer.get<xpu, 3, DType>(s);
    
    TBlob sim_grad_buffer(workspace.dptr_ + sim_size * 2, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> sim_grad_buffer_tensor = sim_grad_buffer.get<xpu, 3, DType>(s);
    
    TBlob sum_softmax_buffer(workspace.dptr_ + sim_size * 3, sum_softmax_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 2, DType> sum_softmax_buffer_tensor = sum_softmax_buffer.get<xpu, 2, DType>(s);
    
    int M = (batch_size - 1) / batch_step_ + 1;

    const int BLOCK_SIZE = 1024;

    const int pos_weight_size = param_.num_group * param_.kernel_size * param_.kernel_size;

    int backward_pos_weight_step = BLOCK_SIZE * param_.kernel_size * param_.kernel_size * param_.num_group;
    // output_grad: [batch_size, num_group, kernel_height, kernel_width, height, width]
    // n = num_group * kernel_height * kernel_width * BLOCK_SIZE * (int((height * width - 1) / BLOCK_SIZE) + 1)



    for (int i = 0; i < M; ++i) {
        int cur_batch_step = batch_step_;
        if (i == M - 1) {
            cur_batch_step = batch_size - (M - 1) * batch_step_;
            CHECK_EQ(cur_batch_step, batch_step_) << "batch_step must be divided by batch_size";
        }
        // sim computation forward 
        SimilarityComputeForwardKernel<DType>
            <<<cuda_get_num_blocks(sim_size), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      sim_size, 
                                      key.dptr_ + key_step * i, 
                                      query.dptr_ + query_step * i, 
                                      pos_weight.dptr_,
                                      cur_batch_step, 
                                      key_channels,
                                      height, 
                                      width,
                                      param_.kernel_size,
                                      param_.kernel_size,
                                      param_.num_group,
                                      param_.scale,
                                      param_.no_define_value,
                                      param_.dilate,
                                      param_.stride,
                                      in_height,
                                      in_width,
                                      param_.sim_method,
                                      sim_buffer_tensor.dptr_);
        MSHADOW_CUDA_POST_KERNEL_CHECK(SimilarityComputeForwardKernel);

        // softmax forward
        if (param_.norm_method == 0) {
          Softmax(softmax_buffer_tensor, sim_buffer_tensor);
        }
        else if (param_.norm_method == 1) {
          Assign(sim_buffer_tensor, kWriteTo, F<mshadow_op::relu>(sim_buffer_tensor));
          sum_softmax_buffer_tensor = reduce_with_axis<red::sum, false>(sim_buffer_tensor, 1) + scalar<DType>(1e-6);
          Assign(softmax_buffer_tensor, kWriteTo, sim_buffer_tensor / broadcast_with_axis(sum_softmax_buffer_tensor, 0, param_.kernel_size * param_.kernel_size));
        }

        // backward to value
        AggregationValueBackwardKernel
            <<<cuda_get_num_blocks(output_step), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      output_step,
                                      softmax_buffer_tensor.dptr_,
                                      output_grad.dptr_ + i * output_step,
                                      cur_batch_step, 
                                      value_channels,
                                      height, 
                                      width,
                                      param_.kernel_size,
                                      param_.kernel_size,
                                      param_.num_group,
                                      param_.dilate,
                                      param_.stride,
                                      in_height,
                                      in_width,
                                      value_grad.dptr_ + i * value_step);
        MSHADOW_CUDA_POST_KERNEL_CHECK(AggregationValueBackwardKernel);

        // backward to softmax grad
        AggregationSoftmaxBackwardKernel
            <<<cuda_get_num_blocks(sim_size), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      sim_size,
                                      value.dptr_ + i * value_step,
                                      output_grad.dptr_ + i * output_step,
                                      cur_batch_step, 
                                      value_channels,
                                      height, 
                                      width,
                                      param_.kernel_size,
                                      param_.kernel_size,
                                      param_.num_group,
                                      param_.dilate,
                                      param_.stride,
                                      in_height,
                                      in_width,
                                      sim_buffer_tensor.dptr_);
        MSHADOW_CUDA_POST_KERNEL_CHECK(AggregationSoftmaxBackwardKernel);

        if (ctx.is_train && param_.dropout_ratio > 0) {
            TBlob mask_buffer(mask.dptr_ + sim_size * i, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
            Tensor<xpu, 3, DType> mask_buffer_tensor = mask_buffer.get<xpu, 3, DType>(s);
            
            Assign(sim_buffer_tensor, kWriteTo, sim_buffer_tensor * mask_buffer_tensor);
        }
        // backward softmax
        // grad of sim written to sim_buffer_tensor
        if (param_.norm_method == 0) {
          sum_softmax_buffer_tensor = reduce_with_axis<red::sum, false>(softmax_buffer_tensor * sim_buffer_tensor, 1);
          Assign(sim_grad_buffer_tensor, kWriteTo,
                 softmax_buffer_tensor * (sim_buffer_tensor - broadcast_with_axis(sum_softmax_buffer_tensor, 0, param_.kernel_size * param_.kernel_size)));
        } 
        else if (param_.norm_method == 1) {
          Assign(sim_grad_buffer_tensor, kWriteTo,
                 (sim_buffer_tensor - broadcast_with_axis(reduce_with_axis<red::sum, false>(softmax_buffer_tensor * sim_buffer_tensor, 1), 0, param_.kernel_size * param_.kernel_size)) / broadcast_with_axis(sum_softmax_buffer_tensor, 0, param_.kernel_size * param_.kernel_size));
          Assign(sim_grad_buffer_tensor, kWriteTo, F<mshadow_op::relu_grad>(softmax_buffer_tensor) * sim_grad_buffer_tensor);
        }
        
        // backward to key & query
        SimilarityComputeBackwardKernel<DType>
            <<<cuda_get_num_blocks(query_step), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      query_step,
                                      key.dptr_ + i * key_step, 
                                      query.dptr_ + i * query_step,
                                      sim_grad_buffer_tensor.dptr_,
                                      batch_step_, 
                                      key_channels,
                                      height, 
                                      width,
                                      param_.kernel_size,
                                      param_.kernel_size,
                                      param_.num_group,
                                      key_per_group,
                                      param_.scale,
                                      param_.dilate,
                                      param_.stride,
                                      in_height,
                                      in_width,
                                      param_.sim_method,
                                      key_grad.dptr_ + i * key_step,
                                      query_grad.dptr_ + i * query_step);
        MSHADOW_CUDA_POST_KERNEL_CHECK(SimilarityComputeBackwardKernel);
        
        // backward to position
        Assign(pos_weight_grad_1d, kAddTo,
            sumall_except_dim<1>(reshape(sim_grad_buffer_tensor, Shape3(batch_step_, pos_weight_size, height * width))));
        /*
        int num_blocks = std::min(mshadow::cuda::kMaxGridNum, (backward_pos_weight_step + BLOCK_SIZE - 1) / BLOCK_SIZE);
        SimilarityComputePositionWeightBackwardKernel<DType, BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      num_blocks * BLOCK_SIZE,
                                      sim_buffer_tensor.dptr_,
                                      batch_step_, 
                                      height * width, 
                                      param_.kernel_size,
                                      param_.kernel_size,
                                      param_.num_group,
                                      pos_weight_grad.dptr_);
        MSHADOW_CUDA_POST_KERNEL_CHECK(SimilarityComputePositionWeightBackwardKernel);
        */
    }
  }

 private:
  NeighborRelationParam param_;
  int batch_step_;
};  // class NeighborRelationGPUOp

template<>
Operator* CreateOp<gpu>(NeighborRelationParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new NeighborRelationGPUOp<gpu, DType>(param);
  })
//#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
  return op;
}
}  // namespace op
}  // namespace mxnet
