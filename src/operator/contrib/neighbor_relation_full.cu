/*!
 * Copyright (c) 2018 Microsoft
 * \file neighbor_relation_full-inl.h
 * \brief neighbor_relation_full Operator
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
#include "./neighbor_relation_full-inl.h"

namespace mshadow {
namespace cuda {
namespace neighbor_relation_full {

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
// output: [batch_size, num_group * in_height * in_width, height, width]

template <typename DType>
__global__ void SimilarityComputeForwardKernel(const int n,
                                      const DType* key, 
                                      const DType* query, 
                                      const DType* pos_weight,
                                      const int batch_size, 
                                      const int key_channels,
                                      const int query_channels,
                                      const int height, 
                                      const int width,
                                      const int num_group,
                                      const DType scale,
                                      const int dilate,
                                      const int key_stride,
                                      const int in_height,
                                      const int in_width,
                                      const int geo_height,
                                      const int geo_width,
                                      const int sim_method,
                                      DType* output) {
  // n = batch_size * num_group * in_height * in_width * height * width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % width;
    int h = index / width;
    int kw = h / height;
    h = h % height;
    int kh = kw / in_width;
    kw = kw % in_width;
    int g = kh / in_height;
    kh = kh % in_height;
    const int b = g / num_group;
    g = g % num_group;

    const int key_per_group = query_channels / num_group;
    //const int half_kh = kernel_height / 2;
    //const int half_kw = kernel_width / 2;
    
    const int spatial_dim = height * width;
    const int in_spatial_dim = in_height * in_width;
    DType sum_sim = 0;
    int query_inds = 0;
    if (sim_method % 2 == 0) {
      query_inds = ((b * num_group + g) * key_per_group * height + h) * width + w;
    }
    const int key_saliency_group = key_channels - query_channels;
    
    int key_inds =( (b * key_channels + g * key_per_group) * in_height + kh) * in_width + kw;
      
    for (int i = 0; i < key_per_group; ++i) {
      if (sim_method % 2 == 0) {
        sum_sim += query[query_inds + i * spatial_dim] * key[key_inds + i * in_spatial_dim] * scale;
      }
      else {
        sum_sim += key[key_inds + i * in_spatial_dim] * scale;
      }
      if (key_saliency_group > 0) {
          int key_sal_inds = (b * key_channels + query_channels + int(g * key_saliency_group) / num_group) * in_spatial_dim
                      + kh * in_width + kw;
          sum_sim += key[key_sal_inds];
      }
    }

    // pos_weight: [(key_stride-1)/2.0-height+1, floor(height/key_stride)*key_stride-1-(key_stride-1)/2.0]  
    // * [(key_stride-1)/2.0-width+1, floor(width/key_stride)*key_stride-1-(key_stride-1)/2.0] *

    //if ((sim_method / 2) % 2 == 1){
    int pos_inds = g * geo_height * geo_width + (kh * key_stride - h + height - 1) * geo_width + kw * key_stride - w + width - 1;
    sum_sim += pos_weight[pos_inds];
    //}

    output[index] = sum_sim;
  }
}

template <typename DType>
__global__ void SimilarityComputeKeyBackwardKernel(const int n,
                                      const DType* key, 
                                      const DType* query,
                                      const DType* output_grad,
                                      const int batch_size, 
                                      const int key_channels,
                                      const int query_channels,
                                      const int height, 
                                      const int width,
                                      const int num_group,
                                      const int key_per_group,
                                      const DType scale,
                                      const int dilate,
                                      const int key_stride,
                                      const int in_height,
                                      const int in_width,
                                      const int geo_height,
                                      const int geo_width,
                                      const int sim_method,
                                      DType* key_grad) {
  // n = batch_size * num_group * key_per_group * in_height * in_width
  CUDA_KERNEL_LOOP(index, n) { 
    const int w = index % in_width;
    int h = index / in_width;
    int kpg = h / in_height;
    h = h % in_height;
    int g = kpg / key_per_group;
    kpg = kpg % key_per_group;
    const int b = g / num_group;
    g = g % num_group;
    
    const int spatial_dim = height * width;
    const int in_spatial_dim = in_height * in_width;
    const int key_saliency_group = key_channels - query_channels;

    //int query_inds = ((b * num_group + g) * key_per_group * height + h) * width + w;
    // output: [batch_size, num_group * in_height * in_width, height, width]
    int output_inds = ((b * num_group + g) * in_spatial_dim + h * in_width + w) * spatial_dim;
    int query_inds = ((b * num_group + g) * key_per_group + kpg) * spatial_dim;
    DType sum_key_grad = 0;
    DType sum_key_sal_grad = 0;

    for (int qh = 0; qh < height; ++qh) {
      for (int qw = 0; qw < width; ++qw) {
        DType c_out_grad = output_grad[output_inds + qh * width + qw];
        if (sim_method == 0) {
          sum_key_grad += c_out_grad * query[query_inds + qh * width + qw] * scale;
        }
        else {
          sum_key_grad += c_out_grad * scale;
        }
        if (key_saliency_group > 0) {
          sum_key_sal_grad += c_out_grad; 
        }
      }
    }

    int key_inds = (b * key_channels + g * key_per_group + kpg) * in_spatial_dim + h * in_width + w;
    key_grad[key_inds] += sum_key_grad;

    
    if (key_saliency_group > 0) {
      int key_sal_inds = (b * key_channels + g * key_per_group + 
                int(g * key_saliency_group) / num_group) * in_spatial_dim + h * in_width + w;
      atomicAdd(key_grad + key_sal_inds, sum_key_sal_grad);
    }
  }
}

template <typename DType>
__global__ void SimilarityComputeQueryBackwardKernel(const int n,
                                      const DType* key, 
                                      const DType* query,
                                      const DType* output_grad,
                                      const int batch_size, 
                                      const int key_channels,
                                      const int query_channels,
                                      const int height, 
                                      const int width,
                                      const int num_group,
                                      const int key_per_group,
                                      const DType scale,
                                      const int dilate,
                                      const int key_stride,
                                      const int in_height,
                                      const int in_width,
                                      const int geo_height,
                                      const int geo_width,
                                      const int sim_method,
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
    
    const int spatial_dim = height * width;
    const int in_spatial_dim = in_height * in_width;

    //int query_inds = ((b * num_group + g) * key_per_group * height + h) * width + w;
    // output: [batch_size, num_group * in_height * in_width, height, width]
    int output_inds = (b * num_group + g) * in_spatial_dim * spatial_dim + h * width + w;
    //int query_inds = ((b * num_group + g) * key_per_group + kpg) * spatial_dim;
    int key_inds = (b * key_channels + g * key_per_group + kpg) * in_spatial_dim;
    DType sum_query_grad = 0;

    for (int kh = 0; kh < in_height; ++kh) {
      for (int kw = 0; kw < in_width; ++kw) {
        DType c_out_grad = output_grad[output_inds + (kh * in_width + kw) * spatial_dim];

        if (sim_method == 0) {
          sum_query_grad += c_out_grad * key[key_inds + kh * in_width + kw] * scale;
        }
      }
    }
    query_grad[index] += sum_query_grad;
  }
}



// pos_weight_grad: [(key_stride-1)/2.0-height+1, floor(height/key_stride)*key_stride-1-(key_stride-1)/2.0]  
// * [(key_stride-1)/2.0-width+1, floor(width/key_stride)*key_stride-1-(key_stride-1)/2.0]
// [num_group, geo_height, geo_width]
template <typename DType>
__global__ void SimilarityComputePositionWeightBackwardKernel(const int n,
                                      const DType* output_grad,
                                      const int batch_size, 
                                      const int height,
                                      const int width,
                                      const int in_height,
                                      const int in_width,
                                      const int geo_height,
                                      const int geo_width,
                                      const int key_stride,
                                      const int num_group,
                                      DType* pos_weight_grad) {

  // threadIdx = BLOCK_SIZE / 2
  //
  // output_grad: [batch_size, num_group, kernel_height, kernel_width, height, width]
  // n = num_group * kernel_height * kernel_width * BLOCK_SIZE * (int((height * width - 1) / BLOCK_SIZE) + 1)
  CUDA_KERNEL_LOOP(index, n) {
    
    const int w = index % geo_width;
    int h = index / geo_width;
    const int g = h / geo_height;
    h = h % geo_height;

    int start_in_h = max((h - height + 1) / key_stride, 0);
    int end_in_h = min( h / key_stride, in_height - 1);
    int start_in_w = max((w - width + 1) / key_stride, 0);
    int end_in_w = min( w / key_stride, in_width - 1);

    const int spatial_dim = height * width;
    const int in_spatial_dim = in_height * in_width;

    DType sum_geo_grad = 0;
    for (int b = 0; b < batch_size; ++b) {
      for (int hin = start_in_h; hin <= end_in_h; ++hin) {
        int hq = hin * key_stride + height - 1 - h;
        if (hq >= 0 && hq < height) {
          for (int win = start_in_w; win <= end_in_w; ++win) {
            int wq = win * key_stride + width - 1 - w;
            if (wq >= 0 && wq < width) {
                sum_geo_grad += output_grad[((b * num_group + g) * in_spatial_dim + hin * in_width + win) * spatial_dim + hq * width + wq];
            }
          }
        }
      }
    }

    pos_weight_grad[index] += sum_geo_grad;
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
// softmax_data: [batch_size, num_group * in_height * in_width, height, width]
// num_group:
// output: [batch_size, value_channels, height, width]

/*
template <typename DType>
__global__ void AggregationForwardKernel(const int n,
                                      const DType* value, 
                                      const DType* softmax_data,
                                      const int batch_size, 
                                      const int value_channels,
                                      const int height, 
                                      const int width,
                                      const int in_height,
                                      const int in_width,
                                      const int num_group,
                                      const int dilate,
                                      const int key_stride,
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

    const int spatial_dim = height * width;
    const int in_spatial_dim = in_height * in_width;

    DType sum_val = 0;

    int value_inds = ((b * num_group + g) * value_per_group + g_in_group) * in_spatial_dim;
    int softmax_inds = ((b * num_group + g) * in_spatial_dim * height + h) * width + w;
    for (int kh = 0; kh < in_height; ++kh) {
      for (int kw = 0; kw < in_width; ++kw) {
        sum_val += value[value_inds + kh * in_width + kw] * softmax_data[softmax_inds + (kh * in_width + kw) * spatial_dim]; 
      }
    }

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
*/

}  // namespace neighbor_relation_full
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class NeighborRelationFullGPUOp : public Operator{
 public:
  explicit NeighborRelationFullGPUOp(NeighborRelationFullParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::neighbor_relation_full;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> key  = in_data[neighborRelationFull::kKey].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> query = in_data[neighborRelationFull::kQuery].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> value = in_data[neighborRelationFull::kValue].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> pos_weight = in_data[neighborRelationFull::kPos].get<xpu, 3, DType>(s);
    Tensor<xpu, 4, DType> output = out_data[neighborRelationFull::kOutput].get<xpu, 4, DType>(s);

    if (req[neighborRelationFull::kOutput] == kWriteTo)
      output = 0;

    int key_channels = key.shape_[1];
    int query_channels = query.shape_[1];
    int height   = query.shape_[2];
    int width    = query.shape_[3];
    int batch_size  = key.shape_[0];
    int value_channels = value.shape_[1];
    int in_height   = key.shape_[2];
    int in_width    = key.shape_[3];

    int geo_height = pos_weight.shape_[1];
    int geo_width = pos_weight.shape_[2];

    int value_per_group = value_channels / param_.num_group;
    
    batch_step_ = std::min(param_.batch_step, batch_size);
    int sim_size = batch_step_ * param_.num_group * in_height * in_width * height * width;
    int key_step = batch_step_ * key_channels * in_height * in_width;
    int query_step = batch_step_ * query_channels * height * width;
    int value_step = batch_step_ * value_channels * in_height * in_width;
    int output_step = batch_step_ * value_channels * height * width;

    int workspace_sum_size = batch_step_ * param_.num_group * height * width;
    int workspace_batchdot_size = batch_step_ * param_.num_group * 3;
    Tensor<xpu, 1, DType> workspace = ctx.requested[neighborRelationFull::kTempSpace]
            .get_space_typed<xpu, 1, DType>(Shape1(2 * sim_size + workspace_sum_size), s);
    
    Tensor<xpu, 1, DType*> workspace_batchdot = ctx.requested[neighborRelationFull::kTempSpace2]
            .get_space_typed<xpu, 1, DType*>(Shape1(workspace_batchdot_size), s);
    
    TShape sim_buffer_shape(3);
    sim_buffer_shape[0] = batch_step_ * param_.num_group;
    sim_buffer_shape[1] = in_height * in_width;
    sim_buffer_shape[2] = height * width;
    
    TShape value_buffer_shape(3);
    value_buffer_shape[0] = batch_step_ * param_.num_group;
    value_buffer_shape[1] = value_per_group;
    value_buffer_shape[2] = in_height * in_width;
    
    TShape out_buffer_shape(3);
    out_buffer_shape[0] = batch_step_ * param_.num_group;
    out_buffer_shape[1] = value_per_group;
    out_buffer_shape[2] = height * width;

    TBlob sim_buffer(workspace.dptr_, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> sim_buffer_tensor = sim_buffer.get<xpu, 3, DType>(s);
    
    TBlob softmax_buffer(workspace.dptr_ + sim_size, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> softmax_buffer_tensor = softmax_buffer.get<xpu, 3, DType>(s);

    TShape sum_softmax_buffer_shape(2);
    sum_softmax_buffer_shape[0] = batch_step_ * param_.num_group;
    sum_softmax_buffer_shape[1] = height * width;

    TBlob sum_softmax_buffer(workspace.dptr_ + sim_size * 2, sum_softmax_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 2, DType> sum_softmax_buffer_tensor = sum_softmax_buffer.get<xpu, 2, DType>(s);

    //TShape workspace_batchdot_shape(1);
    //workspace_batchdot_shape[0] = workspace_batchdot_size;

    //TBlob workspace_batchdot_buffer(workspace.dptr_ + sim_size * 2 + workspace_sum_size, 
    //                Shape1(workspace_batchdot_size), xpu::kDevMask, DataType<DType*>::kFlag);
    //Tensor<xpu, 1, DType*> workspace_batchdot_tensor = workspace_batchdot_buffer.get<xpu, 1, DType*>(s);
    //Tensor<xpu, 1, DType*> workspace_batchdot_tensor((void *)(workspace.dptr_ + sim_size * 2 + workspace_sum_size), workspace_batchdot_shape);
   
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
                                      query_channels,
                                      height, 
                                      width,
                                      param_.num_group,
                                      param_.scale,
                                      param_.dilate,
                                      param_.key_stride,
                                      in_height,
                                      in_width,
                                      geo_height,
                                      geo_width,
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
          Assign(softmax_buffer_tensor, kWriteTo, sim_buffer_tensor / broadcast_with_axis(sum_softmax_buffer_tensor, 0, in_height * in_width));
        }
        else {
          Assign(softmax_buffer_tensor, kWriteTo, sim_buffer_tensor + scalar<DType>(0));
        }
    
        TBlob value_buffer(value.dptr_ + value_step * i, value_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
        Tensor<xpu, 3, DType> value_buffer_tensor = value_buffer.get<xpu, 3, DType>(s);
    
        TBlob out_buffer(output.dptr_ + output_step * i, out_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
        Tensor<xpu, 3, DType> out_buffer_tensor = out_buffer.get<xpu, 3, DType>(s);
      
        BatchGEMM<false, false>(out_buffer_tensor, value_buffer_tensor, softmax_buffer_tensor, (DType)1.0f,
                                        (DType)0.0f,
                                        workspace_batchdot);
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
    using namespace mshadow::cuda::neighbor_relation_full;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    
    Tensor<xpu, 4, DType> output_grad = out_grad[neighborRelationFull::kOutput].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> key  = in_data[neighborRelationFull::kKey].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> query = in_data[neighborRelationFull::kQuery].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> value = in_data[neighborRelationFull::kValue].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> pos_weight = in_data[neighborRelationFull::kPos].get<xpu, 3, DType>(s);
    
    Tensor<xpu, 4, DType> key_grad  = in_grad[neighborRelationFull::kKey].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> query_grad = in_grad[neighborRelationFull::kQuery].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> value_grad = in_grad[neighborRelationFull::kValue].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> pos_weight_grad = in_grad[neighborRelationFull::kPos].get<xpu, 3, DType>(s);
    //Tensor<xpu, 1, DType> pos_weight_grad_1d = in_grad[neighborRelationFull::kPos].get_with_shape<xpu, 1, DType>(Shape1(param_.num_group * param_.kernel_size* param_.kernel_size), s);

    if (req[neighborRelationFull::kKey] == kWriteTo) {
      key_grad = 0;
    }
    if (req[neighborRelationFull::kQuery] == kWriteTo) {
      query_grad = 0;
    }
    if (req[neighborRelationFull::kValue] == kWriteTo) {
      value_grad = 0;
    }
    if (req[neighborRelationFull::kPos] == kWriteTo) {
      //pos_weight_grad_1d = 0;
      pos_weight_grad = 0;
    }

    int key_channels = key.shape_[1];
    int query_channels = query.shape_[1];
    int height   = query.shape_[2];
    int width    = query.shape_[3];
    int batch_size  = key.shape_[0];
    int value_channels = value.shape_[1];
    int in_height   = key.shape_[2];
    int in_width    = key.shape_[3];

    int geo_height = pos_weight.shape_[1];
    int geo_width = pos_weight.shape_[2];

    int value_per_group = value_channels / param_.num_group;
    int key_per_group = query_channels / param_.num_group;
    
    batch_step_ = std::min(param_.batch_step, batch_size);
    int sim_size = batch_step_ * param_.num_group * in_height * in_width * height * width;
    int key_step = batch_step_ * key_channels * in_height * in_width;
    int key_step_nosal = batch_step_ * query_channels * in_height * in_width;
    int query_step = batch_step_ * query_channels * height * width;
    int value_step = batch_step_ * value_channels * in_height * in_width;
    int output_step = batch_step_ * value_channels * height * width;

    int workspace_sum_size = batch_step_ * param_.num_group * height * width;
    int workspace_batchdot_size = batch_step_ * param_.num_group * 3;
    Tensor<xpu, 1, DType> workspace = ctx.requested[neighborRelationFull::kTempSpace]
            .get_space_typed<xpu, 1, DType>(Shape1(3 * sim_size + workspace_sum_size), s);
    
    Tensor<xpu, 2, DType*> workspace_batchdot = ctx.requested[neighborRelationFull::kTempSpace2]
            .get_space_typed<xpu, 2, DType*>(Shape2(2, workspace_batchdot_size), s);

    Tensor<xpu, 1, DType*> workspace_batchdot1 = workspace_batchdot[0];
    Tensor<xpu, 1, DType*> workspace_batchdot2 = workspace_batchdot[1];
    
    TShape sim_buffer_shape(3);
    sim_buffer_shape[0] = batch_step_ * param_.num_group;
    sim_buffer_shape[1] = in_height * in_width;
    sim_buffer_shape[2] = height * width;
    
    TShape value_buffer_shape(3);
    value_buffer_shape[0] = batch_step_ * param_.num_group;
    value_buffer_shape[1] = value_per_group;
    value_buffer_shape[2] = in_height * in_width;
    
    TShape out_buffer_shape(3);
    out_buffer_shape[0] = batch_step_ * param_.num_group;
    out_buffer_shape[1] = value_per_group;
    out_buffer_shape[2] = height * width;

    TShape sum_softmax_buffer_shape(2);
    sum_softmax_buffer_shape[0] = batch_step_ * param_.num_group;
    sum_softmax_buffer_shape[1] = height * width;

    TShape workspace_batchdot_shape(1);
    workspace_batchdot_shape[0] = workspace_batchdot_size;

    TBlob sim_buffer(workspace.dptr_, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> sim_buffer_tensor = sim_buffer.get<xpu, 3, DType>(s);
    
    TBlob softmax_buffer(workspace.dptr_ + sim_size, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> softmax_buffer_tensor = softmax_buffer.get<xpu, 3, DType>(s);
    
    TBlob sim_grad_buffer(workspace.dptr_ + sim_size * 2, sim_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 3, DType> sim_grad_buffer_tensor = sim_grad_buffer.get<xpu, 3, DType>(s);
    
    TBlob sum_softmax_buffer(workspace.dptr_ + sim_size * 3, sum_softmax_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    Tensor<xpu, 2, DType> sum_softmax_buffer_tensor = sum_softmax_buffer.get<xpu, 2, DType>(s);

    //TBlob workspace_batchdot_buffer1(workspace.dptr_ + sim_size * 3 + workspace_sum_size, 
    //                Shape1(workspace_batchdot_size), xpu::kDevMask, DataType<DType*>::kFlag);

    //Tensor<xpu, 1, DType*> workspace_batchdot_tensor1 = workspace_batchdot_buffer1.get<xpu, 1, DType*>(s);

    //TBlob workspace_batchdot_buffer2(workspace.dptr_ + sim_size * 3 + workspace_sum_size + workspace_batchdot_size * 4, 
    //                Shape1(workspace_batchdot_size), xpu::kDevMask, DataType<DType*>::kFlag);

    //Tensor<xpu, 1, DType*> workspace_batchdot_tensor2 = workspace_batchdot_buffer2.get<xpu, 1, DType*>(s);
   
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
                                      query_channels,
                                      height, 
                                      width,
                                      param_.num_group,
                                      param_.scale,
                                      param_.dilate,
                                      param_.key_stride,
                                      in_height,
                                      in_width,
                                      geo_height,
                                      geo_width,
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
          Assign(softmax_buffer_tensor, kWriteTo, sim_buffer_tensor / broadcast_with_axis(sum_softmax_buffer_tensor, 0, in_height * in_width));
        }
        else {
          Assign(softmax_buffer_tensor, kWriteTo, sim_buffer_tensor + scalar<DType>(0));
        }
    
        TBlob value_grad_buffer(value_grad.dptr_ + value_step * i, value_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
        Tensor<xpu, 3, DType> value_grad_buffer_tensor = value_grad_buffer.get<xpu, 3, DType>(s);
    
        TBlob value_buffer(value.dptr_ + value_step * i, value_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
        Tensor<xpu, 3, DType> value_buffer_tensor = value_buffer.get<xpu, 3, DType>(s);
    
        TBlob out_grad_buffer(output_grad.dptr_ + output_step * i, out_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
        Tensor<xpu, 3, DType> out_grad_buffer_tensor = out_grad_buffer.get<xpu, 3, DType>(s);
     
        // Gradient of z = dot(x, y)
        // dy = dot(x.T, dz)
        // dx = dot(dz, y.T)
        BatchGEMM<true, false>(sim_buffer_tensor, value_buffer_tensor, out_grad_buffer_tensor, (DType)1.0f,
                                        (DType)0.0f,
                                        workspace_batchdot1);
        
        BatchGEMM<false, true>(value_grad_buffer_tensor, out_grad_buffer_tensor, softmax_buffer_tensor, (DType)1.0f,
                                        (kAddTo == req[neighborRelationFull::kValue]) ? (DType)1.0f : (DType)0.0f,
                                        workspace_batchdot2);
    
        // backward softmax
        // grad of sim written to sim_buffer_tensor
        if (param_.norm_method == 0) {
          sum_softmax_buffer_tensor = reduce_with_axis<red::sum, false>(softmax_buffer_tensor * sim_buffer_tensor, 1);
          Assign(sim_grad_buffer_tensor, kWriteTo,
                 softmax_buffer_tensor * (sim_buffer_tensor - broadcast_with_axis(sum_softmax_buffer_tensor, 0, in_height * in_width)));
        } 
        else if (param_.norm_method == 1) {
          Assign(sim_grad_buffer_tensor, kWriteTo,
                 (sim_buffer_tensor - broadcast_with_axis(reduce_with_axis<red::sum, false>(softmax_buffer_tensor * sim_buffer_tensor, 1), 0, in_height * in_width)) / broadcast_with_axis(sum_softmax_buffer_tensor, 0, in_height * in_width));
          Assign(sim_grad_buffer_tensor, kWriteTo, F<mshadow_op::relu_grad>(softmax_buffer_tensor) * sim_grad_buffer_tensor);
        }
        else {
          Assign(sim_grad_buffer_tensor, kWriteTo, sim_buffer_tensor + scalar<DType>(0));
        }
        
        // backward to key & query
        SimilarityComputeKeyBackwardKernel<DType>
            <<<cuda_get_num_blocks(key_step_nosal), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      key_step_nosal,
                                      key.dptr_ + i * key_step, 
                                      query.dptr_ + i * query_step,
                                      sim_grad_buffer_tensor.dptr_,
                                      batch_step_, 
                                      key_channels,
                                      query_channels,
                                      height, 
                                      width,
                                      param_.num_group,
                                      key_per_group,
                                      param_.scale,
                                      param_.dilate,
                                      param_.key_stride,
                                      in_height,
                                      in_width,
                                      geo_height,
                                      geo_width,
                                      param_.sim_method,
                                      key_grad.dptr_ + i * key_step);
        MSHADOW_CUDA_POST_KERNEL_CHECK(SimilarityComputeKeyBackwardKernel);
        if (param_.sim_method == 0) {
          // backward to query
          SimilarityComputeQueryBackwardKernel<DType>
              <<<cuda_get_num_blocks(query_step), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      query_step,
                                      key.dptr_ + i * key_step, 
                                      query.dptr_ + i * query_step,
                                      sim_grad_buffer_tensor.dptr_,
                                      batch_step_, 
                                      key_channels,
                                      query_channels,
                                      height, 
                                      width,
                                      param_.num_group,
                                      key_per_group,
                                      param_.scale,
                                      param_.dilate,
                                      param_.key_stride,
                                      in_height,
                                      in_width,
                                      geo_height,
                                      geo_width,
                                      param_.sim_method,
                                      query_grad.dptr_ + i * query_step);
          MSHADOW_CUDA_POST_KERNEL_CHECK(SimilarityComputeQueryBackwardKernel);
        }

        //if ((param_.sim_method / 2) % 2 == 1) {
          // backward to key & query
          int geo_step = param_.num_group * geo_height * geo_width;

          SimilarityComputePositionWeightBackwardKernel<DType>
              <<<cuda_get_num_blocks(geo_step), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                      geo_step,
                                      sim_grad_buffer_tensor.dptr_,
                                      batch_step_, 
                                      height, 
                                      width,
                                      in_height,
                                      in_width,
                                      geo_height,
                                      geo_width,
                                      param_.key_stride,
                                      param_.num_group,
                                      pos_weight_grad.dptr_);
          MSHADOW_CUDA_POST_KERNEL_CHECK(SimilarityComputePositionWeightBackwardKernel);
        //}
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
  NeighborRelationFullParam param_;
  int batch_step_;
};  // class NeighborRelationFullGPUOp

template<>
Operator* CreateOp<gpu>(NeighborRelationFullParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new NeighborRelationFullGPUOp<gpu, DType>(param);
  })
//#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
  return op;
}
}  // namespace op
}  // namespace mxnet
