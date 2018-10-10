/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file feature_sampling.cu
 * \brief FeatureSampling Operator
 * \author Shaoqing Ren, Xizhou Zhu, Jian Guo
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
#include "./feature_sampling-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {
namespace feature_sampling {

template <typename DType>
__device__ DType deformable_im2col_bilinear(const DType* bottom_data, 
  const int height, const int width, DType h, DType w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high;
  int w_high;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (DType)h_low;
  }
  else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (DType)w_low;
  }
  else {
    w_high = w_low + 1;
  }

  DType lh = h - h_low;
  DType lw = w - w_low;
  DType hh = 1 - lh, hw = 1 - lw;

  DType v1 = bottom_data[h_low * width + w_low];
  DType v2 = bottom_data[h_low * width + w_high];
  DType v3 = bottom_data[h_high * width + w_low];
  DType v4 = bottom_data[h_high * width + w_high];
  DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename DType>
__device__ DType get_gradient_weight(DType argmax_h, DType argmax_w,
  const int h, const int w, const int height, const int width) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    //empty
    return 0;
  }

  argmax_h = max(argmax_h, (DType)0.0f);
  argmax_w = max(argmax_w, (DType)0.0f);

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (DType)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1)
  {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (DType)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  DType weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  } else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}

template <typename DType>
__global__ void FeatureSamplingForward(const int n,
                                      const DType* data, const DType* coord, 
                                      const int channels, const int height, const int width, 
                                      DType* output) {
  CUDA_KERNEL_LOOP(index, n) { 
    const int n = index / channels;
    const int c = index % channels;
    
    const int   n_im = int(coord[n * 3 + 0]);
    const DType h_im = coord[n * 3 + 1];
    const DType w_im = coord[n * 3 + 2];
    const DType* data_cur = data + (n_im * channels + c) * height * width;
    
    DType val = static_cast<DType>(0);
    if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
      val = deformable_im2col_bilinear(data_cur, height, width, h_im, w_im);
    }
    
    output[index] += val;
  }
}

template <typename DType>
__global__ void FeatureSamplingBackward(const int n,
                                      DType* data_grad, const DType* coord, 
                                      const int channels, const int height, const int width, 
                                      const DType* output_grad) {
  CUDA_KERNEL_LOOP(index, n) { 
    const int n = index / channels;
    const int c = index % channels;
    
    const int   n_im = int(coord[n * 3 + 0]);
    const DType h_im = coord[n * 3 + 1];
    const DType w_im = coord[n * 3 + 2];
    
    DType* data_grad_cur = data_grad + (n_im * channels + c) * height * width;
    
    const DType cur_top_grad = output_grad[index];
    const int cur_h = (int)h_im;
    const int cur_w = (int)w_im;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
          cur_w + dx >= 0 && cur_w + dx < width &&
          abs(h_im - (cur_h + dy)) < 1 &&
          abs(w_im - (cur_w + dx)) < 1
          ) {
          int cur_bottom_grad_pos = (cur_h + dy) * width + (cur_w + dx);
          DType weight = get_gradient_weight(h_im, w_im, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(data_grad_cur + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

}  // namespace feature_sampling
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class FeatureSamplingGPUOp : public Operator{
 public:
  explicit FeatureSamplingGPUOp(FeatureSamplingParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::feature_sampling;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data  = in_data[featureSampling::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> coord = in_data[featureSampling::kCoord].get<xpu, 2, DType>(s);

    Tensor<xpu, 2, DType> output = out_data[featureSampling::kOutput].get<xpu, 2, DType>(s);
    if (req[featureSampling::kOutput] == kWriteTo)
            output = 0;

    index_t channels = data.shape_[1];
    index_t height   = data.shape_[2];
    index_t width    = data.shape_[3];
    index_t samples  = coord.shape_[0];
    
    index_t num_kernels = samples * channels;    
    FeatureSamplingForward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, data.dptr_, coord.dptr_, channels, height, width, output.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(FeatureSamplingForward);
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
    using namespace mshadow::cuda::feature_sampling;
    using namespace mxnet_op;    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 2, DType> output_diff = out_grad[featureSampling::kOutput].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> coord = in_data[featureSampling::kCoord].get<xpu, 2, DType>(s);

    Tensor<xpu, 4, DType> data_diff  = in_grad[featureSampling::kData].get<xpu, 4, DType>(s);
    if (req[featureSampling::kData] == kWriteTo)
            data_diff = 0;

    index_t channels = data_diff.shape_[1];
    index_t height   = data_diff.shape_[2];
    index_t width    = data_diff.shape_[3];
    index_t samples  = coord.shape_[0];
    
    index_t num_kernels = samples * channels;    
    FeatureSamplingBackward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, data_diff.dptr_, coord.dptr_, channels, height, width, output_diff.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(FeatureSamplingBackward);
  }

 private:
  FeatureSamplingParam param_;
};  // class FeatureSamplingGPUOp

template<>
Operator* CreateOp<gpu>(FeatureSamplingParam param) {
  return new FeatureSamplingGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
