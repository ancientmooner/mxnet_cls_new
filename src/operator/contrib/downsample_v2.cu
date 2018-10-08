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
 * \file downsample_v2.cu
 * \brief DownsampleV2 Operator
 * \author Shaoqing Ren, Xizhou Zhu, Jian Guo, Han Hu
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
#include "../../common/cuda_utils.h"
#include "./downsample_v2-inl.h"

namespace mshadow {
namespace cuda {
namespace downsample_v2 {

template <typename DType>
__global__ void DownsampleV2Forward(const int n, int rescale, int dilate,
                                      const DType* input_data, const int input_spatial_dim,
                                      const int input_height, const int input_width,
                                      const DType* kernel_data, const int kernel_size,
                                      DType* output_data, const int output_spatial_dim,
                                      const int output_height, const int output_width) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int bc = index / output_spatial_dim;
	const int s = index % output_spatial_dim;
    const int oh = s / output_width;
    const int ow = s % output_width;

    __shared__ DType sharemem[9];
   
    if (threadIdx.x < 9) {
        sharemem[threadIdx.x] = kernel_data[threadIdx.x];
    }
    __syncthreads();

    DType kernel_sum = 0;
    const DType* input_data_cur = input_data + bc * input_spatial_dim;
    int kernel_dim = 2 * kernel_size + 1;
    for (int dh = -kernel_size; dh <= kernel_size; dh++) {
        for (int dw = -kernel_size; dw <= kernel_size; dw++) {
            int ih = oh * rescale + dh * dilate;
            int iw = ow * rescale + dw * dilate;
            int kh = kernel_size + dh;
            int kw = kernel_size + dw;
            //ih = min(max(ih, 0), input_height-1);
            //iw = min(max(iw, 0), input_width-1);
            
            DType input_value = 0;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                input_value = input_data_cur[ih * input_width + iw];
            }
            DType kernel_value = sharemem[kh * kernel_dim + kw];
            
            kernel_sum += input_value * kernel_value;
        }
    }
    output_data[index] += kernel_sum;
  }
}

template <typename DType, int X_MEM, int Y_MEM, int STRIDE>
__global__ void DownsampleV2FilterBackward(const int n, int rescale, int dilate,
                                      const int input_spatial_dim,
                                      const int input_height, const int input_width,
                                      const DType* kernel_data, const int kernel_size,
                                      const DType* output_grad, const int output_spatial_dim,
                                      const int output_height, const int output_width,
                                      const DType* input_data, DType* kernel_grad) {
  CUDA_KERNEL_LOOP(index, n) {
    __shared__ DType out_grad_mem[Y_MEM][X_MEM];
    __shared__ DType reduce_mem[X_MEM * Y_MEM];
    __shared__ DType in_data_mem[(Y_MEM + 2) * STRIDE][(X_MEM + 2) * STRIDE];

    const int num_x_part = ceilf(output_width / (float) X_MEM);
    const int num_y_part = ceilf(output_height / (float) Y_MEM);
    const int ind_sp = index % (X_MEM * Y_MEM);
    const int ind_sp_x = ind_sp % X_MEM;
    const int ind_sp_y = ind_sp / X_MEM;
    int ind_x_part = index / (X_MEM * Y_MEM);
    int ind_y_part = ind_x_part / num_x_part;
    const int bc = ind_y_part / num_y_part;
    ind_x_part = ind_x_part % num_x_part;
    ind_y_part = ind_y_part % num_y_part;

    int pad = kernel_size * dilate;
    int kernel_dim = 2 * kernel_size + 1; 
    int out_x_start = ind_x_part * X_MEM;
    int out_y_start = ind_y_part * Y_MEM;
    int in_x_start = out_x_start * rescale - kernel_size * dilate;
    int in_y_start = out_y_start * rescale - kernel_size * dilate; 

    for (int i = ind_sp_x; i < X_MEM * STRIDE + pad * 2; i += X_MEM){
        for (int j = ind_sp_y; j < Y_MEM * STRIDE + pad * 2; j += Y_MEM) {
            int ix = in_x_start + i;
            int iy = in_y_start + j;
            if (ix >= 0 && iy >= 0 && ix < input_width && iy < input_height) {
                int offset_in = bc * input_spatial_dim + iy * input_width + ix;
                in_data_mem[j][i] = input_data[offset_in]; 
            }
            else{
                in_data_mem[j][i] = 0;
            }
        }
        
    }

    if (out_y_start + ind_sp_y < output_height && out_x_start + ind_sp_x < output_width) {
        out_grad_mem[ind_sp_y][ind_sp_x] = output_grad[bc * output_spatial_dim + (out_y_start + ind_sp_y) * output_width + out_x_start + ind_sp_x];
    }
    else {
        out_grad_mem[ind_sp_y][ind_sp_x] = 0;
    }
    __syncthreads();

    for (int i_fy = -kernel_size; i_fy <= kernel_size; ++i_fy) {
        for (int i_fx = -kernel_size; i_fx <= kernel_size; ++i_fx) {
            reduce_mem[ind_sp_y * X_MEM + ind_sp_x] = out_grad_mem[ind_sp_y][ind_sp_x] * in_data_mem[ind_sp_y * STRIDE + (i_fy + kernel_size)][ind_sp_x * STRIDE + (i_fx + kernel_size)];

            for (int isum = X_MEM * Y_MEM / 2; isum > 0; isum >>= 1) {
                __syncthreads();
                if (threadIdx.x < isum) {
                    reduce_mem[threadIdx.x] += reduce_mem[threadIdx.x + isum];
                }
            }
        
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(kernel_grad + (i_fy + kernel_size) * kernel_dim + i_fx + kernel_size, reduce_mem[0]);
            }
        }
    }
  }
}

template <typename DType>
__global__ void DownsampleV2DataBackward(const int n, int rescale, int dilate,
                                      DType* input_grad, const int input_spatial_dim,
                                      const int input_height, const int input_width,
                                      const DType* kernel_data, const int kernel_size,
                                      const DType* output_grad, const int output_spatial_dim,
                                      const int output_height, const int output_width,
                                      const DType* input_data) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int bc = index / input_spatial_dim;
	const int s = index % input_spatial_dim;
    const int ih = s / input_width;
    const int iw = s % input_width;
    int kernel_dim = 2 * kernel_size + 1;

    int pad = kernel_size * dilate;

    __shared__ DType sharemem[9];
   
    if (threadIdx.x < 9) {
        sharemem[threadIdx.x] = kernel_data[threadIdx.x];
    }
    __syncthreads();
    DType localmem[9];
    for (int i =0; i < 9; ++i) {
        localmem[i] = sharemem[i];
    }
    //DType output_grad_value = output_grad[index];
    const int out_h_start = mxnet::common::cuda::CudaMax<int>(
            0, (ih - kernel_dim + pad + rescale) / rescale);
    const int out_h_end = mxnet::common::cuda::CudaMin(
            output_height - 1, (ih + pad) / rescale);
    const int out_w_start = mxnet::common::cuda::CudaMax<int>(
                0, (iw - kernel_dim + pad + rescale) / rescale);
    const int out_w_end = mxnet::common::cuda::CudaMin(
            output_width - 1, (iw + pad) / rescale);

    const int out_grad_offset_temp = bc * output_spatial_dim;
    DType sum = 0;
    for (int out_h = out_h_start; out_h <= out_h_end; ++out_h) {
        const int f_h = ih + pad - out_h * rescale;
        const int filter_offset_h = f_h * kernel_dim;
        const int out_grad_offset_h = out_grad_offset_temp + out_h * output_width;
        for (int out_w = out_w_start; out_w <= out_w_end; ++out_w) {
            const int f_w = iw + pad - out_w * rescale;
            const int filter_offset = filter_offset_h + f_w;
            const int out_grad_offset = out_grad_offset_h + out_w;
            sum += *(output_grad + out_grad_offset) * localmem[filter_offset];
            //sum += *(output_grad + out_grad_offset) * (*(kernel_data + filter_offset));
            
        }
    }
    input_grad[index] += sum;
  }
}

}  // namespace downsample_v2
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class DownsampleV2GPUOp : public Operator{
 public:
  explicit DownsampleV2GPUOp(DownsampleV2Param param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::downsample_v2;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_data = in_data[downsample_v2::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> kernel_data = in_data[downsample_v2::kKernel].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> output_data = out_data[downsample_v2::kOutput].get<xpu, 4, DType>(s);
    if (req[downsample_v2::kOutput] == kWriteTo)
        output_data = 0;
    
    index_t batch_num = input_data.shape_[0];
    index_t channel_num = input_data.shape_[1];
    index_t input_height = input_data.shape_[2];
    index_t input_width = input_data.shape_[3];
    index_t kernel_size = kernel_data.shape_[0] / 2;
    index_t output_height = output_data.shape_[2];
    index_t output_width = output_data.shape_[3];

    index_t num_kernels = batch_num * channel_num * output_height * output_width;
    using namespace mxnet_op;
    DownsampleV2Forward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, param_.rescale, param_.dilate, input_data.dptr_, input_height * input_width, input_height, input_width,
          kernel_data.dptr_, kernel_size, output_data.dptr_, output_height * output_width, output_height, output_width);
    MSHADOW_CUDA_POST_KERNEL_CHECK(DownsampleV2Forward);
    
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
    using namespace mshadow::cuda::downsample_v2;
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_grad = in_grad[downsample_v2::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> kernel_grad = in_grad[downsample_v2::kKernel].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> kernel_data = in_data[downsample_v2::kKernel].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> output_grad = out_grad[downsample_v2::kOutput].get<xpu, 4, DType>(s);

    if (req[downsample_v2::kData] == kWriteTo)
        input_grad = 0;
    
    if (req[downsample_v2::kKernel] == kWriteTo)
        kernel_grad = 0;
    
    const DType* input_data_ptr = NULL;
    DType* kernel_grad_ptr = NULL;
    if (param_.backward_kernel) {
        input_data_ptr = in_data[downsample_v2::kData].get<xpu, 4, DType>(s).dptr_;
        kernel_grad_ptr = kernel_grad.dptr_;
    }
    
    index_t batch_num = input_grad.shape_[0];
    index_t channel_num = input_grad.shape_[1];
    index_t input_height = input_grad.shape_[2];
    index_t input_width = input_grad.shape_[3];
    index_t kernel_size = kernel_data.shape_[0] / 2;
    index_t output_height = output_grad.shape_[2];
    index_t output_width = output_grad.shape_[3];

    index_t num_kernels = batch_num * channel_num * input_height * input_width;
    using namespace mxnet_op;
    DownsampleV2DataBackward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, param_.rescale, param_.dilate, input_grad.dptr_, input_height * input_width, input_height, input_width,
          kernel_data.dptr_, kernel_size, output_grad.dptr_, output_height * output_width, output_height, output_width,
          input_data_ptr);
    MSHADOW_CUDA_POST_KERNEL_CHECK(DownsampleV2DataBackward);

    //if (output_height <= 8 && output_width <= 8){
    if (param_.backward_kernel) {
        if (output_width > 8 && output_height > 8){
            const int X_MEM = 16;
            const int Y_MEM = 16;
            int num_thread = X_MEM * Y_MEM;
            num_kernels = std::ceil(output_width / (float) X_MEM) * std::ceil(output_height / (float) Y_MEM) * (X_MEM * Y_MEM) * batch_num * channel_num;
            
            int max_block = (num_kernels + num_thread - 1) / num_thread;
            const int grid_dim_x = (max_block > mshadow::cuda::kMaxGridDim) ? mshadow::cuda::kMaxGridDim : max_block;
            const int grid_dim_y =
              (max_block > mshadow::cuda::kMaxGridDim) ? (max_block + mshadow::cuda::kMaxGridDim - 1) / mshadow::cuda::kMaxGridDim : 1;
            dim3 num_blocks(grid_dim_x, grid_dim_y);
            dim3 threads_per_block(num_thread);
            if (2 == param_.rescale) {
                DownsampleV2FilterBackward<DType, X_MEM, Y_MEM, 2> // NOLINT_NEXT_LINE(whitespace/operators)
                    << <num_blocks, threads_per_block, 0, mshadow::Stream<gpu>::GetStream(s)>> >
                    (num_kernels, param_.rescale, param_.dilate, input_height * input_width, input_height, input_width,
                    kernel_data.dptr_, kernel_size, output_grad.dptr_, output_height * output_width, output_height, output_width,
                    input_data_ptr, kernel_grad_ptr);
                MSHADOW_CUDA_POST_KERNEL_CHECK(DownsampleV2FilterBackward);
            }
            else if (1 == param_.rescale) {
                DownsampleV2FilterBackward<DType, X_MEM, Y_MEM, 1> // NOLINT_NEXT_LINE(whitespace/operators)
                    << <num_blocks, threads_per_block, 0, mshadow::Stream<gpu>::GetStream(s)>> >
                    (num_kernels, param_.rescale, param_.dilate, input_height * input_width, input_height, input_width,
                    kernel_data.dptr_, kernel_size, output_grad.dptr_, output_height * output_width, output_height, output_width,
                    input_data_ptr, kernel_grad_ptr);
                MSHADOW_CUDA_POST_KERNEL_CHECK(DownsampleV2FilterBackward);
                    
            }
            else{
                printf("rescale not equal to 0 or 1 not supported!\n");
            }
        }
        else {
            const int X_MEM = 8;
            const int Y_MEM = 8;
            int num_thread = X_MEM * Y_MEM;
            num_kernels = std::ceil(output_width / (float) X_MEM) * std::ceil(output_height / (float) Y_MEM) * (X_MEM * Y_MEM) * batch_num * channel_num;
            
            int max_block = (num_kernels + num_thread - 1) / num_thread;
            const int grid_dim_x = (max_block > mshadow::cuda::kMaxGridDim) ? mshadow::cuda::kMaxGridDim : max_block;
            const int grid_dim_y =
              (max_block > mshadow::cuda::kMaxGridDim) ? (max_block + mshadow::cuda::kMaxGridDim - 1) / mshadow::cuda::kMaxGridDim : 1;
            dim3 num_blocks(grid_dim_x, grid_dim_y);
            dim3 threads_per_block(num_thread);
            if (2 == param_.rescale) {
                DownsampleV2FilterBackward<DType, X_MEM, Y_MEM, 2> // NOLINT_NEXT_LINE(whitespace/operators)
                    << <num_blocks, threads_per_block, 0, mshadow::Stream<gpu>::GetStream(s)>> >
                    (num_kernels, param_.rescale, param_.dilate, input_height * input_width, input_height, input_width,
                    kernel_data.dptr_, kernel_size, output_grad.dptr_, output_height * output_width, output_height, output_width,
                    input_data_ptr, kernel_grad_ptr);
                MSHADOW_CUDA_POST_KERNEL_CHECK(DownsampleV2FilterBackward);
            }
            else if (1 == param_.rescale) {
                DownsampleV2FilterBackward<DType, X_MEM, Y_MEM, 1> // NOLINT_NEXT_LINE(whitespace/operators)
                    << <num_blocks, threads_per_block, 0, mshadow::Stream<gpu>::GetStream(s)>> >
                    (num_kernels, param_.rescale, param_.dilate, input_height * input_width, input_height, input_width,
                    kernel_data.dptr_, kernel_size, output_grad.dptr_, output_height * output_width, output_height, output_width,
                    input_data_ptr, kernel_grad_ptr);
                MSHADOW_CUDA_POST_KERNEL_CHECK(DownsampleV2FilterBackward);
                    
            }
            else{
                printf("rescale not equal to 0 or 1 not supported!\n");
            }
        
        }
      }  //}
     }
    
     private:
      DownsampleV2Param param_;
    };  // class DownsampleV2GPUOp
    
    template<>
Operator* CreateOp<gpu>(DownsampleV2Param param) {
  return new DownsampleV2GPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
