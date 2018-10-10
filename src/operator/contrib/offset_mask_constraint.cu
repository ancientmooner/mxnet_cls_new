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
 * \file offset_mask_constraint.cu
 * \brief OffsetMaskConstraint Operator
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
#include "./offset_mask_constraint-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {
namespace offset_mask_constraint {

template <typename DType>
__global__ void OffsetMaskConstraintForward(const int n,
                                      const DType* offset, const DType* mask_constraint, 
                                      const int mask_num, const int spatial_dim, const int height, const int width, 
                                      const int mspatial_dim, const int mheight, const int mwidth, 
                                      const int conv_stride, const int conv_dilate, const int conv_kernel,
                                      const int mask_offset_ratio, const DType ignore_mask, DType* output_mask) {
  CUDA_KERNEL_LOOP(index, n) { 
    const int kindex = index / spatial_dim;
    const int kh = kindex / conv_kernel;
    const int kw = kindex % conv_kernel;
    const int s = index % spatial_dim;  
    const int h = s / width;
	const int w = s % width;
	
    int mh = h * mask_offset_ratio;
    int mw = w * mask_offset_ratio;
    int conv_kernel_radius = conv_kernel / 2;
    
    DType conv_ih = h * conv_stride 
                  + (kh - conv_kernel_radius) * conv_dilate
                  + offset[((kh * conv_kernel + kw) * 2 + 0) * spatial_dim + s] * conv_dilate;
    DType conv_iw = w * conv_stride
                  + (kw - conv_kernel_radius) * conv_dilate
                  + offset[((kh * conv_kernel + kw) * 2 + 1) * spatial_dim + s] * conv_dilate;
    int conv_imh = round(conv_ih * mask_offset_ratio / conv_stride);
    int conv_imw = round(conv_iw * mask_offset_ratio / conv_stride);            

    output_mask[index] = -1;
    
    int center_fg_count = 0;
    int center_bg_count = 0;
    int center_fg_input_fg_count = 0;
    int center_fg_input_bg_count = 0;
    int center_bg_input_fg_count = 0;
    int center_bg_input_bg_count = 0;
    // input/center   fg   border   bg
    //      fg        1      -1      0 
    //    border      -1     -1     -1
    //      bg        0      -1     ig

    // input/center   fg   border   bg
    //      fg        1      ig      0 
    //    border      ig     ig     ig
    //      bg        0      ig     -1    
    
    if (mh < 0 || mh > mheight - 1 || mw < 0 || mw > mwidth - 1) // center out, ignore
            continue;
    
    if (conv_imh < 0 || conv_imh > mheight - 1 || conv_imw < 0 || conv_imw > mwidth - 1) // input out, ignore
            continue;

    for (int m = 0; m < mask_num; m++) {   
        if (mask_constraint[(m * 4 + 1) * mspatial_dim + (mh * mwidth + mw)] == 0) {
            // center bg
            center_bg_count++;
            if (mask_constraint[(m * 4 + 1) * mspatial_dim + (conv_imh * mwidth + conv_imw)] == 0) {
                // input bg
                center_bg_input_bg_count++;
            } else if (mask_constraint[(m * 4 + 0) * mspatial_dim + (conv_imh * mwidth + conv_imw)] == 0) {
                // input border
                continue;
            } else {
                // input fg
                center_bg_input_fg_count++;
            }
        } else if (mask_constraint[(m * 4 + 0) * mspatial_dim + (mh * mwidth + mw)] == 0) {
            // center border
            continue;
        } else {
            // center fg
            center_fg_count++;
            if (mask_constraint[(m * 4 + 1) * mspatial_dim + (conv_imh * mwidth + conv_imw)] == 0) {
                // input bg
                center_fg_input_bg_count++;
            } else if (mask_constraint[(m * 4 + 0) * mspatial_dim + (conv_imh * mwidth + conv_imw)] == 0) {
                // input border
                continue;
            } else {
                // input fg
                center_fg_input_fg_count++;
            }
        }
    }
    
    if (center_fg_count > 0) {
        // center exists fg
        if (center_fg_input_fg_count > 0)
            // input exists the same fg
            output_mask[index] = 1;
        else if (center_fg_input_bg_count == center_fg_count)
            // input strictly bg
            output_mask[index] = 0;
        else 
            // input exists border
            output_mask[index] = ignore_mask;
    } else if (center_bg_count == mask_num) {
        // center strictly bg
        if (center_bg_input_fg_count > 0)
            // input exists fg
            output_mask[index] = 0;
        else if (center_bg_input_bg_count == center_bg_count)
            // input strictly bg
            output_mask[index] = -1;
        else
            // input exists border
            output_mask[index] = ignore_mask;
    } else {
        // center exists border
        output_mask[index] = ignore_mask;
    }
  }
}

template <typename DType>
__global__ void OffsetCompressedMaskConstraintForward(const int n,
                                      const DType* offset, const DType* mask_constraint, 
                                      const int mask_num, const int spatial_dim, const int height, const int width, 
                                      const int mspatial_dim, const int mheight, const int mwidth, 
                                      const int conv_stride, const int conv_dilate, const int conv_kernel,
                                      const int mask_offset_ratio, const DType ignore_mask,
                                      DType* output_mask, DType* output_label) {
  CUDA_KERNEL_LOOP(index, n) { 
    const int kindex = index / spatial_dim;
    const int kh = kindex / conv_kernel;
    const int kw = kindex % conv_kernel;
    const int s = index % spatial_dim;  
    const int h = s / width;
	const int w = s % width;
	
    int mh = h * mask_offset_ratio;
    int mw = w * mask_offset_ratio;
    int conv_kernel_radius = conv_kernel / 2;
    
    DType conv_ih = h * conv_stride 
                  + (kh - conv_kernel_radius) * conv_dilate
                  + offset[((kh * conv_kernel + kw) * 2 + 0) * spatial_dim + s] * conv_dilate;
    DType conv_iw = w * conv_stride
                  + (kw - conv_kernel_radius) * conv_dilate
                  + offset[((kh * conv_kernel + kw) * 2 + 1) * spatial_dim + s] * conv_dilate;
    int conv_imh = round(conv_ih * mask_offset_ratio / conv_stride);
    int conv_imw = round(conv_iw * mask_offset_ratio / conv_stride);            

    output_mask[index] = -1;
    output_label[index] = -1;
    
    if (mh < 0 || mh > mheight - 1 || mw < 0 || mw > mwidth - 1) // center out, ignore
            continue;
    
    if (conv_imh < 0 || conv_imh > mheight - 1 || conv_imw < 0 || conv_imw > mwidth - 1) // input out, ignore
            continue;

    int center_label = int(mask_constraint[(mh * mwidth + mw)]);
    int input_label  = int(mask_constraint[(conv_imh * mwidth + conv_imw)]);
            
    if (center_label > 0) {
        if (input_label >= 0 && input_label != center_label) {
            output_mask[index] = 0;
            output_label[index] = center_label;
        }   
        if (input_label >= 0 && input_label == center_label) {
            output_mask[index] = 1;
            output_label[index] = center_label;
        }
    }  
            
  }
}


template <typename DType>
__global__ void OffsetMaskConstraintBackward(const int n,
                                      const DType* offset, const DType* mask_constraint, 
                                      const int mask_num, const int spatial_dim, const int height, const int width, 
                                      const int mspatial_dim, const int mheight, const int mwidth, 
                                      const int conv_stride, const int conv_dilate, const int conv_kernel,
                                      const int mask_offset_ratio, const float grad_scale, const bool border_constraint,
                                      DType* offset_grad) {
  CUDA_KERNEL_LOOP(index, n) { 
    const int kindex = index / spatial_dim;
    const int kh = kindex / conv_kernel;
    const int kw = kindex % conv_kernel;
    const int s = index % spatial_dim;  
    const int h = s / width;
	const int w = s % width;
	
    int mh = h * mask_offset_ratio;
    int mw = w * mask_offset_ratio;
    int conv_kernel_radius = conv_kernel / 2;
    
    DType conv_ih = h * conv_stride 
                  + (kh - conv_kernel_radius) * conv_dilate
                  + offset[((kh * conv_kernel + kw) * 2 + 0) * spatial_dim + s] * conv_dilate;
    DType conv_iw = w * conv_stride
                  + (kw - conv_kernel_radius) * conv_dilate
                  + offset[((kh * conv_kernel + kw) * 2 + 1) * spatial_dim + s] * conv_dilate;
    int conv_imh = round(conv_ih * mask_offset_ratio / conv_stride);
    int conv_imw = round(conv_iw * mask_offset_ratio / conv_stride);            

    if (mh < 0 || mh > mheight - 1 || mw < 0 || mw > mwidth - 1)
            continue;
    
    for (int m = 0; m < mask_num; m++) {     
        DType mask_weight = mask_constraint[(m * 4 + 0) * mspatial_dim + (mh * mwidth + mw)] * mask_offset_ratio * mask_offset_ratio / mask_num;
        if (mask_weight > 0) {
            DType target_h, target_w;
            if (conv_imh < 0 || conv_imh > mheight - 1 || conv_imw < 0 || conv_imw > mwidth - 1) {
                if (border_constraint) {
                    target_h = min(max(conv_ih, DType(0)), DType(mheight - 1) * conv_stride / mask_offset_ratio);
                    target_w = min(max(conv_iw, DType(0)), DType(mwidth  - 1) * conv_stride / mask_offset_ratio);
                } else {
                    target_h = conv_ih;
                    target_w = conv_iw;
                } 
            } else {
                if (mask_constraint[(m * 4 + 1) * mspatial_dim + (conv_imh * mwidth + conv_imw)] > 0) {
                    target_h = conv_ih;
                    target_w = conv_iw;
                } else {
                    target_h = mask_constraint[(m * 4 + 2) * mspatial_dim + (conv_imh * mwidth + conv_imw)] / mask_offset_ratio * conv_stride;
                    target_w = mask_constraint[(m * 4 + 3) * mspatial_dim + (conv_imh * mwidth + conv_imw)] / mask_offset_ratio * conv_stride;
                } 
            }

            DType offset_grad_h = (conv_ih - target_h) * conv_dilate;
            DType offset_grad_w = (conv_iw - target_w) * conv_dilate;
            offset_grad[((kh * conv_kernel + kw) * 2 + 0) * spatial_dim + s] += offset_grad_h * mask_weight * grad_scale;
            offset_grad[((kh * conv_kernel + kw) * 2 + 1) * spatial_dim + s] += offset_grad_w * mask_weight * grad_scale;
        }
    }
    
  }
}

}  // namespace offset_mask_constraint
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class OffsetMaskConstraintGPUOp : public Operator{
 public:
  explicit OffsetMaskConstraintGPUOp(OffsetMaskConstraintParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::offset_mask_constraint;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> offset = in_data[offsetMaskConstraint::kOffset].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> mask_constraint = in_data[offsetMaskConstraint::kMaskConstraint].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> output = out_data[offsetMaskConstraint::kOutput].get<xpu, 4, DType>(s);
    if (req[offsetMaskConstraint::kOutput] == kWriteTo)
            output = 0;
    output += offset;
    
    if (param_.output_mask) {
        CHECK_EQ(req[offsetMaskConstraint::kGTMask], kWriteTo);
        Tensor<xpu, 4, DType> mask = out_data[offsetMaskConstraint::kGTMask].get<xpu, 4, DType>(s);
        
        index_t height = offset.shape_[2];
        index_t width  = offset.shape_[3];
        index_t mask_num = mask_constraint.shape_[0];
        index_t mheight  = mask_constraint.shape_[2];
        index_t mwidth   = mask_constraint.shape_[3];
        
        if (param_.compressed_mask) {
            Tensor<xpu, 4, DType> label = out_data[offsetMaskConstraint::kGTLabel].get<xpu, 4, DType>(s);
            
            index_t num_kernels = param_.conv_kernel * param_.conv_kernel * height * width;    
            OffsetCompressedMaskConstraintForward // NOLINT_NEXT_LINE(whitespace/operators)
                  <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
                  (num_kernels, offset.dptr_, mask_constraint.dptr_,
                   mask_num, height*width, height, width, mheight*mwidth, mheight, mwidth,
                   param_.conv_stride, param_.conv_dilate, param_.conv_kernel,
                   param_.mask_offset_ratio, param_.ignore_mask,
                   mask.dptr_, label.dptr_);
            MSHADOW_CUDA_POST_KERNEL_CHECK(OffsetCompressedMaskConstraintForward);
        } else {
            index_t num_kernels = param_.conv_kernel * param_.conv_kernel * height * width;    
            OffsetMaskConstraintForward // NOLINT_NEXT_LINE(whitespace/operators)
                  <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
                  (num_kernels, offset.dptr_, mask_constraint.dptr_,
                   mask_num, height*width, height, width, mheight*mwidth, mheight, mwidth,
                   param_.conv_stride, param_.conv_dilate, param_.conv_kernel,
                   param_.mask_offset_ratio, param_.ignore_mask,
                   mask.dptr_);
            MSHADOW_CUDA_POST_KERNEL_CHECK(OffsetMaskConstraintForward);
        }   
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
    using namespace mshadow::cuda::offset_mask_constraint;
    using namespace mxnet_op;    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> output_diff = out_grad[offsetMaskConstraint::kOutput].get<xpu, 4, DType>(s);
    
    Tensor<xpu, 4, DType> offset = in_data[offsetMaskConstraint::kOffset].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> mask_constraint = in_data[offsetMaskConstraint::kMaskConstraint].get<xpu, 4, DType>(s);
    
    Tensor<xpu, 4, DType> offset_diff = in_grad[offsetMaskConstraint::kOffset].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> mask_constraint_diff = in_grad[offsetMaskConstraint::kMaskConstraint].get<xpu, 4, DType>(s);
    
    if (req[offsetMaskConstraint::kOffset] == kWriteTo)
        offset_diff = 0;
    if (req[offsetMaskConstraint::kMaskConstraint] == kWriteTo)
        mask_constraint_diff = 0;

    offset_diff += output_diff;
    
    index_t height = offset.shape_[2];
    index_t width  = offset.shape_[3];
    index_t mask_num = mask_constraint.shape_[0];
    index_t mheight  = mask_constraint.shape_[2];
    index_t mwidth   = mask_constraint.shape_[3];
    
    index_t num_kernels = param_.conv_kernel * param_.conv_kernel * height * width;    
    OffsetMaskConstraintBackward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, offset.dptr_, mask_constraint.dptr_,
           mask_num, height*width, height, width, mheight*mwidth, mheight, mwidth,
           param_.conv_stride, param_.conv_dilate, param_.conv_kernel,
           param_.mask_offset_ratio, param_.grad_scale, param_.border_constraint,
           offset_diff.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(OffsetMaskConstraintBackward);
  }

 private:
  OffsetMaskConstraintParam param_;
};  // class OffsetMaskConstraintGPUOp

template<>
Operator* CreateOp<gpu>(OffsetMaskConstraintParam param) {
  return new OffsetMaskConstraintGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
