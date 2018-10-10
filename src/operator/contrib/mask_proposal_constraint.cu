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
 * \file mask_proposal_constraint.cu
 * \brief MaskProposalConstraint Operator
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
#include "./mask_proposal_constraint-inl.h"


namespace mshadow {
namespace cuda {
namespace mask_proposal_constraint {

template <typename DType>
__global__ void MaskProposalConstraintForward(const int n,
                                      const DType* offset, const DType* mask_constraint, const bool soft_mask,
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
    
    if (mh < 0 || mh > mheight - 1 || mw < 0 || mw > mwidth - 1) // center out, ignore
            continue;
    
    if (conv_imh < 0 || conv_imh > mheight - 1 || conv_imw < 0 || conv_imw > mwidth - 1) // input out, ignore
            continue;

    if (soft_mask) {
        
        DType max_diff_prob = 0;
        for (int m = 0; m < mask_num; m++) {
            DType label_center = mask_constraint[(m * 4 + 2) * mspatial_dim + (mh * mwidth + mw)];
            DType label_input  = mask_constraint[(m * 4 + 2) * mspatial_dim + (conv_imh * mwidth + conv_imw)];
            DType conf_center  = mask_constraint[(m * 4 + 3) * mspatial_dim + (mh * mwidth + mw)];
            DType conf_input   = mask_constraint[(m * 4 + 3) * mspatial_dim + (conv_imh * mwidth + conv_imw)];
            
            if (label_center == 1 && label_input == 0) {
                max_diff_prob = max(max_diff_prob, conf_center-conf_input);
            } 
        } 
        output_mask[index] = 1-max_diff_prob;
        
    } else {
        
        DType max_center_prob = 0;
        DType max_input_prob = 0;
        DType max_both_prob = 0;
        DType max_diff_prob = 0;
        for (int m = 0; m < mask_num; m++) {
            DType conf_center = mask_constraint[(m * 4 + 0) * mspatial_dim + (mh * mwidth + mw)];
            DType conf_input  = mask_constraint[(m * 4 + 1) * mspatial_dim + (conv_imh * mwidth + conv_imw)];
            
            DType sconf_center = mask_constraint[(m * 4 + 1) * mspatial_dim + (mh * mwidth + mw)];
            DType sconf_input  = mask_constraint[(m * 4 + 0) * mspatial_dim + (conv_imh * mwidth + conv_imw)];
            
            if (conf_center > 0) {
                max_center_prob = max(max_center_prob, conf_center);
            }
            
            if (conf_input > 0) {
                max_input_prob = max(max_input_prob, conf_input);
            }
            
            if (conf_center > 0 && conf_input > 0) {
                max_both_prob = max(max_both_prob, conf_center);
            }
            
            if (conf_center > 0 && conf_input == 0) {
                max_diff_prob = max(max_diff_prob, conf_center);
            }
        
            //if (sconf_center == 0 && sconf_input > 0) {
            //    max_diff_prob = max(max_diff_prob, sconf_input);
            //} 
            
        } 
        //if (max_both_prob > max_diff_prob) {
        //    output_mask[index] = 1;
        //} else {
        //    output_mask[index] = 1 - max_diff_prob;
        //}   
        output_mask[index] = 1 - max_diff_prob;
    }    
  }
}

}  // namespace mask_proposal_constraint
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class MaskProposalConstraintGPUOp : public Operator{
 public:
  explicit MaskProposalConstraintGPUOp(MaskProposalConstraintParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::mask_proposal_constraint;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> offset = in_data[maskProposalConstraint::kOffset].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> mask_constraint = in_data[maskProposalConstraint::kMaskConstraint].get<xpu, 4, DType>(s);


    Tensor<xpu, 4, DType> output = out_data[maskProposalConstraint::kOutput].get<xpu, 4, DType>(s);
    CHECK_EQ(req[maskProposalConstraint::kOutput], kWriteTo);
            
    index_t height = offset.shape_[2];
    index_t width  = offset.shape_[3];
    index_t mask_num = mask_constraint.shape_[0];
    index_t mheight  = mask_constraint.shape_[2];
    index_t mwidth   = mask_constraint.shape_[3];
    
    index_t num_kernels = param_.conv_kernel * param_.conv_kernel * height * width;    
    MaskProposalConstraintForward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, offset.dptr_, mask_constraint.dptr_, param_.soft_mask,
           mask_num, height*width, height, width, mheight*mwidth, mheight, mwidth,
           param_.conv_stride, param_.conv_dilate, param_.conv_kernel,
           param_.mask_offset_ratio, param_.ignore_mask,
           output.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(MaskProposalConstraintForward);

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
    using namespace mshadow::cuda::mask_proposal_constraint;
    using namespace mxnet_op;    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> offset_diff = in_grad[maskProposalConstraint::kOffset].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> mask_constraint_diff = in_grad[maskProposalConstraint::kMaskConstraint].get<xpu, 4, DType>(s);
    
    Assign(offset_diff, req[maskProposalConstraint::kOffset], 0);
    Assign(mask_constraint_diff, req[maskProposalConstraint::kMaskConstraint], 0);
  }

 private:
  MaskProposalConstraintParam param_;
};  // class MaskProposalConstraintGPUOp

template<>
Operator* CreateOp<gpu>(MaskProposalConstraintParam param) {
  return new MaskProposalConstraintGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
