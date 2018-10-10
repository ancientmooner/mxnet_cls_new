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
 * \file embed_target_generator.cu
 * \brief EmbedTargetGenerator Operator
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
#include "./embed_target_generator-inl.h"


namespace mshadow {
namespace cuda {
namespace embed_target_generator {

template <typename DType>
__global__ void EmbedTargetGeneratorForward(const int n,
                                      const DType* mask, const int neighborhood_grid_width, const int neighborhood_grid_radius, 
                                      const int stride2, const int mask_num, const int spatial_dim, const int height, const int width, 
                                      DType* output_label) {
  CUDA_KERNEL_LOOP(index, n) { 
    const int ni = index / spatial_dim;
    const int s  = index % spatial_dim;
    const int nh = ni / neighborhood_grid_width;
    const int nw = ni % neighborhood_grid_width;
	const int h = s / width;
    const int w = s % width;
    
    const int offset_h = h + (nh - neighborhood_grid_radius) * stride2;
    const int offset_w = w + (nw - neighborhood_grid_radius) * stride2;
    
    if (offset_h < 0 || offset_h > height - 1 || offset_w < 0 || offset_w > width - 1) {
        output_label[index] = -1;
        continue;
    }
    
    int both_fg = 0;
    int both_bg = 0;
    for (int m = 0; m < mask_num; m++) {
        DType mask_center = mask[(m * 4 + 0) * spatial_dim + (h * width + w)];
        DType mask_offset = mask[(m * 4 + 1) * spatial_dim + (offset_h * width + offset_w)];
        
        if (mask_center > 0 && mask_offset > 0) {
            both_fg++;
        }
        
        if (mask_center == 0 && mask_offset == 0) {
            both_bg++;
        }
    }
    
    if (both_fg > 0) {
        output_label[index] = 1;
    } else if (both_bg == mask_num) {
        output_label[index] = -1;
    } else {
        output_label[index] = 0;
    }
    
  }
}

template <typename DType>
__global__ void EmbedTargetCompressedGeneratorForward(const int n,
                                      const DType* mask, const int neighborhood_grid_width, const int neighborhood_grid_radius, 
                                      const int stride2, const int mask_num, const int spatial_dim, const int height, const int width, 
                                      DType* output_label, DType* output_insts) {
  CUDA_KERNEL_LOOP(index, n) { 
    const int ni = index / spatial_dim;
    const int s  = index % spatial_dim;
    const int nh = ni / neighborhood_grid_width;
    const int nw = ni % neighborhood_grid_width;
	const int h = s / width;
    const int w = s % width;
    
    const int offset_h = h + (nh - neighborhood_grid_radius) * stride2;
    const int offset_w = w + (nw - neighborhood_grid_radius) * stride2;
    
    output_label[index] = 1;
    
    if (offset_h < 0 || offset_h > height - 1 || offset_w < 0 || offset_w > width - 1) {
        continue;
    }
    
    int center_label = int(mask[(h * width + w)]);
    int input_label  = int(mask[(offset_h * width + offset_w)]);
            
    if (center_label > 0) {
        if (input_label >= 0 && input_label != center_label) {
            output_label[index] = 0;
            output_insts[index] = center_label;
        }   
        if (input_label >= 0 && input_label == center_label) {
            output_label[index] = 1;
            output_insts[index] = center_label;
        }
    }
  }
}

}  // namespace embed_target_generator
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class EmbedTargetGeneratorGPUOp : public Operator{
 public:
  explicit EmbedTargetGeneratorGPUOp(EmbedTargetGeneratorParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::embed_target_generator;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> mask = in_data[embedTargetGenerator::kMask].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> output = out_data[embedTargetGenerator::kOutput].get<xpu, 4, DType>(s);
    CHECK_EQ(req[embedTargetGenerator::kOutput], kWriteTo);
            
    index_t mask_num = mask.shape_[0];
    index_t height = mask.shape_[2];
    index_t width  = mask.shape_[3];
    index_t neighborhood_grid_radius = param_.max_displacement / param_.stride2;
    index_t neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

    if (param_.compressed_mask) {
        Tensor<xpu, 4, DType> insts = out_data[embedTargetGenerator::kInst].get<xpu, 4, DType>(s);
        
        index_t num_kernels = neighborhood_grid_width * neighborhood_grid_width * height * width;    
        EmbedTargetCompressedGeneratorForward // NOLINT_NEXT_LINE(whitespace/operators)
            <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
            (num_kernels, mask.dptr_, neighborhood_grid_width, neighborhood_grid_radius, param_.stride2,
            mask_num, height*width, height, width, output.dptr_, insts.dptr_);
        MSHADOW_CUDA_POST_KERNEL_CHECK(EmbedTargetGeneratorForward);
    } else {
        index_t num_kernels = neighborhood_grid_width * neighborhood_grid_width * height * width;    
        EmbedTargetGeneratorForward // NOLINT_NEXT_LINE(whitespace/operators)
            <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
            (num_kernels, mask.dptr_, neighborhood_grid_width, neighborhood_grid_radius, param_.stride2,
            mask_num, height*width, height, width, output.dptr_);
        MSHADOW_CUDA_POST_KERNEL_CHECK(EmbedTargetGeneratorForward);
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
    using namespace mshadow::cuda::embed_target_generator;
    using namespace mxnet_op;    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> mask_diff = in_grad[embedTargetGenerator::kMask].get<xpu, 4, DType>(s);
    
    Assign(mask_diff, req[embedTargetGenerator::kMask], 0);
  }

 private:
  EmbedTargetGeneratorParam param_;
};  // class EmbedTargetGeneratorGPUOp

template<>
Operator* CreateOp<gpu>(EmbedTargetGeneratorParam param) {
  return new EmbedTargetGeneratorGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet