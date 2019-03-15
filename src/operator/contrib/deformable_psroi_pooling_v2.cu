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
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_psroi_pooling.cu
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
#include "./deformable_psroi_pooling_v2-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

#define DeformablePSROIPOOLING_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow {
namespace cuda {
  template <typename DType>
  __device__ DType bilinear_interp_v2(
    const DType* data,
    const DType x,
    const DType y,
    const int width,
    const int height) {
    int x1 = floor(x);
    int x2 = ceil(x);
    int y1 = floor(y);
    int y2 = ceil(y);
    DType dist_x = static_cast<DType>(x - x1);
    DType dist_y = static_cast<DType>(y - y1);
    DType value11 = data[y1*width + x1];
    DType value12 = data[y2*width + x1];
    DType value21 = data[y1*width + x2];
    DType value22 = data[y2*width + x2];
    DType value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12
      + dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
    return value;
  }

  template <typename DType>
  __global__ void DeformablePSROIPoolv2ForwardKernel(
    const int count,
    const DType** stage_bottom_data,
    const DType* stage_spatial_scale,
    const int channels,
    const int* stage_height, const int* stage_width,
    const int pooled_height, const int pooled_width,
    const DType* bottom_rois, const DType* bottom_trans,
    const bool no_trans,
    const DType trans_std,
    const int sample_per_part,
    const int output_dim,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class,
    DType* top_data,
    DType* top_count) {
    CUDA_KERNEL_LOOP(index, count) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      const DType* offset_bottom_rois = bottom_rois + n * 6;
      int roi_stage_ind = offset_bottom_rois[0];
      const DType* bottom_data = stage_bottom_data[roi_stage_ind];
      const int height = stage_height[roi_stage_ind];
      const int width = stage_width[roi_stage_ind];
      const DType spatial_scale = stage_spatial_scale[roi_stage_ind];
      
      int roi_batch_ind = offset_bottom_rois[1];
      DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
      DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[3])) * spatial_scale - 0.5;
      DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
      DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[5]) + 1.) * spatial_scale - 0.5;

      // Force too small ROIs to be 1x1
      DType roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      DType roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
      DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

      DType sub_bin_size_h = bin_size_h / static_cast<DType>(sample_per_part);
      DType sub_bin_size_w = bin_size_w / static_cast<DType>(sample_per_part);

      int part_h = floor(static_cast<DType>(ph) / pooled_height*part_size);
      int part_w = floor(static_cast<DType>(pw) / pooled_width*part_size);
      int class_id = ctop / channels_each_class;
      DType trans_x = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;
      DType trans_y = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2 + 1)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;

      DType wstart = static_cast<DType>(pw)* bin_size_w
        + roi_start_w;
      wstart += trans_x * roi_width;
      DType hstart = static_cast<DType>(ph) * bin_size_h
        + roi_start_h;
      hstart += trans_y * roi_height;

      DType sum = 0;
      int count = 0;
      int gw = floor(static_cast<DType>(pw) * group_size / pooled_width);
      int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
      gw = min(max(gw, 0), group_size - 1);
      gh = min(max(gh, 0), group_size - 1);

      const DType* offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
      for (int ih = 0; ih < sample_per_part; ih++) {
        for (int iw = 0; iw < sample_per_part; iw++) {
          DType w = wstart + iw*sub_bin_size_w;
          DType h = hstart + ih*sub_bin_size_h;
          // bilinear interpolation
          if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
            continue;
          }
          w = min(max(w, 0.), width - 1.);
          h = min(max(h, 0.), height - 1.);
          int c = (ctop*group_size + gh)*group_size + gw;
          DType val = bilinear_interp_v2(offset_bottom_data + c*height*width, w, h, width, height);
          sum += val;
          count++;
        }
      }
      top_data[index] = count == 0 ? static_cast<DType>(0) : sum / count;
      top_count[index] = count;
    }
  }

  template<typename DType>
  inline void DeformablePSROIPoolv2Forward(const Tensor<gpu, 4, DType> &out,
    const std::vector<Tensor<gpu, 4, DType>> &datas,
    const Tensor<gpu, 2, DType> &bbox,
    const Tensor<gpu, 4, DType> &trans,
    const Tensor<gpu, 4, DType> &top_count,
    const bool no_trans,
    const nnvm::Tuple<float> spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {
    // LOG(INFO) << "DeformablePSROIPoolForward";

    std::vector<const DType*> _stage_bottom_data;
    for (int i = 0; i < datas.size(); i++)
        _stage_bottom_data.push_back(datas[i].dptr_);
    const DType **stage_bottom_data;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_bottom_data, sizeof(const DType*) * _stage_bottom_data.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_bottom_data, &_stage_bottom_data[0], sizeof(const DType*) * _stage_bottom_data.size(),
                                cudaMemcpyHostToDevice));
    
    const DType *bottom_rois = bbox.dptr_;
    const DType *bottom_trans = no_trans ? NULL : trans.dptr_;
    DType *top_data = out.dptr_;
    DType *top_count_data = top_count.dptr_;
    const int count = out.shape_.Size();
    const int channels = datas[0].size(1);
    
    std::vector<int> _stage_height;
    for (int i = 0; i < datas.size(); i++)
        _stage_height.push_back(datas[i].size(2));
    int *stage_height;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_height, sizeof(int) * _stage_height.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_height, &_stage_height[0], sizeof(int) * _stage_height.size(),
                                cudaMemcpyHostToDevice));

    std::vector<int> _stage_width;
    for (int i = 0; i < datas.size(); i++)
        _stage_width.push_back(datas[i].size(3));
    int *stage_width;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_width, sizeof(int) * _stage_width.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_width, &_stage_width[0], sizeof(int) * _stage_width.size(),
                                cudaMemcpyHostToDevice));
                                
    const int pooled_height = pooled_size;
    const int pooled_width = pooled_size;
    const int num_classes = no_trans ? 1 : trans.size(1) / 2;
    const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

    std::vector<DType> _stage_spatial_scale;
    for (int i = 0; i < spatial_scale.ndim(); i++)
        _stage_spatial_scale.push_back(spatial_scale[i]);
    DType *stage_spatial_scale;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_spatial_scale, sizeof(DType) * _stage_spatial_scale.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_spatial_scale, &_stage_spatial_scale[0], sizeof(DType) * _stage_spatial_scale.size(),
                                cudaMemcpyHostToDevice));
    
    
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    DeformablePSROIPoolv2ForwardKernel<DType><<<mxnet::op::mxnet_op::cuda_get_num_blocks(count), kBaseThreadNum, 0, stream>>>(
      count, stage_bottom_data, stage_spatial_scale, channels, stage_height, stage_width, pooled_height, pooled_width,
      bottom_rois, bottom_trans, no_trans, trans_std, sample_per_part, output_dim,
      group_size, part_size, num_classes, channels_each_class, top_data, top_count_data);
    DeformablePSROIPOOLING_CUDA_CHECK(cudaPeekAtLastError());
    
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_bottom_data));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_height));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_width));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_spatial_scale));
  }


  template <typename DType>
  __global__ void DeformablePSROIPoolv2BackwardAccKernel(
    const int count,
    const DType* top_diff,
    const DType* top_count,
    const int num_rois,
    const DType* stage_spatial_scale,
    const int channels,
    const int* stage_height, const int* stage_width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    DType** stage_bottom_data_diff, DType* bottom_trans_diff,
    const DType** stage_bottom_data,
    const DType* bottom_rois,
    const DType* bottom_trans,
    const bool no_trans,
    const DType trans_std,
    const int sample_per_part,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class) {
    CUDA_KERNEL_LOOP(index, count) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      const DType* offset_bottom_rois = bottom_rois + n * 6;
      int roi_stage_ind = offset_bottom_rois[0];
      DType* bottom_data_diff = stage_bottom_data_diff[roi_stage_ind];
      const DType* bottom_data = stage_bottom_data[roi_stage_ind];
      const int height = stage_height[roi_stage_ind];
      const int width = stage_width[roi_stage_ind];
      const DType spatial_scale = stage_spatial_scale[roi_stage_ind];
      
      int roi_batch_ind = offset_bottom_rois[1];
      DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
      DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[3])) * spatial_scale - 0.5;
      DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
      DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[5]) + 1.) * spatial_scale - 0.5;

      // Force too small ROIs to be 1x1
      DType roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      DType roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
      DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

      DType sub_bin_size_h = bin_size_h / static_cast<DType>(sample_per_part);
      DType sub_bin_size_w = bin_size_w / static_cast<DType>(sample_per_part);

      int part_h = floor(static_cast<DType>(ph) / pooled_height*part_size);
      int part_w = floor(static_cast<DType>(pw) / pooled_width*part_size);
      int class_id = ctop / channels_each_class;
      DType trans_x = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;
      DType trans_y = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2 + 1)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;

      DType wstart = static_cast<DType>(pw)* bin_size_w
        + roi_start_w;
      wstart += trans_x * roi_width;
      DType hstart = static_cast<DType>(ph) * bin_size_h
        + roi_start_h;
      hstart += trans_y * roi_height;

      if (top_count[index] <= 0) {
        continue;
      }
      DType diff_val = top_diff[index] / top_count[index];
      const DType* offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
      DType* offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
      int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
      int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
      gw = min(max(gw, 0), group_size - 1);
      gh = min(max(gh, 0), group_size - 1);

      for (int ih = 0; ih < sample_per_part; ih++) {
        for (int iw = 0; iw < sample_per_part; iw++) {
          DType w = wstart + iw*sub_bin_size_w;
          DType h = hstart + ih*sub_bin_size_h;
          // bilinear interpolation
          if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
            continue;
          }
          w = min(max(w, 0.), width - 1.);
          h = min(max(h, 0.), height - 1.);
          int c = (ctop*group_size + gh)*group_size + gw;
          // backward on feature
          int x0 = floor(w);
          int x1 = ceil(w);
          int y0 = floor(h);
          int y1 = ceil(h);
          DType dist_x = w - x0, dist_y = h - y0;
          DType q00 = (1 - dist_x)*(1 - dist_y);
          DType q01 = (1 - dist_x)*dist_y;
          DType q10 = dist_x*(1 - dist_y);
          DType q11 = dist_x*dist_y;
          int bottom_index_base = c * height *width;
          atomicAdd(offset_bottom_data_diff + bottom_index_base + y0*width + x0, q00*diff_val);
          atomicAdd(offset_bottom_data_diff + bottom_index_base + y1*width + x0, q01*diff_val);
          atomicAdd(offset_bottom_data_diff + bottom_index_base + y0*width + x1, q10*diff_val);
          atomicAdd(offset_bottom_data_diff + bottom_index_base + y1*width + x1, q11*diff_val);

          if (no_trans) {
            continue;
          }
          DType U00 = offset_bottom_data[bottom_index_base + y0*width + x0];
          DType U01 = offset_bottom_data[bottom_index_base + y1*width + x0];
          DType U10 = offset_bottom_data[bottom_index_base + y0*width + x1];
          DType U11 = offset_bottom_data[bottom_index_base + y1*width + x1];
          DType diff_x = (U11*dist_y + U10*(1 - dist_y) - U01*dist_y - U00*(1 - dist_y))
            *trans_std*diff_val;
          diff_x *= roi_width;
          DType diff_y = (U11*dist_x + U01*(1 - dist_x) - U10*dist_x - U00*(1 - dist_x))
            *trans_std*diff_val;
          diff_y *= roi_height;

          atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2)
                                           * part_size + part_h)
                                           * part_size + part_w, diff_x);
          atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1)
                                           * part_size + part_h)
                                           * part_size + part_w, diff_y);
        }
      }
    }
  }


  template<typename DType>
  inline void DeformablePSROIPoolv2BackwardAcc(const std::vector<Tensor<gpu, 4, DType>> &in_grads,
    const Tensor<gpu, 4, DType> &trans_grad,
    const Tensor<gpu, 4, DType> &out_grad,
    const std::vector<Tensor<gpu, 4, DType>> &datas,
    const Tensor<gpu, 2, DType> &bbox,
    const Tensor<gpu, 4, DType> &trans,
    const Tensor<gpu, 4, DType> &top_count,
    const bool no_trans,
    const nnvm::Tuple<float> spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {
    // LOG(INFO) << "DeformablePSROIPoolBackward";
    const DType *top_diff = out_grad.dptr_;
    
    std::vector<const DType*> _stage_bottom_data;
    for (int i = 0; i < datas.size(); i++)
        _stage_bottom_data.push_back(datas[i].dptr_);
    const DType **stage_bottom_data;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_bottom_data, sizeof(const DType*) * _stage_bottom_data.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_bottom_data, &_stage_bottom_data[0], sizeof(const DType*) * _stage_bottom_data.size(),
                                cudaMemcpyHostToDevice));
    
    const DType *bottom_rois = bbox.dptr_;
    const DType *bottom_trans = no_trans ? NULL : trans.dptr_;
    
    std::vector<DType*> _stage_bottom_data_diff;
    for (int i = 0; i < in_grads.size(); i++)
        _stage_bottom_data_diff.push_back(in_grads[i].dptr_);
    DType **stage_bottom_data_diff;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_bottom_data_diff, sizeof(DType*) * _stage_bottom_data_diff.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_bottom_data_diff, &_stage_bottom_data_diff[0], sizeof(DType*) * _stage_bottom_data_diff.size(),
                                cudaMemcpyHostToDevice));

    DType *bottom_trans_diff = no_trans ? NULL : trans_grad.dptr_;
    const DType *top_count_data = top_count.dptr_;
    const int count = out_grad.shape_.Size();
    const int num_rois = bbox.size(0);
    const int channels = in_grads[0].size(1);
    
    std::vector<int> _stage_height;
    for (int i = 0; i < in_grads.size(); i++)
        _stage_height.push_back(in_grads[i].size(2));
    int *stage_height;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_height, sizeof(int) * _stage_height.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_height, &_stage_height[0], sizeof(int) * _stage_height.size(),
                                cudaMemcpyHostToDevice));

    std::vector<int> _stage_width;
    for (int i = 0; i < in_grads.size(); i++)
        _stage_width.push_back(in_grads[i].size(3));
    int *stage_width;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_width, sizeof(int) * _stage_width.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_width, &_stage_width[0], sizeof(int) * _stage_width.size(),
                                cudaMemcpyHostToDevice));
                                
    const int pooled_height = pooled_size;
    const int pooled_width = pooled_size;
    const int num_classes = no_trans ? 1 : trans_grad.size(1) / 2;
    const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

    std::vector<DType> _stage_spatial_scale;
    for (int i = 0; i < spatial_scale.ndim(); i++)
        _stage_spatial_scale.push_back(spatial_scale[i]);
    DType *stage_spatial_scale;
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMalloc(&stage_spatial_scale, sizeof(DType) * _stage_spatial_scale.size()));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaMemcpy(stage_spatial_scale, &_stage_spatial_scale[0], sizeof(DType) * _stage_spatial_scale.size(),
                                cudaMemcpyHostToDevice));
    
    cudaStream_t stream = Stream<gpu>::GetStream(in_grads[0].stream_);
    DeformablePSROIPoolv2BackwardAccKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
      kBaseThreadNum, 0, stream >> >(
      count, top_diff, top_count_data, num_rois, stage_spatial_scale, channels, stage_height, stage_width,
      pooled_height, pooled_width, output_dim, stage_bottom_data_diff, bottom_trans_diff,
      stage_bottom_data, bottom_rois, bottom_trans, no_trans, trans_std, sample_per_part,
      group_size, part_size, num_classes, channels_each_class);
    DeformablePSROIPOOLING_CUDA_CHECK(cudaPeekAtLastError());
    
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_bottom_data));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_bottom_data_diff));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_height));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_width));
    DeformablePSROIPOOLING_CUDA_CHECK(cudaFree(stage_spatial_scale));
  }

}  // namespace cuda

  template<typename DType>
  inline void DeformablePSROIPoolv2Forward(const Tensor<gpu, 4, DType> &out,
    const std::vector<Tensor<gpu, 4, DType>> &datas,
    const Tensor<gpu, 2, DType> &bbox,
    const Tensor<gpu, 4, DType> &trans,
    const Tensor<gpu, 4, DType> &top_count,
    const bool no_trans,
    const nnvm::Tuple<float> spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {
    cuda::DeformablePSROIPoolv2Forward(out, datas, bbox, trans, top_count, no_trans, spatial_scale,
      output_dim, group_size, pooled_size, part_size, sample_per_part, trans_std);
  }

  template<typename DType>
  inline void DeformablePSROIPoolv2BackwardAcc(const std::vector<Tensor<gpu, 4, DType>> &in_grads,
    const Tensor<gpu, 4, DType> &trans_grad,
    const Tensor<gpu, 4, DType> &out_grad,
    const std::vector<Tensor<gpu, 4, DType>> &datas,
    const Tensor<gpu, 2, DType> &bbox,
    const Tensor<gpu, 4, DType> &trans,
    const Tensor<gpu, 4, DType> &top_count,
    const bool no_trans,
    const nnvm::Tuple<float> spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {
    cuda::DeformablePSROIPoolv2BackwardAcc(in_grads, trans_grad, out_grad, datas, bbox, trans,
      top_count, no_trans, spatial_scale, output_dim, group_size, pooled_size, part_size,
      sample_per_part, trans_std);
  }

}  // namespace mshadow


namespace mxnet {
namespace op {

  template<>
  Operator* CreateOp<gpu>(DeformablePSROIPoolingv2Param param, int dtype) {
    Operator* op = nullptr;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new DeformablePSROIPoolingv2Op<gpu, DType>(param);
    });
    return op;
  }

}  // namespace op
}  // namespace mxnet
