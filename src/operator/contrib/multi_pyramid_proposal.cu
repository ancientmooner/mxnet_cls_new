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
 * \file multi_pyramid_proposal.cu
 * \brief MultiPyramidProposal Operator
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
#include "./multi_pyramid_proposal-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {
namespace multi_pyramid_proposal {

// scores are (b, 2 * anchor, h, w)
// workspace_proposals are (b, h * w * anchor, 5)
// w defines "x" and h defines "y"
// count should be total anchors numbers, h * w * anchors
template<typename DType>
__global__ void ProposalGridKernel(const int count,
                                   const int num_anchors,
                                   const int height,
                                   const int width,
                                   const float feature_stride,
                                   const DType* scores,
                                   DType* workspace_proposals) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % width;
    int h = (index / num_anchors / width) % height;
    int b = index / num_anchors / width / height;

    workspace_proposals[index * 5 + 0] = workspace_proposals[a * 5 + 0] + w * feature_stride;
    workspace_proposals[index * 5 + 1] = workspace_proposals[a * 5 + 1] + h * feature_stride;
    workspace_proposals[index * 5 + 2] = workspace_proposals[a * 5 + 2] + w * feature_stride;
    workspace_proposals[index * 5 + 3] = workspace_proposals[a * 5 + 3] + h * feature_stride;
    workspace_proposals[index * 5 + 4] =
        scores[((b * (2 * num_anchors) + a + num_anchors) * height + h) * width + w];
  }
}

// boxes are (b, h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (b, h * w * anchor, 5)
// count should be total anchors numbers, b * h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename DType>
__global__ void BBoxPredKernel(const int count,
                               const int num_anchors,
                               const int feat_height,
                               const int feat_width,
                               const float feature_stride,
                               const DType* im_infos,
                               const DType* boxes,
                               const DType* deltas,
                               DType* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = (index / num_anchors / feat_width) % feat_height;
    int b = index / num_anchors / feat_width / feat_height;

    float im_height = float(im_infos[b * 3]);
    float im_width = float(im_infos[b * 3 + 1]);
    int real_height = static_cast<int>(im_height / feature_stride);
    int real_width = static_cast<int>(im_width / feature_stride);

    float width = float(boxes[index * 5 + 2]) - float(boxes[index * 5 + 0]) + 1.0f;
    float height = float(boxes[index * 5 + 3]) - float(boxes[index * 5 + 1]) + 1.0f;
    float ctr_x = float(boxes[index * 5 + 0]) + 0.5f * (width - 1.0f);
    float ctr_y = float(boxes[index * 5 + 1]) + 0.5f * (height - 1.0f);

    int ba = (b * num_anchors + a);
    float dx = float(deltas[((ba * 4) * feat_height + h) * feat_width + w]);
    float dy = float(deltas[((ba * 4 + 1) * feat_height + h) * feat_width + w]);
    float dw = float(deltas[((ba * 4 + 2) * feat_height + h) * feat_width + w]);
    float dh = float(deltas[((ba * 4 + 3) * feat_height + h) * feat_width + w]);

    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;

    float pred_x1 = pred_ctr_x - 0.5f * (pred_w - 1.0f);
    float pred_y1 = pred_ctr_y - 0.5f * (pred_h - 1.0f);
    float pred_x2 = pred_ctr_x + 0.5f * (pred_w - 1.0f);
    float pred_y2 = pred_ctr_y + 0.5f * (pred_h - 1.0f);

    pred_x1 = max(min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = max(min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = max(min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = max(min(pred_y2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// boxes are (b, h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (b, h * w * anchor, 5)
// count should be total anchors numbers, b * h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename DType>
__global__ void IoUPredKernel(const int count,
                              const int num_anchors,
                              const int feat_height,
                              const int feat_width,
                              const float feature_stride,
                              const DType* im_infos,
                              const DType* boxes,
                              const DType* deltas,
                              DType* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = (index / num_anchors / feat_width) % feat_height;
    int b = index / num_anchors / feat_width / feat_height;

    float im_height = float(im_infos[b * 3]);
    float im_width = float(im_infos[b * 3 + 1]);
    int real_height = static_cast<int>(im_height / feature_stride);
    int real_width = static_cast<int>(im_width / feature_stride);

    float x1 = float(boxes[index * 5 + 0]);
    float y1 = float(boxes[index * 5 + 1]);
    float x2 = float(boxes[index * 5 + 2]);
    float y2 = float(boxes[index * 5 + 3]);

    int ba = (b * num_anchors + a);
    float dx1 = float(deltas[((ba * 4) * feat_height + h) * feat_width + w]);
    float dy1 = float(deltas[((ba * 4 + 1) * feat_height + h) * feat_width + w]);
    float dx2 = float(deltas[((ba * 4 + 2) * feat_height + h) * feat_width + w]);
    float dy2 = float(deltas[((ba * 4 + 3) * feat_height + h) * feat_width + w]);

    float pred_x1 = max(min(x1 + dx1, im_width - 1.0f), 0.0f);
    float pred_y1 = max(min(y1 + dy1, im_height - 1.0f), 0.0f);
    float pred_x2 = max(min(x2 + dx2, im_width - 1.0f), 0.0f);
    float pred_y2 = max(min(y2 + dy2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// filter box with stride less than rpn_min_size
// filter: set score to zero
// dets (b, n, 5)
template<typename DType>
__global__ void FilterBoxKernel(const int count,
                                const int count_anchors,
                                const float original_min_size,
                                const DType* im_infos,
                                DType* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int b = index / count_anchors;
    float iw = float(dets[index * 5 + 2]) - float(dets[index * 5 + 0]) + 1.0f;
    float ih = float(dets[index * 5 + 3]) - float(dets[index * 5 + 1]) + 1.0f;
    float min_size = original_min_size * float(im_infos[b * 3 + 2]);
    if (iw < min_size || ih < min_size) {
      dets[index * 5 + 0] -= min_size / 2;
      dets[index * 5 + 1] -= min_size / 2;
      dets[index * 5 + 2] += min_size / 2;
      dets[index * 5 + 3] += min_size / 2;
      dets[index * 5 + 4] = -1.0f;
    }
  }
}

// reorder proposals obtained on different stides of feature maps
// src (batch_size * acc_count * 5); dst (batch_size * acc_count * 5)
template<typename DType>
__global__ void ReorganizedProposalsKernel(const int count,
                                const int batch_ind,
                                const int batch_size,
                                const int start_cnt,
                                const int end_cnt,
                                const DType* src,
                                DType* dst) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x) {
    int src_index = batch_size * start_cnt + batch_ind * (end_cnt - start_cnt) + index;
    int dst_index = start_cnt + index;
    dst[dst_index] = src[src_index];
  }
}

// reorder proposals obtained on different stides of feature maps
// src (batch_size * acc_count * 5); dst (batch_size * acc_count * 5)
template<typename DType>
__global__ void ReorganizedProposalsKernel(const int count,
                                const int batch_ind,
                                const int batch_size,
                                const int acc_cnt1,
                                const int acc_cnt2,
                                const int acc_cnt3,
                                const int acc_cnt4,
                                const int acc_cnt5,
                                const int acc_cnt6,
                                const int acc_cnt7,
                                const int acc_cnt8,
                                const int acc_cnt9,
                                const int acc_cnt10,
                                const int acc_cnt11,
                                const int acc_cnt12,
                                const int acc_cnt13,
                                const int acc_cnt14,
                                const int acc_cnt15,
                                const DType* src,
                                DType* dst) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int src_index = 0;
    if (index < acc_cnt1) {
        src_index = batch_ind * acc_cnt1 + index; 
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt2){
        src_index = batch_size * acc_cnt1 + batch_ind * (acc_cnt2 - acc_cnt1) + index - acc_cnt1;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt3) {
        src_index = batch_size * acc_cnt2 + batch_ind * (acc_cnt3 - acc_cnt2) + index - acc_cnt2;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt4) {
        src_index = batch_size * acc_cnt3 + batch_ind * (acc_cnt4 - acc_cnt3) + index - acc_cnt3;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt5) {
        src_index = batch_size * acc_cnt4 + batch_ind * (acc_cnt5 - acc_cnt4) + index - acc_cnt4;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt6) {
        src_index = batch_size * acc_cnt5 + batch_ind * (acc_cnt6 - acc_cnt5) + index - acc_cnt5;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt7) {
        src_index = batch_size * acc_cnt6 + batch_ind * (acc_cnt7 - acc_cnt6) + index - acc_cnt6;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt8) {
        src_index = batch_size * acc_cnt7 + batch_ind * (acc_cnt8 - acc_cnt7) + index - acc_cnt7;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt9) {
        src_index = batch_size * acc_cnt8 + batch_ind * (acc_cnt9 - acc_cnt8) + index - acc_cnt8;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt10) {
        src_index = batch_size * acc_cnt9 + batch_ind * (acc_cnt10 - acc_cnt9) + index - acc_cnt9;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt11) {
        src_index = batch_size * acc_cnt10 + batch_ind * (acc_cnt11 - acc_cnt10) + index - acc_cnt10;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt12) {
        src_index = batch_size * acc_cnt11 + batch_ind * (acc_cnt12 - acc_cnt11) + index - acc_cnt11;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt13) {
        src_index = batch_size * acc_cnt12 + batch_ind * (acc_cnt13 - acc_cnt12) + index - acc_cnt12;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt14) {
        src_index = batch_size * acc_cnt13 + batch_ind * (acc_cnt14 - acc_cnt13) + index - acc_cnt13;
        dst[index] = src[src_index];
    }
    else if (index < acc_cnt15) {
        src_index = batch_size * acc_cnt14 + batch_ind * (acc_cnt15 - acc_cnt14) + index - acc_cnt14;
        dst[index] = src[src_index];
    }
  }
}

// copy score and init order
// dets (n, 5); score (n, ); order (n, )
// count should be n (total anchors or proposals)
template<typename DType>
__global__ void CopyScoreKernel(const int count,
                                const DType* dets,
                                DType* score,
                                int* order) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    score[index] = dets[index * 5 + 4];
    order[index] = index;
  }
}

// reorder proposals according to order and keep the top_n proposals
// prev_dets (n, 5); order (n, ); dets (n, 5)
// count should be output anchor numbers (top_n)
template<typename DType>
__global__ void ReorderProposalsKernel(const int count,
                                       const DType* prev_dets,
                                       const int* order,
                                       DType* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    const int order_i = order[index];
    for (int j = 0; j < 5; j ++) {
      dets[index * 5 + j] = prev_dets[order_i * 5 + j];
    }
  }
}

template<typename DType>
__device__ inline DType devIoU(DType const * const a, DType const * const b) {
  float left = max(float(a[0]), float(b[0])), right = min(float(a[2]), float(b[2]));
  float top = max(float(a[1]), float(b[1])), bottom = min(float(a[3]), float(b[3]));
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (float(a[2]) - float(a[0]) + 1) * (float(a[3]) - float(a[1]) + 1);
  float Sb = (float(b[2]) - float(b[0]) + 1) * (float(b[3]) - float(b[1]) + 1);
  return interS / (Sa + Sb - interS);
}

template<typename DType>
__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const DType *dev_boxes, uint64_t *dev_mask) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ DType block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const DType *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


template<typename DType>
void _nms(const mshadow::Tensor<gpu, 2, DType>& boxes,
          const float nms_overlap_thresh,
          int *keep,
          int *num_out) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int boxes_num = boxes.size(0);
  const int boxes_dim = boxes.size(1);

  DType* boxes_dev = boxes.dptr_;
  uint64_t* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  FRCNN_CUDA_CHECK(cudaMalloc(&mask_dev,
                              boxes_num * col_blocks * sizeof(uint64_t)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());
  std::vector<uint64_t> mask_host(boxes_num * col_blocks);
  FRCNN_CUDA_CHECK(cudaMemcpy(&mask_host[0],
                              mask_dev,
                              sizeof(uint64_t) * boxes_num * col_blocks,
                              cudaMemcpyDeviceToHost));

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[num_to_keep++] = i;
      uint64_t *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  FRCNN_CUDA_CHECK(cudaFree(mask_dev));
}

// copy proposals to output
// dets (top_n, 5); keep (top_n, ); out (top_n, )
// count should be top_n (total anchors or proposals)
template<typename DType>
__global__ void PrepareOutput(const int count,
                              const DType* dets,
                              const int* keep,
                              const int out_size,
                              const int image_index,
                              DType* out,
                              DType* score) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    out[index * 5] = image_index;
    if (index < out_size) {
      int keep_i = keep[index];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    } else {
      int keep_i = keep[index % out_size];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    }
  }
}
}  // namespace multi_pyramid_proposal
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class MultiPyramidProposalGPUOp : public Operator{
 public:
  explicit MultiPyramidProposalGPUOp(MultiPyramidProposalParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda;
    using namespace mshadow::cuda::multi_pyramid_proposal;
    CHECK_EQ(in_data.size(), param_.feature_stride.ndim() * 2 + 1);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[proposal::kOut], kWriteTo);
    /*CHECK_EQ(in_data[proposal::kClsProb].shape_[0], 1)
      << "Sorry, multiple images each device is not implemented.";*/

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 2, DType> im_info = in_data[proposal::kImInfo].get<xpu, 2, DType>(s);

    Tensor<xpu, 2, DType> out = out_data[proposal::kOut].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> out_score = out_data[proposal::kScore].get<xpu, 2, DType>(s);

    int num_images = im_info.size(0);

    // Copy generated anchors to GPU
    DType* workspace_proposals_ptr = NULL;
    
    int acc_count_anchors = 0;
    std::vector<int> acc_count_anchors_vec;
    for (int sind = 0; sind < param_.feature_stride.ndim(); ++sind) {
      Tensor<xpu, 4, DType> scores = in_data[2 * sind + 1].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> bbox_deltas = in_data[2 * sind + 2].get<xpu, 4, DType>(s);

      int num_anchors = scores.size(1) / 2;
      int height = scores.size(2);
      int width = scores.size(3);
      int count_anchors = num_anchors * height * width;  // count of total anchors
      int count = num_images * count_anchors;
      if (0 == sind) {
        // At most 1.5 times of the first stride conv map

        float ratio = 0.0;

        for (int sind2 = 0; sind2 < param_.feature_stride.ndim(); ++sind2){
            ratio += 1.0 / (param_.feature_stride[sind2] * param_.feature_stride[sind2]);
        }
        ratio = ratio * (param_.feature_stride[0] * param_.feature_stride[0]) + 0.5;

        FRCNN_CUDA_CHECK(cudaMalloc(&workspace_proposals_ptr,
                                sizeof(DType) * (int)(num_images * count_anchors * 5 * ratio)));
      }

      // Generate first anchors based on base anchor
      std::vector<DType> base_anchor(4);
      base_anchor[0] = 0.0;
      base_anchor[1] = 0.0;
      base_anchor[2] = param_.feature_stride[sind] - 1.0;
      base_anchor[3] = param_.feature_stride[sind] - 1.0;
      CHECK_EQ(num_anchors, param_.ratios.ndim() * param_.scales.ndim());
      std::vector<DType> anchors;
      std::vector<float> cratios;
      std::vector<float> cscales;

      for (unsigned int i_r = 0; i_r < param_.ratios.ndim(); ++i_r) {
        cratios.push_back(param_.ratios[i_r]);
      }
      
      for (unsigned int i_s = 0; i_s < param_.scales.ndim(); ++i_s) {
        float base_scale = 1.0f;
        if (sind < param_.feat_base_scales.ndim()){
          base_scale = param_.feat_base_scales[sind];
        }
        cscales.push_back(base_scale * param_.scales[i_s]);
      }

      utils::GenerateAnchors(base_anchor,
                           cratios,
                           cscales,
                           &anchors);
      Tensor<xpu, 3, DType> workspace_proposals(workspace_proposals_ptr + acc_count_anchors * num_images * 5,
                                       Shape3(num_images, count_anchors, 5));
      FRCNN_CUDA_CHECK(cudaMemcpy(workspace_proposals.dptr_, &anchors[0],
                                sizeof(DType) * anchors.size(), cudaMemcpyHostToDevice));

      // Copy proposals to a mesh grid
      dim3 dimGrid((count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
      dim3 dimBlock(kMaxThreadsPerBlock);
      CheckLaunchParam(dimGrid, dimBlock, "ProposalGrid");
      ProposalGridKernel<<<dimGrid, dimBlock>>>(
        count, num_anchors, height, width, param_.feature_stride[sind],
        scores.dptr_, workspace_proposals.dptr_);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      // Transform anchors and bbox_deltas into bboxes
      CheckLaunchParam(dimGrid, dimBlock, "BBoxPred");
      if (param_.iou_loss) {
        IoUPredKernel<<<dimGrid, dimBlock>>>(
          count, num_anchors, height, width, param_.feature_stride[sind], im_info.dptr_,
          workspace_proposals.dptr_, bbox_deltas.dptr_, workspace_proposals.dptr_);
      } else {
        BBoxPredKernel<<<dimGrid, dimBlock>>>(
          count, num_anchors, height, width, param_.feature_stride[sind], im_info.dptr_,
          workspace_proposals.dptr_, bbox_deltas.dptr_, workspace_proposals.dptr_);
      }
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      // filter boxes with less than rpn_min_size
      CheckLaunchParam(dimGrid, dimBlock, "FilterBox");
      FilterBoxKernel<<<dimGrid, dimBlock>>>(
        count, count_anchors, param_.rpn_min_size, im_info.dptr_, workspace_proposals.dptr_);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      acc_count_anchors += count_anchors;
      acc_count_anchors_vec.push_back(acc_count_anchors);
    }
    // set to -1 for max
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n
                                                         : acc_count_anchors;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, acc_count_anchors);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    DType* workspace_reorganized_proposals_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&workspace_reorganized_proposals_ptr, sizeof(DType) * acc_count_anchors * 5));
    Tensor<xpu, 1, DType> workspace_reorganized_proposals(workspace_reorganized_proposals_ptr, Shape1(acc_count_anchors * 5));
    // Copy score to a continuous memory
    DType* score_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&score_ptr, sizeof(DType) * acc_count_anchors));
    Tensor<xpu, 1, DType> score(score_ptr, Shape1(acc_count_anchors));
    int* order_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&order_ptr, sizeof(int) * acc_count_anchors));
    Tensor<xpu, 1, int> order(order_ptr, Shape1(acc_count_anchors));
    DType* workspace_ordered_proposals_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&workspace_ordered_proposals_ptr,
        sizeof(DType) * rpn_pre_nms_top_n * 5));
    Tensor<xpu, 2, DType> workspace_ordered_proposals(workspace_ordered_proposals_ptr,
        Shape2(rpn_pre_nms_top_n, 5));

    int* keep;
    FRCNN_CUDA_CHECK(cudaMalloc(&keep, sizeof(int) * rpn_pre_nms_top_n));
      
    Tensor<xpu, 1, DType> workspace_proposals(workspace_proposals_ptr,
                                       Shape1(num_images * acc_count_anchors * 5));

    for (int b = 0; b < num_images; b++) {
      /*
      int acc_cnt1 = 0;
      int acc_cnt2 = 0;
      int acc_cnt3 = 0;
      int acc_cnt4 = 0;
      int acc_cnt5 = 0;
      int acc_cnt6 = 0;
      int acc_cnt7 = 0;
      int acc_cnt8 = 0;
      int acc_cnt9 = 0;
      int acc_cnt10 = 0;
      int acc_cnt11 = 0;
      int acc_cnt12 = 0;
      int acc_cnt13 = 0;
      int acc_cnt14 = 0;
      int acc_cnt15 = 0;

      if (acc_count_anchors_vec.size() >= 1) {
        acc_cnt1 = acc_count_anchors_vec[0] * 5;
      }
      if (acc_count_anchors_vec.size() >= 2) {
        acc_cnt2 = acc_count_anchors_vec[1] * 5;
      }
      else{
        acc_cnt2 = acc_cnt1;
      }
      if (acc_count_anchors_vec.size() >= 3) {
        acc_cnt3 = acc_count_anchors_vec[2] * 5;
      }
      else{
        acc_cnt3 = acc_cnt2;
      }
      if (acc_count_anchors_vec.size() >= 4) {
        acc_cnt4 = acc_count_anchors_vec[3] * 5;
      }
      else{
        acc_cnt4 = acc_cnt3;
      }
      if (acc_count_anchors_vec.size() >= 5) {
        acc_cnt5 = acc_count_anchors_vec[4] * 5;
      }
      else{
        acc_cnt5 = acc_cnt4;
      }
      if (acc_count_anchors_vec.size() >= 6) {
        acc_cnt6 = acc_count_anchors_vec[5] * 5;
      }
      else{
        acc_cnt6 = acc_cnt5;
      }
      if (acc_count_anchors_vec.size() >= 7) {
        acc_cnt7 = acc_count_anchors_vec[6] * 5;
      }
      else{
        acc_cnt7 = acc_cnt6;
      }
      if (acc_count_anchors_vec.size() >= 8) {
        acc_cnt8 = acc_count_anchors_vec[7] * 5;
      }
      else{
        acc_cnt8 = acc_cnt7;
      }
      if (acc_count_anchors_vec.size() >= 9) {
        acc_cnt9 = acc_count_anchors_vec[8] * 5;
      }
      else{
        acc_cnt9 = acc_cnt8;
      }
      if (acc_count_anchors_vec.size() >= 10) {
        acc_cnt10 = acc_count_anchors_vec[9] * 5;
      }
      else{
        acc_cnt10 = acc_cnt9;
      }
      if (acc_count_anchors_vec.size() >= 11) {
        acc_cnt11 = acc_count_anchors_vec[10] * 5;
      }
      else{
        acc_cnt11 = acc_cnt10;
      }
      if (acc_count_anchors_vec.size() >= 12) {
        acc_cnt12 = acc_count_anchors_vec[11] * 5;
      }
      else{
        acc_cnt12 = acc_cnt11;
      }
      if (acc_count_anchors_vec.size() >= 13) {
        acc_cnt13 = acc_count_anchors_vec[12] * 5;
      }
      else{
        acc_cnt13 = acc_cnt12;
      }
      if (acc_count_anchors_vec.size() >= 14) {
        acc_cnt14 = acc_count_anchors_vec[13] * 5;
      }
      else{
        acc_cnt14 = acc_cnt13;
      }
      if (acc_count_anchors_vec.size() >= 15) {
        acc_cnt15 = acc_count_anchors_vec[14] * 5;
      }
      else{
        acc_cnt15 = acc_cnt14;
      }
      */
      dim3 dimGrid;
      dim3 dimBlock;
      for (int sind = 0; sind < param_.feature_stride.ndim(); ++sind) {
        int acc_start = 0;
        int acc_end = acc_count_anchors_vec[sind] * 5;
        if (0 != sind) {
          acc_start = acc_count_anchors_vec[sind - 1] * 5; 
        }
        dimGrid = dim3((acc_end - acc_start + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
        dimBlock = dim3(kMaxThreadsPerBlock);
        std::string name = "ReorganizeProposals";
        name += std::to_string(sind);
        CheckLaunchParam(dimGrid, dimBlock, name.c_str());
        ReorganizedProposalsKernel << <dimGrid, dimBlock >> >(
              acc_end - acc_start, b, num_images, 
              acc_start, acc_end, 
              workspace_proposals.dptr_, 
              workspace_reorganized_proposals.dptr_);
        FRCNN_CUDA_CHECK(cudaPeekAtLastError());
      }
      
      //FRCNN_CUDA_CHECK(cudaPeekAtLastError());
      dimGrid = dim3((acc_count_anchors + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
      dimBlock = dim3(kMaxThreadsPerBlock);
      CheckLaunchParam(dimGrid, dimBlock, "CopyScore");
      CopyScoreKernel << <dimGrid, dimBlock >> >(
          acc_count_anchors, workspace_reorganized_proposals.dptr_,
          score.dptr_, order.dptr_);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      // argsort score, save order
      thrust::stable_sort_by_key(thrust::device,
            score.dptr_,
            score.dptr_ + score.size(0),
            order.dptr_,
            thrust::greater<DType>());
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      // Reorder proposals according to order

      dimGrid.x = (rpn_pre_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      CheckLaunchParam(dimGrid, dimBlock, "ReorderProposals");
      ReorderProposalsKernel << <dimGrid, dimBlock >> >(
            rpn_pre_nms_top_n, workspace_reorganized_proposals.dptr_,
            order.dptr_, workspace_ordered_proposals.dptr_);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      // perform nms
      std::vector<int> _keep(workspace_ordered_proposals.size(0));
      int out_size = 0;
      _nms(workspace_ordered_proposals,
            param_.threshold,
            &_keep[0],
            &out_size);

      // copy nms result to gpu
      FRCNN_CUDA_CHECK(cudaMemcpy(keep, &_keep[0], sizeof(int) * _keep.size(),
            cudaMemcpyHostToDevice));

      // copy results after nms
      dimGrid.x = (rpn_post_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      CheckLaunchParam(dimGrid, dimBlock, "PrepareOutput");
      PrepareOutput << <dimGrid, dimBlock >> >(
            rpn_post_nms_top_n, workspace_ordered_proposals.dptr_, keep, out_size, b,
            out.dptr_ + b * rpn_post_nms_top_n * 5, out_score.dptr_ + b * rpn_post_nms_top_n);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());
    }
    // free temporary memory
    FRCNN_CUDA_CHECK(cudaFree(keep));
    FRCNN_CUDA_CHECK(cudaFree(workspace_ordered_proposals_ptr));
    FRCNN_CUDA_CHECK(cudaFree(workspace_proposals_ptr));
    FRCNN_CUDA_CHECK(cudaFree(workspace_reorganized_proposals_ptr));
    FRCNN_CUDA_CHECK(cudaFree(score_ptr));
    FRCNN_CUDA_CHECK(cudaFree(order_ptr));
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
    CHECK_EQ(in_grad.size(), 2 * param_.feature_stride.ndim() + 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, DType>(s);
    Assign(ginfo, req[proposal::kImInfo], 0);
    for (int sind = 0; sind < param_.feature_stride.ndim(); ++sind) {
      Tensor<xpu, 4, DType> gscores = in_grad[2 * sind + 1].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> gbbox = in_grad[2 * sind + 2].get<xpu, 4, DType>(s);
      // can not assume the grad would be zero
      Assign(gscores, req[2 * sind + 1], 0);
      Assign(gbbox, req[2 * sind + 2], 0);
    }
  }

 private:
  MultiPyramidProposalParam param_;
};  // class MultiProposalGPUOp

template<>
Operator* CreateOp<gpu>(MultiPyramidProposalParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiPyramidProposalGPUOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
