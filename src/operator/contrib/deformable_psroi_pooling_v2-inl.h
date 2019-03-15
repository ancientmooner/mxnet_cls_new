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
* \file deformable_psroi_pooling-inl.h
* \brief deformable psroi pooling operator and symbol
* \author Yi Li, Guodong Zhang, Jifeng Dai
*/
#ifndef MXNET_OPERATOR_CONTRIB_DEFORMABLE_PSROI_POOLING_V2_INL_H_
#define MXNET_OPERATOR_CONTRIB_DEFORMABLE_PSROI_POOLING_V2_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"


namespace mxnet {
namespace op {

  // Declare enumeration of input order to make code more intuitive.
  // These enums are only visible within this header
namespace deformablepsroipoolv2 {
  enum DeformablePSROIPoolingv2OpInputs { kData, kBox, kTrans };
  enum DeformablePSROIPoolingv2OpOutputs { kOut, kTopCount };
}  // deformablepsroipoolv2

struct DeformablePSROIPoolingv2Param : public dmlc::Parameter<DeformablePSROIPoolingv2Param> {
  nnvm::Tuple<float> spatial_scale;
  int output_dim;
  int group_size;
  int pooled_size;
  int part_size;
  int sample_per_part;
  float trans_std;
  bool no_trans;
  int num_args;
  int num_stage;
  DMLC_DECLARE_PARAMETER(DeformablePSROIPoolingv2Param) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
      .describe("Number of inputs to be roi_pooled.");
    float tmp[] = {0.0625};
    DMLC_DECLARE_FIELD(spatial_scale).set_default(nnvm::Tuple<float>(tmp, tmp + 1))
      .describe("Ratio of input feature map height (or w) to raw image height (or w). "
        "Equals the reciprocal of total stride in convolutional layers");
    DMLC_DECLARE_FIELD(output_dim).describe("fix output dim");
    DMLC_DECLARE_FIELD(group_size).describe("fix group size");
    DMLC_DECLARE_FIELD(pooled_size).describe("fix pooled size");
    DMLC_DECLARE_FIELD(part_size).set_default(0).describe("fix part size");
    DMLC_DECLARE_FIELD(sample_per_part).set_default(1).describe("fix samples per part");
    DMLC_DECLARE_FIELD(trans_std).set_default(0.0).set_range(0.0, 1.0)
      .describe("fix transition std");
    DMLC_DECLARE_FIELD(no_trans).set_default(false)
      .describe("Whether to disable trans parameter.");
    //DMLC_DECLARE_FIELD(num_stage).set_range(1, 10000)
    //  .describe("num stage");
  }
};

template<typename xpu, typename DType>
class DeformablePSROIPoolingv2Op : public Operator {
 public:
  explicit DeformablePSROIPoolingv2Op(DeformablePSROIPoolingv2Param p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
    const std::vector<TBlob> &in_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &out_data,
    const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t in_expected = param_.num_stage + (param_.no_trans? 1 : 2);
    size_t out_expected = 2;
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(out_data[deformablepsroipoolv2::kOut].shape_[0],
             in_data[param_.num_stage - 1 + deformablepsroipoolv2::kBox].shape_[0]);
    CHECK_EQ(out_data[deformablepsroipoolv2::kTopCount].shape_[0],
             in_data[param_.num_stage - 1 + deformablepsroipoolv2::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    std::vector<Tensor<xpu, 4, DType>> datas;
    for (int i = 0; i < param_.num_stage; ++i) {
        Tensor<xpu, 4, DType> data = in_data[i].get<xpu, 4, DType>(s);
        CHECK_EQ(data.CheckContiguous(), true);
        datas.push_back(data);
    }
    Tensor<xpu, 2, DType> bbox = in_data[param_.num_stage - 1 + deformablepsroipoolv2::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[deformablepsroipoolv2::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> top_count = out_data[deformablepsroipoolv2::kTopCount]
                                        .get<xpu, 4, DType>(s);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(top_count.CheckContiguous(), true);
    out = -FLT_MAX;
    top_count = 0.0f;

    Tensor<xpu, 4, DType> trans;
    if (!param_.no_trans) {
      trans = in_data[param_.num_stage - 1 + deformablepsroipoolv2::kTrans].get<xpu, 4, DType>(s);
    }
    DeformablePSROIPoolv2Forward(out, datas, bbox, trans, top_count, param_.no_trans,
      param_.spatial_scale, param_.output_dim, param_.group_size, param_.pooled_size,
      param_.part_size, param_.sample_per_part, param_.trans_std);
  }

  virtual void Backward(const OpContext &ctx,
    const std::vector<TBlob> &out_grad,
    const std::vector<TBlob> &in_data,
    const std::vector<TBlob> &out_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &in_grad,
    const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t in_expected = param_.num_stage + (param_.no_trans? 1 : 2);
    size_t out_expected = 2;
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(out_grad[deformablepsroipoolv2::kOut].shape_[0],
             in_data[param_.num_stage - 1 + deformablepsroipoolv2::kBox].shape_[0]);
    CHECK_EQ(out_data[deformablepsroipoolv2::kTopCount].shape_[0],
             in_data[param_.num_stage - 1 + deformablepsroipoolv2::kBox].shape_[0]);
    for (int i = 0; i < param_.num_stage; ++i) {
      CHECK_NE(req[i], kWriteInplace) <<
      "DeformablePSROIPoolingv2: Backward doesn't support kWriteInplace.";
    }
    CHECK_NE(req[param_.num_stage - 1 + deformablepsroipoolv2::kBox], kWriteInplace) <<
      "DeformablePSROIPoolingv2: Backward doesn't support kWriteInplace.";
    // CHECK_NE(req[param_.num_stage - 1 + deformablepsroipoolv2::kTrans], kWriteInplace) <<
    //  "DeformablePSROIPoolingv2: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[deformablepsroipoolv2::kOut].get<xpu, 4, DType>(s);
    std::vector<Tensor<xpu, 4, DType>> datas;
    for (int i = 0; i < param_.num_stage; ++i) {
        Tensor<xpu, 4, DType> data = in_data[i].get<xpu, 4, DType>(s);
        CHECK_EQ(data.CheckContiguous(), true);
        datas.push_back(data);
    }
    Tensor<xpu, 2, DType> bbox = in_data[param_.num_stage - 1 + deformablepsroipoolv2::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> top_count = out_data[deformablepsroipoolv2::kTopCount]
                                        .get<xpu, 4, DType>(s);                                    
    std::vector<Tensor<xpu, 4, DType>> grad_ins;
    for (int i = 0; i < param_.num_stage; ++i) {
        Tensor<xpu, 4, DType> grad_in = in_grad[i].get<xpu, 4, DType>(s);
        CHECK_EQ(grad_in.CheckContiguous(), true);
        Assign(grad_in, req[i], 0);
        grad_ins.push_back(grad_in);
    }
    Tensor<xpu, 2, DType> grad_roi = in_grad[param_.num_stage - 1 + deformablepsroipoolv2::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> grad_trans;
    Tensor<xpu, 4, DType> trans;
    if (!param_.no_trans) {
      CHECK_EQ(in_grad.size(), in_expected);
      trans = in_data[param_.num_stage - 1 + deformablepsroipoolv2::kTrans].get<xpu, 4, DType>(s);
      grad_trans = in_grad[param_.num_stage - 1 + deformablepsroipoolv2::kTrans].get<xpu, 4, DType>(s);
    }

    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(top_count.CheckContiguous(), true);

    
    if (!param_.no_trans) {
      Assign(grad_trans, req[param_.num_stage - 1 + deformablepsroipoolv2::kTrans], 0);
    }
    DeformablePSROIPoolv2BackwardAcc(grad_ins, grad_trans, grad_out, datas, bbox, trans,
      top_count, param_.no_trans, param_.spatial_scale, param_.output_dim, param_.group_size,
      param_.pooled_size, param_.part_size, param_.sample_per_part, param_.trans_std);
    Assign(grad_roi, req[param_.num_stage - 1 + deformablepsroipoolv2::kBox], 0);
  }

 private:
  DeformablePSROIPoolingv2Param param_;
};  // class DeformablePSROIPoolingv2Op

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(DeformablePSROIPoolingv2Param param, int dtype);

#if DMLC_USE_CXX11
class DeformablePSROIPoolingv2Prop : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (param_.no_trans) {
      std::vector<std::string> ret;
      for (int i = 0; i < param_.num_stage; ++i) {
        ret.push_back(std::string("data") + std::to_string(i));
      }
      ret.push_back("rois");
      return ret;
    } else {
      std::vector<std::string> ret;
      for (int i = 0; i < param_.num_stage; ++i) {
        ret.push_back(std::string("data") + std::to_string(i));
      }
      ret.push_back("rois");
      ret.push_back("trans");
      return ret;
    }
  }

  std::vector<std::string> ListOutputs() const override {
    return{ "output", "top_count" };
  }

  int NumOutputs() const override {
    return 2;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
    if (param_.part_size == 0) {
      param_.part_size = param_.pooled_size;
    }
    printf("dpool num_args: %d\n", param_.num_args);
    if (param_.no_trans) {
      param_.num_stage = param_.num_args - 1;
    } else {
      param_.num_stage = param_.num_args - 2; 
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (param_.no_trans) {
      CHECK_EQ(in_shape->size(), param_.num_stage + 1) << "Input:[data, rois]";
    } else {
      CHECK_EQ(in_shape->size(), param_.num_stage + 2) << "Input:[data, rois, trans]";
      // trans: [num_rois, 2, pooled_h, pooled_w]
      TShape tshape = in_shape->at(param_.num_stage - 1 + deformablepsroipoolv2::kTrans);
      CHECK_EQ(tshape.ndim(), 4) << "trans should be a 4D tensor of shape";
    }

    // data: [batch_size, c, h, w]
    for (int i = 0; i < param_.num_stage; ++i) {
      TShape dshape = in_shape->at(i);
      CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";
    }

    // bbox: [num_rois, 6]
    TShape bshape = in_shape->at(param_.num_stage - 1 + deformablepsroipoolv2::kBox);
    CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 6]";
    CHECK_EQ(bshape[1], 6) << "bbox should be a 2D tensor of shape [batch, 6]";

    // out: [num_rois, c, pooled_h, pooled_w]
    // top_count: [num_rois, c, pooled_h, pooled_w]
    out_shape->clear();
    out_shape->push_back(
      Shape4(bshape[0], param_.output_dim, param_.pooled_size, param_.pooled_size));
    out_shape->push_back(
      Shape4(bshape[0], param_.output_dim, param_.pooled_size, param_.pooled_size));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
    std::vector<int> *out_type,
    std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 2);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    DeformablePSROIPoolingv2Prop* deformable_psroi_pooling_sym = new DeformablePSROIPoolingv2Prop();
    deformable_psroi_pooling_sym->param_ = this->param_;
    return deformable_psroi_pooling_sym;
  }

  std::string TypeString() const override {
    return "_contrib_DeformablePSROIPoolingv2";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.no_trans) {
      std::vector<int> deps;
      for (int i = 0; i < param_.num_stage; ++i) {
        deps.push_back(in_data[i]);
      }
      deps.push_back(in_data[param_.num_stage - 1 + deformablepsroipoolv2::kBox]);
      deps.push_back(out_grad[deformablepsroipoolv2::kOut]);
      deps.push_back(out_data[deformablepsroipoolv2::kTopCount]);
      return deps;
    } else {
      std::vector<int> deps;
      for (int i = 0; i < param_.num_stage; ++i) {
        deps.push_back(in_data[i]);
      }
      deps.push_back(in_data[param_.num_stage - 1 + deformablepsroipoolv2::kBox]);
      deps.push_back(in_data[param_.num_stage - 1 + deformablepsroipoolv2::kTrans]);
      deps.push_back(out_grad[deformablepsroipoolv2::kOut]);
      deps.push_back(out_data[deformablepsroipoolv2::kTopCount]);
      return deps;
    }
  }


  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const override;


 private:
  DeformablePSROIPoolingv2Param param_;
};  // class DeformablePSROIPoolingv2Prop
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_DEFORMABLE_PSROI_POOLING_V2_INL_H_
