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
 * \file downsample_v2-inl.h
 * \brief DownsampleV2 Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Xizhou Zhu, Han Hu
*/
#ifndef MXNET_OPERATOR_CONTRIB_DOWNSAMPLE_V2_INL_H_
#define MXNET_OPERATOR_CONTRIB_DOWNSAMPLE_V2_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"


namespace mxnet {
namespace op {

namespace downsample_v2 {
enum DownsampleV2OpInputs {kData, kKernel};
enum DownsampleV2OpOutputs {kOutput};
enum DownsampleV2ForwardResource {kTempResource};
}  // downsample_v2

struct DownsampleV2Param : public dmlc::Parameter<DownsampleV2Param> {
  bool backward_kernel;
  int rescale;
  int dilate;
  DMLC_DECLARE_PARAMETER(DownsampleV2Param) {
    DMLC_DECLARE_FIELD(backward_kernel).set_default(false)
    .describe("Whether backward to kernel");
    DMLC_DECLARE_FIELD(rescale).set_default(2)
    .describe("DownsampleV2 scale");
    DMLC_DECLARE_FIELD(dilate).set_default(1)
    .describe("DownsampleV2 dilate");
  }
};

template<typename xpu>
Operator *CreateOp(DownsampleV2Param param);

#if DMLC_USE_CXX11
class DownsampleV2Prop : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, kernel]";
    const TShape &dshape = in_shape->at(downsample_v2::kData);
    if (dshape.ndim() == 0) return false;
    const TShape &kshape = in_shape->at(downsample_v2::kKernel);
    CHECK_EQ(kshape.ndim(), 2U);
    CHECK_EQ(kshape[0], kshape[1]);
    CHECK_EQ(kshape[0] / 2 * 2 + 1, kshape[0]) << "Only support odd kernel dim";;
    
    Shape<4> output_shape = Shape4(dshape[0], dshape[1], (dshape[2]-1)/param_.rescale+1, (dshape[3]-1)/param_.rescale+1);
    out_shape->clear();
    out_shape->push_back(output_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DownsampleV2Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_DownsampleV2";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.backward_kernel)
        return{ out_grad[downsample_v2::kOutput], in_data[downsample_v2::kData], in_data[downsample_v2::kKernel] };
    else
        return{ out_grad[downsample_v2::kOutput], in_data[downsample_v2::kKernel] };
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "kernel"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  DownsampleV2Param param_;
};  // class DownsampleV2Prop

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_DOWNSAMPLE_V2_INL_H_
