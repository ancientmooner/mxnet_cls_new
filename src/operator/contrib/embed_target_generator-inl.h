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
 * \file embed_target_generator-inl.h
 * \brief EmbedTargetGenerator Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Xizhou Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_EMBED_TARGET_GENERATOR_INL_H_
#define MXNET_OPERATOR_CONTRIB_EMBED_TARGET_GENERATOR_INL_H_

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

namespace embedTargetGenerator {
enum EmbedTargetGeneratorOpInputs {kMask};
enum EmbedTargetGeneratorOpOutputs {kOutput, kInst};
enum EmbedTargetGeneratorForwardResource {kTempResource};
}  // embedTargetGenerator

struct EmbedTargetGeneratorParam : public dmlc::Parameter<EmbedTargetGeneratorParam> {
  int max_displacement;
  int stride2;
  bool compressed_mask;
    
  DMLC_DECLARE_PARAMETER(EmbedTargetGeneratorParam) {
    DMLC_DECLARE_FIELD(max_displacement).set_default(1).describe("max_displacement");
    DMLC_DECLARE_FIELD(stride2).set_default(1).describe("stride2");
    DMLC_DECLARE_FIELD(compressed_mask).set_default(false).describe("compressed_mask");
  }
};

template<typename xpu>
Operator *CreateOp(EmbedTargetGeneratorParam param);

#if DMLC_USE_CXX11
class EmbedTargetGeneratorProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[mask]";
    const TShape &dshape = in_shape->at(embedTargetGenerator::kMask);
    if (dshape.ndim() == 0) return false;
 
    int neighborhood_grid_radius = param_.max_displacement / param_.stride2;
    int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
    int top_channels = neighborhood_grid_width * neighborhood_grid_width;
 
    out_shape->clear();
    out_shape->push_back(Shape4(1, top_channels, dshape[2], dshape[3]));
    if (param_.compressed_mask)
        out_shape->push_back(Shape4(1, top_channels, dshape[2], dshape[3]));
    return true;    
  }

  OperatorProperty* Copy() const override {
    auto ptr = new EmbedTargetGeneratorProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_EmbedTargetGenerator";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {};
  }
  
  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumVisibleOutputs() const override {
    if (param_.compressed_mask)
        return 2;
    else
        return 1;
  }

  int NumOutputs() const override {
    if (param_.compressed_mask)
        return 2;
    else
        return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"mask"};
  }

  std::vector<std::string> ListOutputs() const override {
    if (param_.compressed_mask)
        return {"output", "inst"};
    else
        return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  EmbedTargetGeneratorParam param_;
};  // class EmbedTargetGeneratorProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_EMBED_TARGET_GENERATOR_INL_H_