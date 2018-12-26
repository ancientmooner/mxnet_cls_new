/*!
 * Copyright (c) 2018 Microsoft
 * \file neighbor_relation-inl.h
 * \brief neighbor_relation Operator
 * \author Han Hu
*/
#ifndef MXNET_OPERATOR_CONTRIB_NEIGHBOR_RELATION_INL_H_
#define MXNET_OPERATOR_CONTRIB_NEIGHBOR_RELATION_INL_H_

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

//neighbor_relation(value_data, key_data, query_data, pos_weight, offset_grids_repeat, scale, num_group=32, kernel_size=7):

namespace neighborRelation {
enum NeighborRelationOpInputs {kValue, kKey, kQuery, kPos};
enum NeighborRelationOpOutputs {kOutput};
enum NeighborRelationForwardResource {kTempSpace};
}  // neighborRelation

struct NeighborRelationParam : public dmlc::Parameter<NeighborRelationParam> {
  
  int num_group;
  int kernel_size;
  int batch_step;
  int dilate;
  int stride;
  float scale;
  float no_define_value;
  int norm_method;
  int sim_method;
  DMLC_DECLARE_PARAMETER(NeighborRelationParam) {
    DMLC_DECLARE_FIELD(num_group).set_default(32)
      .describe("Number of relation groups.");
    DMLC_DECLARE_FIELD(kernel_size).set_default(7)
      .describe("kernel size of relation computation.");
    DMLC_DECLARE_FIELD(batch_step).set_default(32)
      .describe("one time batch relation computation. Must be divided by batch_size");
    DMLC_DECLARE_FIELD(dilate).set_default(1)
      .describe("dilate value");
    DMLC_DECLARE_FIELD(scale).set_default(1.0)
      .describe("scale of relation computation.");
    DMLC_DECLARE_FIELD(no_define_value).set_default(0.0)
      .describe("the value of similairty when no definition is giving.");
    DMLC_DECLARE_FIELD(norm_method).set_default(0)
      .describe("0: softmax; 1: ReLU-Norm");
    DMLC_DECLARE_FIELD(sim_method).set_default(0)
      .describe("0: dot; 1: add");
  }
};

template<typename xpu>
Operator *CreateOp(NeighborRelationParam param, int dtype);

#if DMLC_USE_CXX11
class NeighborRelationProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 4) << "Input:[value, key, query, pos_weight]";
    const TShape &vshape = in_shape->at(neighborRelation::kValue);
    const TShape &kshape = in_shape->at(neighborRelation::kKey);
    const TShape &qshape = in_shape->at(neighborRelation::kQuery);


    //kshape [batch_size, key_channels, height, width]
    //qshape [batch_size, key_channels, height, width]
    //vshape [batch_size, value_channels, height, width]
    //pos_weight [num_group, kernel_size, kernel_size]

    //out_shape vshape

    CHECK_EQ(vshape[0], kshape[0]) << "value and key should have the same batch size";
    CHECK_EQ(vshape[2], kshape[2]) << "value and key should have the same height";
    CHECK_EQ(vshape[3], kshape[3]) << "value and key should have the same width";
    CHECK_EQ(vshape[0], qshape[0]) << "value and query should have the same batch size";
    CHECK_EQ(vshape[2], qshape[2]) << "value and query should have the same height";
    CHECK_EQ(vshape[3], qshape[3]) << "value and query should have the same width";
    
    CHECK_EQ(kshape[1], qshape[1]) << "key and query should have the same channel dim";
    
    const TShape &pshape = in_shape->at(neighborRelation::kPos);
    //CHECK_EQ(pshape[0], param_.num_group) << "pos_weight must be of shape [num_group, kernel_y, kenel_x]";

    //TShape oshape(4);
    //oshape[0] = vshape[0];
    //oshape[1] = vshape[1];
    //oshape[2] = (int) ((vshape[2] - 1) / param_.stride) + 1;
    //oshape[3] = (int) ((vshape[3] - 1) / param_.stride) + 1;
    out_shape->clear();
    out_shape->push_back(vshape);
    return true; 
  }

  OperatorProperty* Copy() const override {
    auto ptr = new NeighborRelationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_NeighborRelation";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return{ ResourceRequest::kTempSpace };
  }
  
  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return{ ResourceRequest::kTempSpace };
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[neighborRelation::kOutput], 
            in_data[neighborRelation::kValue],
            in_data[neighborRelation::kKey],
            in_data[neighborRelation::kQuery],
            in_data[neighborRelation::kPos]};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"value", "key", "query", "pos_weight"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }
  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  NeighborRelationParam param_;
};  // class NeighborRelationProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_FEATURE_SAMPLING_INL_H_
