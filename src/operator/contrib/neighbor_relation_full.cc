/*!
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT license [see LICENSE for details]
 * \file neighbor_relation_full.cc
 * \brief
 * \author Han Hu
*/

#include "./neighbor_relation_full-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class NeighborRelationFullOp : public Operator{
 public:
  explicit NeighborRelationFullOp(NeighborRelationFullParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    LOG(FATAL) << "not implemented";
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    LOG(FATAL) << "not implemented";
  }

 private:
  NeighborRelationFullParam param_;
};  // class FeatureSamplingOp

template<>
Operator *CreateOp<cpu>(NeighborRelationFullParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new NeighborRelationFullOp<cpu, DType>(param);
  })
  return op;
}

Operator* NeighborRelationFullProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                        std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(NeighborRelationFullParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_NeighborRelationFull, NeighborRelationFullProp)
.describe("Neighbor Relation")
.add_argument("value", "NDArray-or-Symbol", "Value.")
.add_argument("key", "NDArray-or-Symbol", "Key.")
.add_argument("query", "NDArray-or-Symbol", "Query.")
.add_argument("pos_weight", "NDArray-or-Symbol", "Pos Weight.")
.add_arguments(NeighborRelationFullParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
