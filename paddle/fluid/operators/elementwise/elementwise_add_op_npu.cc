/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
void NpuBroadcastTo(const framework::ExecutionContext& ctx, const Tensor* dst,
                    const Tensor* src, int axis, Tensor* transformed_src) {
  // dst.dims().size() > src.dims().size();
  framework::DDim dst_dims = dst->dims();
  framework::DDim src_dims = src->dims();
  framework::DDim trim_src_dims = trim_trailing_singular_dims(src_dims);
  axis = (axis == -1 ? dst_dims.size() - src_dims.size() : axis);
  auto tile_axis = axis + trim_src_dims.size();

  auto stream =
      ctx.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();

  auto expand_to_dims = framework::slice_ddim(dst_dims, 0, tile_axis);
  auto expand_to_dims_vec = framework::vectorize<int>(expand_to_dims);

  Tensor tmp_src;
  tmp_src.ShareDataWith(*src);
  tmp_src.Resize(trim_src_dims);

  Tensor tmp_tensor;
  tmp_tensor.mutable_data<T>(expand_to_dims, ctx.GetPlace());
  const auto& expand_runner = NpuOpRunner("ExpandD", {tmp_src}, {tmp_tensor},
                                          {{"shape", expand_to_dims_vec}});
  expand_runner.Run(stream);

  Tensor tmp_tensor_2;
  if (tile_axis < dst_dims.size()) {
    expand_to_dims_vec.push_back(1);
    tmp_tensor.Resize(framework::make_ddim(expand_to_dims_vec));
    auto tiles_num = framework::product(
        framework::slice_ddim(dst_dims, tile_axis, dst_dims.size()));
    tmp_tensor_2.mutable_data<T>(dst_dims, ctx.GetPlace());
    const auto& tile_runner =
        NpuOpRunner("TileWithAxis", {tmp_tensor}, {tmp_tensor_2},
                    {{"axis", static_cast<int64_t>(tile_axis)},
                     {"tiles", static_cast<int64_t>(tiles_num)}});
    tile_runner.Run(stream);
  } else {
    tmp_tensor_2.ShareDataWith(tmp_tensor);
  }
  framework::TensorCopy(tmp_tensor_2, ctx.GetPlace(), transformed_src);
}

template <typename T>
class ElementwiseAddNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    Tensor transformed_x, transformed_y;
    if (x->dims() != y->dims()) {
      int axis = ctx.Attr<int>("axis");
      if (x->dims().size() >= y->dims().size()) {
        transformed_x.ShareDataWith(*x);
        transformed_y.mutable_data<T>(x->dims(), ctx.GetPlace());
        NpuBroadcastTo<T>(ctx, x, y, axis, &transformed_y);
      } else {
        transformed_y.ShareDataWith(*y);
        transformed_x.mutable_data<T>(y->dims(), ctx.GetPlace());
        NpuBroadcastTo<T>(ctx, y, x, axis, &transformed_x);
      }
    } else {
      transformed_x.ShareDataWith(*x);
      transformed_y.ShareDataWith(*y);
    }
    const auto& runner =
        NpuOpRunner("Add", {transformed_x, transformed_y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename T>
class ElementwiseAddGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // NOTE(zhiqiu): It seems Ascend Sub follow the broadcast sematics with
    // default axis=-1?
    // So, the sub_grad should do reduce if needed.
    // For example, the shape of each variable in elementwise_sub:
    // x, dx: [2, 3, 5]
    // y, dy: [1, 5]
    // out, dout: [2, 3, 5]
    // Then, out = x - y  =>  dx = dout, dy = -dout
    // And, the shape of dy can be computed by two stages reduce,
    // 1. [2, 3, 5] => [3, 5], ReduceSumD on axis = 0, keep_dims = false.
    // 2. [3, 5] => [1, 5], ReduceSumD on axis = 0, keep_dims = true.

    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      // For dx
      // stage 1
      auto reduce_ndim = dout->dims().size() - dx->dims().size();
      std::vector<int> axes;
      for (auto i = 0; i < reduce_ndim; ++i) {
        axes.push_back(i);
      }
      Tensor* tmp_dout = const_cast<Tensor*>(dout);
      Tensor reduced_dout(dx->type());
      if (axes.size() != 0) {
        std::vector<int64_t> reduced_dout_dims;
        for (auto i = reduce_ndim; i < dout->dims().size(); ++i) {
          reduced_dout_dims.push_back(dout->dims()[i]);
        }
        reduced_dout.Resize(framework::make_ddim(reduced_dout_dims));
        reduced_dout.mutable_data<T>(ctx.GetPlace());
        const auto& runner =
            NpuOpRunner("ReduceSumD", {*dout}, {reduced_dout},
                        {{"axes", axes}, {"keep_dims", false}});
        runner.Run(stream);
        tmp_dout = &reduced_dout;
      }

      // stage 2
      axes.clear();
      for (auto i = 0; i < dx->dims().size(); ++i) {
        if (dx->dims()[i] == 1) {
          axes.push_back(i);
        }
      }
      if (axes.size() != 0) {
        const auto& runner = NpuOpRunner("ReduceSumD", {*tmp_dout}, {*dx},
                                         {{"axes", axes}, {"keep_dims", true}});
        runner.Run(stream);
      } else {
        framework::TensorCopy(
            *tmp_dout, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), dx);
      }
    }

    if (dy) {
      // For dy
      // stage 1
      auto reduce_ndim = dout->dims().size() - dy->dims().size();
      std::vector<int> axes;
      for (auto i = 0; i < reduce_ndim; ++i) {
        axes.push_back(i);
      }
      Tensor* tmp_dout = const_cast<Tensor*>(dout);
      Tensor reduced_dout(dout->type());
      if (axes.size() != 0) {
        std::vector<int64_t> reduced_dout_dims;
        for (auto i = reduce_ndim; i < dout->dims().size(); ++i) {
          reduced_dout_dims.push_back(dout->dims()[i]);
        }
        reduced_dout.Resize(framework::make_ddim(reduced_dout_dims));
        reduced_dout.mutable_data<T>(ctx.GetPlace());
        const auto& runner =
            NpuOpRunner("ReduceSumD", {*dout}, {reduced_dout},
                        {{"axes", axes}, {"keep_dims", false}});
        runner.Run(stream);
        tmp_dout = &reduced_dout;
      }

      // stage 2
      axes.clear();
      for (auto i = 0; i < dy->dims().size(); ++i) {
        if (dy->dims()[i] == 1) {
          axes.push_back(i);
        }
      }
      if (axes.size() != 0) {
        dy->mutable_data<T>(ctx.GetPlace());
        const auto& runner = NpuOpRunner("ReduceSumD", {*tmp_dout}, {*dy},
                                         {{"axes", axes}, {"keep_dims", true}});
        runner.Run(stream);
      } else {
        framework::TensorCopy(
            *tmp_dout, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), dy);
      }
    }
  }
};

template <typename T>
class ElementwiseAddGradWithAxisNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(x->dims().size() - y->dims().size()) : axis);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      std::vector<int64_t> reduce_axes;
      if (dx->dims() != dout->dims()) {
        framework::DDim trim_dx_dims = trim_trailing_singular_dims(dx->dims());
        for (int64_t ax = 0; ax < dout->dims().size(); ax++) {
          if (ax < axis || ax >= trim_dx_dims.size() + axis) {
            reduce_axes.push_back(ax);
          }
        }
        // dont need care the axis with dim 1
        if (!reduce_axes.empty()) {
          const auto& runner =
              NpuOpRunner("ReduceSumD", {*dout}, {*dx},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        framework::TensorCopy(
            *dout, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), dx);
      }
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      std::vector<int64_t> reduce_axes;
      if (dy->dims() != dout->dims()) {
        framework::DDim trim_dy_dims = trim_trailing_singular_dims(dy->dims());
        for (int64_t ax = 0; ax < dout->dims().size(); ax++) {
          if (ax < axis || ax >= trim_dy_dims.size() + axis) {
            reduce_axes.push_back(ax);
          }
        }
        // dont care the axis with dim 1
        if (!reduce_axes.empty()) {
          const auto& runner =
              NpuOpRunner("ReduceSumD", {*dout}, {*dy},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        framework::TensorCopy(
            *dout, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), dy);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(elementwise_add, ops::ElementwiseAddNPUKernel<float>,
                       ops::ElementwiseAddNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(elementwise_add_grad,
                       ops::ElementwiseAddGradWithAxisNPUKernel<float>,
                       ops::ElementwiseAddGradWithAxisNPUKernel<plat::float16>);
