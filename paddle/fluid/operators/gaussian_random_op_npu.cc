/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fill_constant_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/npu_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUGaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    auto* out = ctx.Output<framework::Tensor>("Out");

    float mean = ctx.Attr<float>("mean");
    float std = ctx.Attr<float>("std");
    auto seed = ctx.Attr<int>("seed");

    auto shape = GetShape(ctx);
    out->Resize(shape);
    out->mutable_data<T>(ctx.GetPlace());
    framework::Tensor out_shape;
    out_shape.mutable_data<int>(framework::make_ddim({out->dims().size()}),
                                out->place());
    framework::TensorFromVector<int>(framework::vectorize<int>(out->dims()),
                                     dev_ctx, &out_shape);

    auto dtype = ConvertToNpuDtype(out->type());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& standard_runner = NpuOpRunner(
        "StandardNormal", {out_shape}, {*out},
        {{"dtype", dtype}, {"seed", seed}, {"seed2", static_cast<int>(0)}});
    standard_runner.Run(stream);

    const auto& muls_runner =
        NpuOpRunner("Muls", {*out}, {*out}, {{"value", std}});
    muls_runner.Run(stream);

    const auto& adds_runner =
        NpuOpRunner("Adds", {*out}, {*out}, {{"value", mean}});
    adds_runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(gaussian_random,
                       paddle::operators::NPUGaussianRandomKernel<float>,
                       paddle::operators::NPUGaussianRandomKernel<double>);
