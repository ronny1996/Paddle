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
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fill_constant_op.h"

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/npu_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    const Tensor *in_x = ctx.Input<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string data_format = ctx.Attr<std::string>("data_format");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    bool global_pooling = ctx.Attr<bool>("global_pooling");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";

    auto in_x_dims = in_x->dims();
    framework::DDim data_dims;
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
    }

    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    int64_t pooling_mode = (pooling_type == "max" ? 0 : 1);  // 0: max, 1: avg
    const auto &runner =
        NpuOpRunner("Pooling", {*in_x}, {*out},
                    {{"mode", pooling_mode},
                     {"global_pooling", global_pooling},
                     {"window", ksize},
                     {"stride", strides},
                     {"pad", paddings},
                     {"dilation", std::vector<int64_t>({1, 1, 1, 1})},
                     {"ceil_mode", static_cast<int64_t>(0)},
                     {"data_format", data_format}});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
};

template <typename T>
class NPUPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *out = ctx.Input<Tensor>("Out");
    const Tensor *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *in_x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    in_x_grad->mutable_data<T>(ctx.GetPlace());

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::string data_format = ctx.Attr<std::string>("data_format");
    bool global_pooling = ctx.Attr<bool>("global_pooling");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";

    // update paddings
    auto in_x_dims = in_x->dims();
    framework::DDim data_dims;
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
    }
    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);

    auto stream = dev_ctx.stream();
    if (pooling_type == "max") {
      Tensor argmax_tensor;
      argmax_tensor.mutable_data<T>(in_x->dims(), ctx.GetPlace());

      const auto &runner1 = NpuOpRunner(
          "MaxPoolWithArgmaxV2", {*in_x}, {*out, argmax_tensor},
          {{"ksize", ksize}, {"strides", strides}, {"pads", paddings}});
      runner1.Run(stream);
      const auto &runner2 = NpuOpRunner(
          "MaxPoolGradWithArgmaxV2", {*in_x, *out_grad, argmax_tensor},
          {*in_x_grad},
          {{"ksize", ksize}, {"strides", strides}, {"pads", paddings}});
      runner2.Run(stream);
    } else if (pooling_type == "avg") {
      Tensor input_shape_tensor;
      input_shape_tensor.mutable_data<T>(in_x->dims(), ctx.GetPlace());

      const auto &runner =
          NpuOpRunner("AvgPoolV2Grad", {input_shape_tensor, *out_grad},
                      {*in_x_grad}, {{"ksize", ksize},
                                     {"strides", strides},
                                     {"pads", paddings},
                                     {"global_pooling", global_pooling},
                                     {"ceil_mode", false},
                                     {"exclusive", exclusive},
                                     {"data_format", data_format}});
      runner.Run(stream);
    }
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(pool2d, paddle::operators::NPUPoolOpKernel<float>,
                       paddle::operators::NPUPoolOpKernel<double>);
REGISTER_OP_NPU_KERNEL(pool2d_grad,
                       paddle::operators::NPUPoolGradOpKernel<float>,
                       paddle::operators::NPUPoolGradOpKernel<double>);
