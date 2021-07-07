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
    const auto &runner = NpuOpRunner(
        "Pooling", {*in_x}, {*out},
        {{"mode", pooling_mode},
         {"global_pooling", global_pooling},
         {"window", ksize},
         {"stride", strides},
         {"pad", paddings},
         {"dilation", std::vector<int64_t>({1, 1, 1, 1})},
         {"ceil_mode", static_cast<int64_t>(1)},  // 1: floor, 0: ceil
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
    std::vector<int> ksize_vec(4, 1);
    std::vector<int> strides_vec(4, 1);
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
      ksize_vec[1] = ksize[0];
      ksize_vec[2] = ksize[1];
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
      ksize_vec[2] = ksize[0];
      ksize_vec[3] = ksize[1];
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
    }
    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);

    auto stream = dev_ctx.stream();
    if (pooling_type == "max") {
      const auto &runner =
          NpuOpRunner("MaxPoolV3Grad", {*in_x, *out, *out_grad}, {*in_x_grad},
                      {{"ksize", ksize_vec},
                       {"strides", strides_vec},
                       {"padding_mode", std::string("CALCULATED")},
                       {"pads", paddings},
                       {"data_format", data_format},
                       {"global_pooling", global_pooling},
                       {"ceil_mode", false}});  // 0: floor, 1: ceil
      runner.Run(stream);
    } else if (pooling_type == "avg") {
      // const auto &runner =
      //     NpuOpRunner("AvgPoolV2GradD", {*out_grad}, {*in_x_grad},
      //                 {{"orig_input_shape", framework::vectorize(in_x->dims())},
      //                  {"ksize", ksize_vec},
      //                  {"strides", strides_vec},
      //                  {"padding_mode", std::string("CALCULATED")},
      //                  {"pads", paddings},
      //                  {"data_format", data_format},
      //                  {"global_pooling", global_pooling},
      //                  {"ceil_mode", false},  // 0: floor, 1: ceil
      //                  {"exclusive", exclusive}});
      // runner.Run(stream);

      auto cpu_dev_ctx = platform::CPUDeviceContext(platform::CPUPlace());
      Tensor cpu_in_x, cpu_out, cpu_in_x_grad, cpu_out_grad;
      cpu_in_x.mutable_data<T>(in_x->dims(), cpu_dev_ctx.GetPlace());
      cpu_in_x_grad.mutable_data<T>(in_x_grad->dims(), cpu_dev_ctx.GetPlace());
      cpu_out.mutable_data<T>(out->dims(), cpu_dev_ctx.GetPlace());
      cpu_out_grad.mutable_data<T>(out_grad->dims(), cpu_dev_ctx.GetPlace());

      framework::TensorCopy(*in_x, cpu_dev_ctx.GetPlace(), dev_ctx, &cpu_in_x);
      framework::TensorCopy(*out, cpu_dev_ctx.GetPlace(), dev_ctx, &cpu_out);
      framework::TensorCopy(*out_grad, cpu_dev_ctx.GetPlace(), dev_ctx, &cpu_out_grad);
      dev_ctx.Wait();
      paddle::operators::math::Pool2dGradFunctor<platform::CPUDeviceContext, paddle::operators::math::AvgPoolGrad<T>, T>
          pool2d_backward;
      paddle::operators::math::AvgPoolGrad<T> pool_process;
      pool2d_backward(cpu_dev_ctx, cpu_in_x, cpu_out, cpu_out_grad, ksize, strides,
                      paddings, data_format, exclusive, adaptive,
                      &cpu_in_x_grad, pool_process);
      framework::TensorCopy(cpu_in_x_grad, dev_ctx.GetPlace(), dev_ctx, in_x_grad);
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
