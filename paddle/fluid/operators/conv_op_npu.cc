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
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fill_constant_op.h"
#include "paddle/fluid/operators/math/padding.h"

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/npu_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUConvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    const Tensor* input = ctx.Input<Tensor>("Input");  // nhwc
    auto* filter = ctx.Input<Tensor>("Filter");        // hwcn, c = i_c / groups
    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());
    const std::vector<int> strides =
        ctx.Attr<std::vector<int>>("strides");  // nhwc, n,c must be set to 1
    std::vector<int> paddings =
        ctx.Attr<std::vector<int>>("paddings");  // t b l r
    std::vector<int> dilations =
        ctx.Attr<std::vector<int>>("dilations");  // nhwc, n,c must be set to 1
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    const bool channel_last = data_format == "NHWC";

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
      filter_data_dims =
          framework::slice_ddim(filter_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
      filter_data_dims = framework::slice_ddim(filter_dims, 2, in_dims.size());
    }

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int> strides_vec(4, 1);
    std::vector<int> dilations_vec(4, 1);

    if (channel_last) {
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      dilations_vec[1] = dilations[0];
      dilations_vec[2] = dilations[1];
    } else {
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
      dilations_vec[2] = dilations[0];
      dilations_vec[3] = dilations[1];
    }

    const auto& runner = NpuOpRunner("Conv2D", {*input, *filter}, {*output},
                                     {{"strides", strides_vec},
                                      {"pads", paddings},
                                      {"dilations", dilations_vec},
                                      {"groups", groups}});

    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
};

template <typename T>
class NPUConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();

    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
    }

    const std::vector<int> strides =
        ctx.Attr<std::vector<int>>("strides");  // nhwc, n,c must be set to 1
    std::vector<int> paddings =
        ctx.Attr<std::vector<int>>("paddings");  // t b l r
    std::vector<int> dilations =
        ctx.Attr<std::vector<int>>("dilations");  // nhwc, n,c must be set to 1
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    const bool channel_last = data_format == "NHWC";

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
      filter_data_dims =
          framework::slice_ddim(filter_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
      filter_data_dims = framework::slice_ddim(filter_dims, 2, in_dims.size());
    }

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int> strides_vec(4, 1);
    std::vector<int> dilations_vec(4, 1);

    if (channel_last) {
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      dilations_vec[1] = dilations[0];
      dilations_vec[2] = dilations[1];
    } else {
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
      dilations_vec[2] = dilations[0];
      dilations_vec[3] = dilations[1];
    }

    std::string data_format_str = "NCHW";
    if (channel_last) {
      data_format_str = "NHWC";
    }

    if (filter_grad) {
      std::vector<int> filter_shape_vec =
          framework::vectorize<int>(filter->dims());
      const auto& runner =
          NpuOpRunner("Conv2DBackpropFilterD", {*input, *output_grad},
                      {*filter_grad}, {{"filter_size", filter_shape_vec},
                                       {"strides", strides_vec},
                                       {"pads", paddings},
                                       {"dilations", dilations_vec},
                                       {"groups", groups},
                                       {"data_format", data_format_str}});
      auto stream = dev_ctx.stream();
      runner.Run(stream);
    }
    if (input_grad) {
      std::vector<int> input_shape_vec =
          framework::vectorize<int>(input->dims());
      const auto& runner =
          NpuOpRunner("Conv2DBackpropInputD", {*filter, *output_grad},
                      {*input_grad}, {{"input_size", input_shape_vec},
                                      {"strides", strides_vec},
                                      {"pads", paddings},
                                      {"dilations", dilations_vec},
                                      {"groups", groups},
                                      {"data_format", data_format_str}});
      auto stream = dev_ctx.stream();
      runner.Run(stream);
    }
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(conv2d, paddle::operators::NPUConvOpKernel<float>,
                       paddle::operators::NPUConvOpKernel<double>);
REGISTER_OP_NPU_KERNEL(conv2d_grad,
                       paddle::operators::NPUConvGradOpKernel<float>,
                       paddle::operators::NPUConvGradOpKernel<double>);
