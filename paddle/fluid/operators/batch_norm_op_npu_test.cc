// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(batch_norm);
USE_OP_DEVICE_KERNEL(batch_norm, NPU);

namespace paddle {
namespace operators {

template <typename T>
void TestBatchNormOp(const platform::DeviceContext& ctx) {
  auto place = ctx.GetPlace();
  framework::OpDesc desc;
  framework::Scope scope;

  desc.SetType("batch_norm");
  desc.SetAttr("is_test", false);
  desc.SetAttr("use_global_stats", false);
  desc.SetAttr("data_layout", "NCHW");

  desc.SetInput("X", {"X"});
  framework::DDim x_tensor_dims({1, 1, 2, 2});
  std::vector<float> x_tensor_data({1, 1, 1, 1, 2, 2, 2, 2});
  int x_tensor_numel = static_cast<int>(framework::product(x_tensor_dims));
  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  framework::TensorFromVector(x_tensor_data, ctx, x_tensor);
  x_tensor->Resize(x_tensor_dims);

  desc.SetInput("Scale", {"Scale"});
  framework::DDim scale_tensor_dims({1});
  std::vector<float> scale_tensor_data({1});
  int scale_tensor_numel =
      static_cast<int>(framework::product(scale_tensor_dims));
  auto scale_tensor = scope.Var("Scale")->GetMutable<framework::LoDTensor>();
  framework::TensorFromVector(scale_tensor_data, ctx, scale_tensor);
  scale_tensor->Resize(scale_tensor_dims);

  desc.SetInput("Bias", {"Bias"});
  framework::DDim bias_tensor_dims({1});
  std::vector<float> bias_tensor_data({1});
  int bias_tensor_numel =
      static_cast<int>(framework::product(bias_tensor_dims));
  auto bias_tensor = scope.Var("Bias")->GetMutable<framework::LoDTensor>();
  framework::TensorFromVector(bias_tensor_data, ctx, bias_tensor);
  bias_tensor->Resize(bias_tensor_dims);

  desc.SetOutput("MeanOut", {"MeanOut"});
  auto mean_out_tensor =
      scope.Var("MeanOut")->GetMutable<framework::LoDTensor>();
  desc.SetOutput("VarianceOut", {"VarianceOut"});
  auto variance_out_tensor =
      scope.Var("VarianceOut")->GetMutable<framework::LoDTensor>();
  desc.SetOutput("SavedMean", {"SavedMean"});
  auto saved_mean_tensor =
      scope.Var("SavedMean")->GetMutable<framework::LoDTensor>();
  desc.SetOutput("SavedVariance", {"SavedVariance"});
  auto saved_variance_tensor =
      scope.Var("SavedVariance")->GetMutable<framework::LoDTensor>();

  desc.SetOutput("Y", {"Y"});
  auto out_tensor = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  auto op = framework::OpRegistry::CreateOp(desc);
  auto before_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << before_run_str;

  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  auto after_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << after_run_str;

  int out_tensor_numel =
      static_cast<int>(framework::product(out_tensor->dims()));
  printf("output_tensor dims is: %s\n", out_tensor->dims().to_str().c_str());
  auto out_data = out_tensor->data<T>();
  for (int i = 0; i < out_tensor_numel; i++) {
    printf("%f ", static_cast<float>(out_data[i]));
  }
}

TEST(test_batch_op, cpu_place) {
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  TestBatchNormOp<float>(ctx);
}

#if defined(PADDLE_WITH_ASCEND_CL)
TEST(test_batch_op, npu_place) {
  platform::NPUPlace place(0);
  platform::NPUDeviceContext ctx(place);
  TestBatchNormOp<float>(ctx);
}
#endif

}  // namespace operators
}  // namespace paddle
