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

USE_OP(gaussian_random);

namespace paddle {
namespace operators {

template <typename T>
void TestGaussianRandomOp(const platform::DeviceContext& ctx) {
  auto place = ctx.GetPlace();
  framework::OpDesc desc;
  framework::Scope scope;

  desc.SetType("gaussian_random");

  desc.SetInput("ShapeTensor", {"ShapeTensor"});
  framework::DDim shape_tensor_dims({2});
  std::vector<T> shape_tensor_data({2, 3});
  size_t shape_tensor_numel =
      static_cast<size_t>(framework::product(shape_tensor_dims));
  auto shape_tensor =
      scope.Var("ShapeTensor")->GetMutable<framework::LoDTensor>();
  framework::TensorFromVector(shape_tensor_data, ctx, shape_tensor);
  shape_tensor->Resize(shape_tensor_dims);

  desc.SetOutput("Out", {"Out"});
  auto out_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();
  auto op = framework::OpRegistry::CreateOp(desc);
  auto before_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << before_run_str;

  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  auto after_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << after_run_str;

  size_t out_tensor_numel =
      static_cast<size_t>(framework::product(out_tensor->dims()));
  printf("output_tensor dims is: %s\n", out_tensor->dims().to_str().c_str());
  auto out_data = out_tensor->data<T>();
  for (size_t i = 0; i < out_tensor_numel; i++) {
    printf("%f ", static_cast<float>(out_data[i]));
  }
}

#if defined(PADDLE_WITH_ASCEND_CL)
TEST(test_gather_op, gpu_place) {
  platform::NPUPlace place(0);
  platform::NPUDeviceContext ctx(place);
  TestGaussianRandomOp<float>(ctx);
}
#endif

}  // namespace operators
}  // namespace paddle
