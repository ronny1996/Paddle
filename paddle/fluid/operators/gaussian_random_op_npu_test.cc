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

USE_OP(gather);

namespace paddle {
namespace operators {

template <typename T>
void TestGatherOP(const platform::DeviceContext& ctx) {
  auto place = ctx.GetPlace();
  framework::OpDesc desc;
  framework::Scope scope;

  desc.SetType("gather");

  desc.SetInput("X", {"X"});
  framework::DDim input_dims({3, 2});
  size_t input_numel = static_cast<size_t>(framework::product(input_dims));
  auto input_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  std::vector<T> input_data({1, 2, 3, 4, 5, 6});
  framework::TensorFromVector(input_data, ctx, input_tensor);
  input_tensor->Resize(input_dims);

  desc.SetInput("Index", {"Index"});
  framework::DDim index_dims({2});
  size_t index_numel = static_cast<size_t>(framework::product(index_dims));
  auto index_tensor = scope.Var("Index")->GetMutable<framework::LoDTensor>();
  std::vector<int64_t> index_data({1, 2});
  framework::TensorFromVector(index_data, ctx, index_tensor);
  index_tensor->Resize(index_dims);

  desc.SetOutput("Out", {"Out"});
  auto output_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();

  auto op = framework::OpRegistry::CreateOp(desc);

  auto before_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << before_run_str;

  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  auto after_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << after_run_str;

  printf("output_tensor dims is: %s\n", output_tensor->dims().to_str().c_str());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_gather_op, gpu_place) {
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);
  TestGatherOP<float>(ctx);
}
#endif

}  // namespace operators
}  // namespace paddle
