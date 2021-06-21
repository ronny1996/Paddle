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
