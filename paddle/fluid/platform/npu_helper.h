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

#pragma once
#ifdef PADDLE_WITH_ASCEND_CL
#include <glog/logging.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"

#include "ge/ge_api.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace paddle {
namespace platform {
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace platform {

ge::DataType VarTypeToGeType(paddle::framework::proto::VarType::Type type) {
  if (type == paddle::framework::proto::VarType::FP16) {
    return ge::DataType::DT_FLOAT16;
  } else if (type == paddle::framework::proto::VarType::FP32) {
    return ge::DataType::DT_FLOAT;
  } else if (type == paddle::framework::proto::VarType::FP64) {
    return ge::DataType::DT_DOUBLE;
  } else if (type == paddle::framework::proto::VarType::INT32) {
    return ge::DataType::DT_INT32;
  } else if (type == paddle::framework::proto::VarType::INT64) {
    return ge::DataType::DT_INT64;
  } else {
    PADDLE_THROW(paddle::platform::errors::Unimplemented(
        "Not support %s as tensor type.",
        paddle::framework::DataTypeToString(type)));
  }
}

template <typename T>
class GeDataType;

#define DECLARE_GE_DATA_TYPE(DTYPE, SCALING_TYPE, BN_TYPE, GE_TYPE) \
  template <>                                                       \
  class GeDataType<DTYPE> {                                         \
   public:                                                          \
    static const ge::DataType type = GE_TYPE;                       \
    using ScalingParamType = const SCALING_TYPE;                    \
    using BatchNormParamType = BN_TYPE;                             \
    static ScalingParamType* kOne() {                               \
      static ScalingParamType v = 1.0;                              \
      return &v;                                                    \
    }                                                               \
    static ScalingParamType* kZero() {                              \
      static ScalingParamType v = 0.0;                              \
      return &v;                                                    \
    }                                                               \
  }

DECLARE_GE_DATA_TYPE(float16, float, float, ge::DataType::DT_FLOAT16);
DECLARE_GE_DATA_TYPE(float, float, float, ge::DataType::DT_FLOAT);
DECLARE_GE_DATA_TYPE(double, double, double, ge::DataType::DT_DOUBLE);

#undef DECLARE_GE_DATA_TYPE

}  // namespace platform
}  // namespace paddle
#endif  // PADDLE_WITH_ASCEND_CL
