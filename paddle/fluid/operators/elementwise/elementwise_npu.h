/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
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

  // 1.expand the axis with dim 1 in middle
  Tensor trimmed_src;
  trimmed_src.ShareDataWith(*src);
  trimmed_src.Resize(trim_src_dims);

  auto tmp_expand_to_dims = trim_trailing_singular_dims(
      framework::slice_ddim(dst_dims, axis, tile_axis));
  for (auto i = 0; i < trim_src_dims.size(); i++) {
    if (trim_src_dims[i] == 1 && tmp_expand_to_dims[i] > 1) {
      Tensor tmp_src;
      auto tmp_ddim = trimmed_src.dims();
      tmp_ddim[i] = tmp_expand_to_dims[i];
      tmp_src.mutable_data<T>(tmp_ddim, ctx.GetPlace());
      const auto& tmp_runner =
          NpuOpRunner("TileWithAxis", {trimmed_src}, {tmp_src},
                      {{"axis", static_cast<int64_t>(i)},
                       {"tiles", static_cast<int64_t>(tmp_expand_to_dims[i])}});
      tmp_runner.Run(stream);
      trimmed_src.ShareDataWith(tmp_src);
      trimmed_src.Resize(tmp_ddim);
    }
  }

  // 2.expand the ahead axis
  auto expand_to_dims = framework::slice_ddim(dst_dims, 0, tile_axis);
  auto expand_to_dims_vec = framework::vectorize<int>(expand_to_dims);
  Tensor tmp_tensor;
  tmp_tensor.mutable_data<T>(expand_to_dims, ctx.GetPlace());
  const auto& expand_runner = NpuOpRunner(
      "ExpandD", {trimmed_src}, {tmp_tensor}, {{"shape", expand_to_dims_vec}});
  expand_runner.Run(stream);

  // 3.expand the tail axis
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

}  // namespace operators
}  // namespace paddle
#endif
