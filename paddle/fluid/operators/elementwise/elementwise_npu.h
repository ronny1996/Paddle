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
void NpuBroadcast(const platform::NPUDeviceContext& dev_ctx, const Tensor* src, int axis, const framework::DDim& dst_dims, Tensor* transformed_src) {
  auto stream = dev_ctx.stream();

  // 1. expand the axis with dim 1 
  auto src_dims = src->dims();
  Tensor tmp_src;
  tmp_src.ShareDataWith(*src);
  tmp_src.Resize(src_dims);
  for (int i = 0; i < src_dims.size(); ++i) {
    if (src_dims[i] == 1 && dst_dims[i + axis] > 1) {
      Tensor tmp_tensor;
      auto tmp_tensor_dims = tmp_src.dims();
      tmp_tensor_dims[i] = dst_dims[i + axis];
      tmp_tensor.mutable_data<T>(tmp_tensor_dims, dev_ctx.GetPlace());
      const auto& runner =
          NpuOpRunner("TileWithAxis", {tmp_src}, {tmp_tensor},
                      {{"axis", static_cast<int64_t>(i)},
                       {"tiles", static_cast<int64_t>(dst_dims[i + axis])}});
      runner.Run(stream);
      tmp_src.ShareDataWith(tmp_tensor);
      tmp_src.Resize(tmp_tensor_dims);
    }
  }

  // 2.expand the ahead axis
  auto prev = framework::product(framework::slice_ddim(dst_dims, 0, axis));
  if (prev > 1) {
    Tensor tmp_tensor;
    auto tmp_tensor_dims = framework::slice_ddim(dst_dims, 0, axis + src_dims.size());
    tmp_tensor.mutable_data<T>(tmp_tensor_dims, dev_ctx.GetPlace());
    const auto& runner = NpuOpRunner(
        "ExpandD", {tmp_src}, {tmp_tensor}, {{"shape", framework::vectorize<int64_t>(tmp_tensor_dims)}});
    runner.Run(stream);
      tmp_src.ShareDataWith(tmp_tensor);
      tmp_src.Resize(tmp_tensor_dims);
  }

  // 3.expand the tail axis
  auto post = framework::product(framework::slice_ddim(dst_dims, axis + src_dims.size(), dst_dims.size()));
  if (post > 1) {
    auto src_dims_vec = framework::vectorize<int>(tmp_src.dims());
    src_dims_vec.push_back(1);
    tmp_src.Resize(framework::make_ddim(src_dims_vec));

    Tensor tmp_tensor;
    tmp_tensor.mutable_data<T>(dst_dims, dev_ctx.GetPlace());
    const auto& runner =
        NpuOpRunner("TileWithAxis", {tmp_src}, {tmp_tensor},
                    {{"axis", static_cast<int64_t>(axis + src_dims.size())},
                     {"tiles", static_cast<int64_t>(post)}});
    runner.Run(stream);
    tmp_src.ShareDataWith(tmp_tensor);
  }
  tmp_src.Resize(dst_dims);
  framework::TensorCopy(tmp_src, dev_ctx.GetPlace(), transformed_src);
}

template <typename T>
void NpuElementWiseOpBroadcast(const platform::NPUDeviceContext& dev_ctx, const Tensor* x, const Tensor* y, int axis, Tensor* transformed_x, Tensor* transformed_y) {
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  std::vector<int> dst_dims_vec = framework::vectorize<int>(x_dims);

  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
    dst_dims_vec = framework::vectorize<int>(y_dims);
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  int x_axis = is_xsize_larger? 0 : axis;
  int y_axis = is_xsize_larger? axis : 0;

  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));

  for (int i = 0; i < x_dims.size(); ++i) {
    dst_dims_vec[i + x_axis] = std::max(dst_dims_vec[i + x_axis], static_cast<int>(x_dims[i]));
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    dst_dims_vec[i + y_axis] = std::max(dst_dims_vec[i + y_axis], static_cast<int>(y_dims[i]));
  }

  auto dst_dims = framework::make_ddim(dst_dims_vec);
  NpuBroadcast<T>(dev_ctx, x, x_axis, dst_dims, transformed_x);
  NpuBroadcast<T>(dev_ctx, y, y_axis, dst_dims, transformed_y);

#if 0
  auto stream = dev_ctx.stream();
  // 1. expand the axis with dim 1 
  Tensor tmp_x;
  tmp_x.ShareDataWith(*x);
  tmp_x.Resize(x_dims);
  for (int i = 0; i < x_dims.size(); ++i) {
    if (x_dims[i] == 1 && dst_dims_vec[i] > 1) {
      Tensor tmp_tensor;
      auto tmp_tensor_dims = tmp_x.dims();
      tmp_tensor_dims[i] = dst_dims_vec[i];
      tmp_tensor.mutable_data<T>(tmp_tensor_dims, dev_ctx.GetPlace());
      const auto& runner =
          NpuOpRunner("TileWithAxis", {tmp_x}, {tmp_tensor},
                      {{"axis", static_cast<int64_t>(i)},
                       {"tiles", static_cast<int64_t>(dst_dims_vec[i])}});
      runner.Run(stream);
      tmp_x.ShareDataWith(tmp_tensor);
      tmp_x.Resize(tmp_tensor_dims);
    }
  }

  Tensor tmp_y;
  tmp_y.ShareDataWith(*y);
  tmp_y.Resize(y_dims);
  for (int i = 0; i < y_dims.size(); ++i) {
    if (y_dims[i] == 1 && dst_dims_vec[i] > 1) {
      Tensor tmp_tensor;
      auto tmp_tensor_dims = tmp_y.dims();
      tmp_tensor_dims[i] = dst_dims_vec[i];
      tmp_tensor.mutable_data<T>(tmp_tensor_dims, dev_ctx.GetPlace());
      const auto& runner =
          NpuOpRunner("TileWithAxis", {tmp_y}, {tmp_tensor},
                      {{"axis", static_cast<int64_t>(i)},
                       {"tiles", static_cast<int64_t>(dst_dims_vec[i])}});
      runner.Run(stream);
      tmp_y.ShareDataWith(tmp_tensor);
      tmp_y.Resize(tmp_tensor_dims);
    }
  }

  // 2.expand the ahead axis
  auto x_prev = framework::product(framework::slice_ddim(dst_dims, 0, x_axis));
  auto y_prev = framework::product(framework::slice_ddim(dst_dims, 0, y_axis));
  if (x_prev > 1) {
    Tensor tmp_tensor;
    auto tmp_tensor_dims = framework::slice_ddim(dst_dims, 0, x_axis + x_dims.size());
    tmp_tensor.mutable_data<T>(tmp_tensor_dims, dev_ctx.GetPlace());
    const auto& runner = NpuOpRunner(
        "ExpandD", {tmp_x}, {tmp_tensor}, {{"shape", framework::vectorize<int64_t>(tmp_tensor_dims)}});
    runner.Run(stream);
      tmp_x.ShareDataWith(tmp_tensor);
      tmp_x.Resize(tmp_tensor_dims);
  }
  if (y_prev > 1) {
    Tensor tmp_tensor;
    auto tmp_tensor_dims = framework::slice_ddim(dst_dims, 0, y_axis + y_dims.size());
    tmp_tensor.mutable_data<T>(tmp_tensor_dims, dev_ctx.GetPlace());
    const auto& runner = NpuOpRunner(
        "ExpandD", {tmp_y}, {tmp_tensor}, {{"shape", framework::vectorize<int64_t>(tmp_tensor_dims)}});
    runner.Run(stream);
    tmp_y.ShareDataWith(tmp_tensor);
    tmp_y.Resize(tmp_tensor_dims);
  }

  // 3.expand the tail axis
  auto x_post = framework::product(framework::slice_ddim(dst_dims, x_axis + x_dims.size(), dst_dims.size()));
  auto y_post = framework::product(framework::slice_ddim(dst_dims, y_axis + y_dims.size(), dst_dims.size()));
  if (x_post > 1) {
    auto x_dims_vec = framework::vectorize<int>(tmp_x.dims());
    x_dims_vec.push_back(1);
    tmp_x.Resize(framework::make_ddim(x_dims_vec));

    Tensor tmp_tensor;
    tmp_tensor.mutable_data<T>(dst_dims, dev_ctx.GetPlace());
    const auto& runner =
        NpuOpRunner("TileWithAxis", {tmp_x}, {tmp_tensor},
                    {{"axis", static_cast<int64_t>(x_axis + x_dims.size())},
                     {"tiles", static_cast<int64_t>(x_post)}});
    runner.Run(stream);
    tmp_x.ShareDataWith(tmp_tensor);
  }
  if (y_post > 1) {
    auto y_dims_vec = framework::vectorize<int>(tmp_y.dims());
    y_dims_vec.push_back(1);
    tmp_y.Resize(framework::make_ddim(y_dims_vec));

    Tensor tmp_tensor;
    tmp_tensor.mutable_data<T>(dst_dims, dev_ctx.GetPlace());
    const auto& runner =
        NpuOpRunner("TileWithAxis", {tmp_y}, {tmp_tensor},
                    {{"axis", static_cast<int64_t>(y_axis + y_dims.size())},
                     {"tiles", static_cast<int64_t>(y_post)}});
    runner.Run(stream);
    tmp_y.ShareDataWith(tmp_tensor);
  }
  tmp_x.Resize(dst_dims);
  tmp_y.Resize(dst_dims);
  framework::TensorCopy(tmp_x, dev_ctx.GetPlace(), transformed_x);
  framework::TensorCopy(tmp_y, dev_ctx.GetPlace(), transformed_y);
#endif
}
#if 0
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
#endif
}  // namespace operators
}  // namespace paddle
#endif
