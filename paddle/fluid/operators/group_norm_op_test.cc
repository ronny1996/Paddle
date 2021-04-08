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

#include <fstream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(group_norm);
USE_OP(group_norm_grad);

namespace paddle {
namespace operators {

template <typename T>
static void print_data(const T* data, const int64_t numel,
                       const std::vector<int64_t> dims,
                       const std::string name) {
  printf("------------%s------------\n", name.c_str());
  size_t stride_idx = dims.size() - 1;
  size_t stride = dims[stride_idx];
  while (stride < 2) {
    stride_idx--;
    stride = dims[stride_idx];
  }
  int64_t index = 0;
  while (index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", static_cast<float>(data[index]));
    } else {
      printf("%d ", static_cast<int>(data[index]));
    }
    if ((index + 1) % stride == 0) printf("\n");
    index++;
  }
}

// read data
template <typename T>
static void read_tensor_data(const platform::DeviceContext& ctx,
                             const framework::DDim dims,
                             framework::LoDTensor* tensor,
                             const std::string name,
                             const std::string filename) {
  int numel = static_cast<int>(framework::product(dims));
  std::vector<T> data;
  std::ifstream fin(filename.c_str());
  for (int i = 0; i < numel; ++i) {
    T value;
    fin >> value;
    data.push_back(value);
  }
  fin.close();
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
  print_data(data.data(), numel, framework::vectorize(dims), name);
}

// feed data
template <typename T>
static void feed_tensor_data(const platform::DeviceContext& ctx,
                             const framework::DDim dims,
                             framework::LoDTensor* tensor,
                             const int number_limit, const std::string name) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    // data[i] = static_cast<T>(i % number_limit);
    data[i] = static_cast<T>(1);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
  //   print_data(data.data(), numel, framework::vectorize(dims), name);
}

// input
const std::vector<int64_t> input_dim_vec = {2, 3, 4, 5};
const int64_t input_numel = std::accumulate(
    input_dim_vec.begin(), input_dim_vec.end(), 1, std::multiplies<int64_t>());
const std::vector<int64_t> scale_dim_vec = {3};
const int64_t scale_numel = std::accumulate(
    input_dim_vec.begin(), input_dim_vec.end(), 1, std::multiplies<int64_t>());
const std::vector<int64_t> bias_dim_vec = {3};
const int64_t bias_numel = std::accumulate(
    input_dim_vec.begin(), input_dim_vec.end(), 1, std::multiplies<int64_t>());
// attrs
const float epsilon = 1e-5;
const int groups = 2;
const std::string data_layout = "NCHW";

template <typename T>
void TestGroupNorm(const platform::DeviceContext& ctx,
                   std::vector<float>& y_out, std::vector<float>& mean_out,
                   std::vector<float>& var_out, std::vector<float>& x_grad_out,
                   std::vector<float>& scale_grad_out,
                   std::vector<float>& bias_grad_out) {
  auto place = ctx.GetPlace();
  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim input_dims = framework::make_ddim(input_dim_vec);
  framework::DDim scale_dims = framework::make_ddim(scale_dim_vec);
  framework::DDim bias_dims = framework::make_ddim(bias_dim_vec);

  // --------------- forward ----------------------
  desc_fwd.SetType("group_norm");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetInput("Scale", {"Scale"});
  desc_fwd.SetInput("Bias", {"Bias"});

  desc_fwd.SetOutput("Y", {"Y"});
  desc_fwd.SetOutput("Mean", {"Mean"});
  desc_fwd.SetOutput("Variance", {"Variance"});

  desc_fwd.SetAttr("epsilon", epsilon);
  desc_fwd.SetAttr("groups", groups);
  desc_fwd.SetAttr("data_layout", data_layout);

  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto scale_tensor = scope.Var("Scale")->GetMutable<framework::LoDTensor>();
  auto bias_tensor = scope.Var("Bias")->GetMutable<framework::LoDTensor>();

  auto y_tensor = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  auto mean_tensor = scope.Var("Mean")->GetMutable<framework::LoDTensor>();
  auto var_tensor = scope.Var("Variance")->GetMutable<framework::LoDTensor>();

  // feed input data
  #if 0
  feed_tensor_data<T>(ctx, input_dims, x_tensor, input_numel, "x_tensor");
  feed_tensor_data<T>(ctx, scale_dims, scale_tensor, scale_numel,
                      "scale_tensor");
  feed_tensor_data<T>(ctx, bias_dims, bias_tensor, bias_numel, "bias_tensor");
  #else
  read_tensor_data<T>(ctx, input_dims, x_tensor, "x", "x_tensor.txt");
  read_tensor_data<T>(ctx, scale_dims, scale_tensor, "scale",
  "scale_tensor.txt");
  read_tensor_data<T>(ctx, bias_dims, bias_tensor, "bias",
  "bias_tensor.txt");
  #endif

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*y_tensor, ctx, &y_out);
  framework::TensorToVector(*mean_tensor, ctx, &mean_out);
  framework::TensorToVector(*var_tensor, ctx, &var_out);

  // --------------- backward ----------------------
  desc_bwd.SetType("group_norm_grad");
  desc_bwd.SetInput("Y", {"Y"});
  desc_bwd.SetInput(framework::GradVarName("Y"), {framework::GradVarName("Y")});
  desc_bwd.SetInput("Variance", {"Variance"});
  desc_bwd.SetInput("Scale", {"Scale"});
  desc_bwd.SetInput("Bias", {"Bias"});

  desc_bwd.SetAttr("epsilon", epsilon);
  desc_bwd.SetAttr("groups", groups);
  desc_bwd.SetAttr("data_layout", data_layout);

  desc_bwd.SetOutput(framework::GradVarName("X"),
                     {framework::GradVarName("X")});
  desc_bwd.SetOutput(framework::GradVarName("Scale"),
                     {framework::GradVarName("Scale")});
  desc_bwd.SetOutput(framework::GradVarName("Bias"),
                     {framework::GradVarName("Bias")});

  auto y_grad_tensor = scope.Var(framework::GradVarName("Y"))
                           ->GetMutable<framework::LoDTensor>();

  auto x_grad_tensor = scope.Var(framework::GradVarName("X"))
                           ->GetMutable<framework::LoDTensor>();
  auto scale_grad_tensor = scope.Var(framework::GradVarName("Scale"))
                               ->GetMutable<framework::LoDTensor>();
  auto bias_grad_tensor = scope.Var(framework::GradVarName("Bias"))
                              ->GetMutable<framework::LoDTensor>();

  // feed loss_grad_tensor data
  feed_tensor_data<T>(ctx, y_tensor->dims(), y_grad_tensor, y_tensor->numel(),
                      "y_grad_tensor");

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*x_grad_tensor, ctx, &x_grad_out);
  framework::TensorToVector(*scale_grad_tensor, ctx, &scale_grad_out);
  framework::TensorToVector(*bias_grad_tensor, ctx, &bias_grad_out);
}

template <typename T>
static void compare_results(const std::vector<T> cpu_out,
                            const std::vector<T> gpu_out, const int64_t numel,
                            const std::vector<int64_t> dims,
                            const std::string name) {
  auto result = std::equal(
      cpu_out.begin(), cpu_out.end(), gpu_out.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-9; });
  if (!result) {
    LOG(INFO) << "=========== Ouptut " << name << " is NOT Equal ===========";
    print_data(cpu_out.data(), numel, dims, name + "_cpu");
    print_data(gpu_out.data(), numel, dims, name + "_gpu");
  } else {
    LOG(INFO) << "=========== Ouptut " << name
              << " is Equal in CPU and GPU ===========";
    print_data(cpu_out.data(), numel, dims, name + "_cpu");
    print_data(gpu_out.data(), numel, dims, name + "_gpu");
  }
}

template <typename T>
static void compare_results(const std::vector<T> cpu_out,
                            const std::vector<T> gpu_out,
                            const std::string name) {
  auto result = std::equal(
      cpu_out.begin(), cpu_out.end(), gpu_out.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-3; });
  if (!result) {
    LOG(INFO) << "=========== Ouptut " << name << " is NOT Equal ===========";
    printf("\ncpu_out:\n");
    for (auto& t : cpu_out) {
      printf("%.4f ", t);
    }
    printf("\ngpu_out:\n");
    for (auto& t : gpu_out) {
      printf("%.4f ", t);
    }
    printf("\n");
  } else {
    LOG(INFO) << "=========== Ouptut " << name
              << " is Equal in CPU and GPU ===========";
    printf("\ncpu_out:\n");
    for (auto& t : cpu_out) {
      printf("%.4f ", t);
    }
    printf("\ngpu_out:\n");
    for (auto& t : gpu_out) {
      printf("%.4f ", t);
    }
    printf("\n");
  }
}

TEST(test_group_norm_op, compare_cpu_and_gpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> y_out_cpu;
  std::vector<float> mean_out_cpu;
  std::vector<float> var_out_cpu;
  std::vector<float> x_grad_out_cpu;
  std::vector<float> scale_grad_out_cpu;
  std::vector<float> bias_grad_out_cpu;
  TestGroupNorm<float>(cpu_ctx, y_out_cpu, mean_out_cpu, var_out_cpu,
                       x_grad_out_cpu, scale_grad_out_cpu, bias_grad_out_cpu);

  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> y_out_gpu;
  std::vector<float> mean_out_gpu;
  std::vector<float> var_out_gpu;
  std::vector<float> x_grad_out_gpu;
  std::vector<float> scale_grad_out_gpu;
  std::vector<float> bias_grad_out_gpu;
  TestGroupNorm<float>(gpu_ctx, y_out_gpu, mean_out_gpu, var_out_gpu,
                       x_grad_out_gpu, scale_grad_out_gpu, bias_grad_out_gpu);

#define COMPARE_CPU_AND_GPU(out) \
  compare_results<float>(out##_cpu, out##_gpu, #out)
  COMPARE_CPU_AND_GPU(y_out);
  COMPARE_CPU_AND_GPU(mean_out);
  COMPARE_CPU_AND_GPU(var_out);
  COMPARE_CPU_AND_GPU(x_grad_out);
  COMPARE_CPU_AND_GPU(scale_grad_out);
  COMPARE_CPU_AND_GPU(bias_grad_out);
#undef COMPARE_CPU_AND_GPU
}

}  // namespace operators
}  // namespace paddle
