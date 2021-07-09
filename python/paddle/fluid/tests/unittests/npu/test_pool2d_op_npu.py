#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import sys
import unittest
import numpy as np
sys.path.append("..")

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest
from test_pool2d_op import pool2D_forward_naive
from paddle.nn.functional import avg_pool2d, max_pool2d

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMaxPool2D_Op(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True

    def init_data_type(self):
        self.dtype = np.float32

    def init_pool_type(self):
        self.pool_type = "max"

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_data_shape(self):
        self.shape = [2, 3, 5, 5]

    def init_test_case(self):
        self.global_pool = False
        self.ceil_mode = False
        self.adaptive = False
        self.exclusive = True
        self.padding_algorithm = "EXPLICIT"
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def setUp(self):
        self.set_npu()
        self.op_type = "pool2d"
        self.use_mkldnn = False
        self.init_data_type()
        self.init_data_format()
        self.init_data_shape()
        self.init_pool_type()
        self.init_test_case()

        input = (np.random.random(self.shape).astype(self.dtype) - 0.5) * 5
        output = pool2D_forward_naive(
            input, self.ksize, self.strides, self.paddings, self.global_pool,
            self.ceil_mode, self.exclusive, self.adaptive, self.data_format,
            self.pool_type, self.padding_algorithm).astype(self.dtype)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(input)}
        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'use_cudnn': False,
            'use_mkldnn': self.use_mkldnn,
            'ceil_mode': self.ceil_mode,
            'data_format': self.data_format,
            'exclusive': self.exclusive,
            'adaptive': self.adaptive,
            "padding_algorithm": self.padding_algorithm,
        }
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output_with_place(
            fluid.NPUPlace(0), atol=1e-3, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMaxPool2D_Op_case2(TestMaxPool2D_Op):
    def init_test_case(self):
        self.global_pool = False
        self.ceil_mode = False
        self.adaptive = False
        self.exclusive = True
        self.padding_algorithm = "EXPLICIT"
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [1, 1]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMaxPool2D_Op_NHWC(TestMaxPool2D_Op):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_data_shape(self):
        self.shape = [2, 5, 5, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgPool2D_Op(TestMaxPool2D_Op):
    def init_pool_type(self):
        self.pool_type = "avg"


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgPool2D_Op_case2(TestAvgPool2D_Op):
    def init_test_case(self):
        self.global_pool = False
        self.ceil_mode = False
        self.adaptive = False
        self.exclusive = True
        self.padding_algorithm = "EXPLICIT"
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [1, 1]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgPool2D_Op_NHWC(TestMaxPool2D_Op_NHWC):
    def init_pool_type(self):
        self.pool_type = "avg"


def pool2d_grad_naive(x, ksize, strides, paddings, pool_type, data_format="NCHW"):
    if data_format == "NHWC":
        x = x.transpose([0, 3, 1, 2])
    x_grad = np.zeros(x.shape, dtype=np.float32)
    N, C, H, W = x.shape

    for n in range(N):
        for c in range(C):
            for h in range(0, H - ksize[0] + 1, strides[0]):
                for w in range(0, W - ksize[1] + 1, strides[1]):
                    if pool_type == "max":
                        idx = np.argmax(x[n, c, h:h + ksize[0], w:w + ksize[1]]
                                        .flatten())
                        idx_h = idx // ksize[1]
                        idx_w = idx % ksize[1]
                        x_grad[n, c, h + idx_h, w + idx_w] += 1
                    elif pool_type == "avg":
                        idx = np.meshgrid(
                            range(h, h + ksize[0]), range(w, w + ksize[1]))
                        x_grad[n, c, idx[0], idx[1]] += 1 / ksize[0] / ksize[1]
    if data_format == "NHWC":
        x_grad = x_grad.transpose([0, 2, 3, 1])
    return x_grad


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestPool2D_API(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def check_pool(self,
                   place,
                   shape,
                   ksize,
                   strides,
                   paddings,
                   pool_type='max'):
        with fluid.dygraph.guard(place):
            input_np = np.random.random(shape).astype("float32")
            input_var = paddle.to_tensor(input_np, stop_gradient=False)

            result_np = pool2D_forward_naive(
                input_np,
                ksize=ksize,
                strides=strides,
                paddings=paddings,
                pool_type=pool_type)
            if pool_type == 'max':
                result = max_pool2d(
                    input_var,
                    kernel_size=ksize,
                    stride=strides,
                    padding=paddings)
            else:
                result = avg_pool2d(
                    input_var,
                    kernel_size=ksize,
                    stride=strides,
                    padding=paddings)
            self.assertTrue(np.allclose(result.numpy(), result_np, rtol=1e-3))

            result.sum().backward()
            input_grad_np = pool2d_grad_naive(
                input_np,
                ksize=ksize,
                strides=strides,
                paddings=paddings,
                pool_type=pool_type)
            self.assertTrue(
                np.allclose(
                    input_var.grad.numpy(), input_grad_np, rtol=1e-3))

    def test_pool2d(self):
        place = core.NPUPlace(0)
        self.check_pool(
            place, [2, 3, 5, 5], [2, 2], [2, 2], [0, 0], pool_type='max')
        self.check_pool(
            place, [2, 3, 5, 5], [2, 2], [1, 1], [0, 0], pool_type='avg')


if __name__ == "__main__":
    unittest.main()
