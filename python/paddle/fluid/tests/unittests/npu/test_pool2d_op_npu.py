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

paddle.enable_static()

@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMaxPool2D_Op(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True
      
    def _get_places(self):
        return [fluid.NPUPlace(0)]

    def init_data_type(self):
        self.dtype = np.float32

    def init_pool_type(self):
        self.pool_type = "max"

    def init_test_case(self):
        self.global_pool = False
        self.ceil_mode = False
        self.adaptive = False
        self.exclusive = True
        self.padding_algorithm = "EXPLICIT"
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]
        self.shape = [2, 3, 5, 5]
        self.data_format = "NCHW"

    
    def setUp(self):
        self.set_npu()
        self.op_type = "pool2d"
        self.use_mkldnn = False
        self.init_data_type()
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
        self.check_output(atol=1e-3, check_dygraph=False)
  
    def test_check_grad(self):
        self.check_grad(set(["X"]), "Out", max_relative_error=1e-3, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAvgPool2D_Op(TestMaxPool2D_Op):
    def init_pool_type(self):
        self.pool_type = "avg"

if __name__ == "__main__":
    unittest.main()
