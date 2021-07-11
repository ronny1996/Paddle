#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from test_momentum_op import calculate_momentum_by_numpy

paddle.enable_static()
SEED = 2021

class TestMomentumOp1(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True

    def setUp(self):
        self.set_npu()
        self.op_type = "momentum"
        self.dtype = np.float32
        self.init_dtype()

        param = np.random.random((123, 321)).astype(self.dtype)
        grad = np.random.random((123, 321)).astype(self.dtype)
        velocity = np.zeros((123, 321)).astype(self.dtype)
        learning_rate = np.array([0.001]).astype(np.float32)
        mu = 0.0001
        use_nesterov = False

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu}

        param_out, velocity_out = calculate_momentum_by_numpy(
            param=param,
            grad=grad,
            mu=mu,
            velocity=velocity,
            use_nesterov=use_nesterov,
            learning_rate=learning_rate)

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(core.NPUPlace(0), check_dygraph=False)

class TestMomentumOp2(OpTest):
    '''Test Momentum with default values for attributes
    '''
    def set_npu(self):
        self.__class__.use_npu = True

    def setUp(self):
        self.set_npu()
        self.op_type = "momentum"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        mu = 0.0001
        use_nesterov = True

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu, 'use_nesterov': use_nesterov}

        param_out, velocity_out = calculate_momentum_by_numpy(
            param=param,
            grad=grad,
            mu=mu,
            velocity=velocity,
            use_nesterov=use_nesterov,
            learning_rate=learning_rate)

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def test_check_output(self):
        self.check_output_with_place(core.NPUPlace(0), check_dygraph=False)

if __name__ == "__main__":
    unittest.main()
