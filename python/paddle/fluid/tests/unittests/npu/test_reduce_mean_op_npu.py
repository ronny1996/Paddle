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

paddle.enable_static()

@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestReduceMean(OpTest):
    def setUp(self):
        np.random.seed(2021)
        self.set_npu()
        self.init_dtype()
        self.place = paddle.NPUPlace(0)
        self.init_op_type()
        self.initTestCase()

        self.use_mkldnn = False
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keep_dim,
            'reduce_all': self.reduce_all
        }
        self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
        if self.attrs['reduce_all']:
            self.outputs = {'Out': self.inputs['X'].mean()}
        else:
            self.outputs = {
                'Out': self.inputs['X'].mean(axis=self.axis,
                                            keepdims=self.attrs['keep_dim'])
            }

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_op_type(self):
        self.op_type = "reduce_mean"
        self.use_mkldnn = False
        self.keep_dim = False
        self.reduce_all = False

    def initTestCase(self):
        self.shape = (5, 6)
        self.axis = (0, )

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
