# Copyright 2021 Xilinx Inc.
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

import mxnet as mx
#from operator_py.RmSelfAtten import *


def non_local_block(insym, num_filter, mode='Embedded Gaussian', resample=True, ith=0):
    """Return nonlocal neural network block
    Parameters
    ----------
    insym : mxnet symbol
        Input symbol
    num_filter : int
        Number of input channels
    mode : str
        `mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`
    """
    # only Embedded Gaussian mode for 3d feature is implemented
    inter_filter = num_filter / 2 if num_filter >= 1024 else num_filter
    indata1 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=inter_filter,
                                 no_bias=True, name='nonlocal_conv%d1' % ith)
    indata2 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=inter_filter,
                                 no_bias=True, name='nonlocal_conv%d2' % ith)

    # data size: batch_size x (num_filter / 2) x HW
    indata1 = mx.sym.reshape(indata1, shape=(0, 0, -1))
    indata2 = mx.sym.reshape(indata2, shape=(0, 0, -1))

    # f size: batch_size x HW x HW
    f = mx.sym.batch_dot(lhs=indata1, rhs=indata2, transpose_a=True, name='nonlocal_dot%d1' % ith)

    # add softmax layer
    f = mx.sym.softmax(f, axis=2)

    indata3 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=inter_filter,
                                 no_bias=True, name='nonlocal_conv3%d' % ith)
    # g size: batch_size x (num_filter / 2) x HW
    g = mx.sym.reshape(indata3, shape=(0, 0, -1))

    y = mx.sym.batch_dot(lhs=f, rhs=g, transpose_b=True, name='nonlocal_dot%d2' % ith)
    y = mx.sym.reshape_like(lhs=mx.sym.transpose(y, axes=(0, 2, 1)), rhs=indata3)
    # y = mx.sym.reshape_like(lhs=y, rhs=indata3)
    y = mx.sym.Convolution(y, kernel=(1, 1), stride=(1, 1), num_filter=num_filter,
                           no_bias=True, name='nonlocal_conv%d4' % ith)
    outsym = insym + y
    return outsym
