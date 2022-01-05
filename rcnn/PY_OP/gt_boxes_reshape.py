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


from __future__ import print_function
import sys
import mxnet as mx
import numpy as np
from distutils.util import strtobool
from ..config import config, generate_config


class gt_boxes_reshapeOperator(mx.operator.CustomOp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(gt_boxes_reshapeOperator, self).__init__()
        self.stride = int(stride)
        self.prefix = prefix
        generate_config(network, dataset)

    def forward(self, is_train, req, in_data, out_data, aux):
        gt_boxes = in_data[0].asnumpy()
        gt_landmarks = in_data[1].asnumpy()
        #print(gt_boxes)
        indices = np.where(gt_boxes != -1)[1]
        max_index = max(indices)
        shape0 = gt_boxes.shape[0]
        gt_boxes = gt_boxes[:,:max_index+1,:].reshape(gt_boxes.shape[1],gt_boxes.shape[2])
        gt_landmarks = gt_landmarks[:,:max_index+1,:,:].reshape(gt_landmarks.shape[1], gt_landmarks.shape[2], gt_landmarks.shape[3])
        # temp = gt_boxes[:, 4].copy()
        # for i in range(gt_boxes.shape[1] - 1):
        #     gt_boxes[:, gt_boxes.shape[1] - i - 1] = gt_boxes[:, gt_boxes.shape[1] - i - 2]
        # gt_boxes[:, 0] = temp
        #gt_output = gt_boxes.reshape(shape0, gt_boxes.shape[0], gt_boxes.shape[1])

        self.assign(out_data[0], req[0], gt_boxes)
        self.assign(out_data[1], req[1], gt_landmarks)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('gt_boxes_reshape')
class gt_boxes_reshapeProp(mx.operator.CustomOpProp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(gt_boxes_reshapeProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['gt_boxes', 'gt_landmarks']

    def list_outputs(self):
        return ['gt_boxes', 'gt_landmarks']

    def infer_shape(self, in_shape):
        gt_boxes_shape = in_shape[0]
        #print('in_rpn_ohem', labels_shape, anchor_weight_shape)
        out_shape = (gt_boxes_shape[0], gt_boxes_shape[1],gt_boxes_shape[2])
        return gt_boxes_shape, out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return gt_boxes_reshapeOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


