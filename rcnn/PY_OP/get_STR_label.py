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


class get_STR_labelOperator(mx.operator.CustomOp):
    def __init__(self, stride=0, network='', landmark = False, prefix=''):
        super(get_STR_labelOperator, self).__init__()
        self.stride = int(stride)
        self.prefix = prefix
        self.landmark = landmark

    def forward(self, is_train, req, in_data, out_data, aux):
        label_STR = in_data[0]
        print(len(label_STR))
        label = label_STR['%s_label_stride%d' % (self.prefix, self.stride)]
        bbox_target = label_STR['%s_bbox_target_stride%d' % (self.prefix, self.stride)]
        bbox_weight = label_STR['%s_bbox_weight_stride%d' % (self.prefix, self.stride)]
        # if landmark:
        #     landmark_weight = STC_para[stride][2]
        if self.landmark:
            landmark_target = label_STR['%s_landmark_target_stride%d' % (self.prefix, self.stride)]
            landmark_weight = label_STR['%s_landmark_weight_stride%d' % (self.prefix, self.stride)]

        self.assign(out_data[0], req[0], [label,bbox_target,bbox_weight,landmark_target,landmark_weight])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('get_STR_label')
class get_STR_labelProp(mx.operator.CustomOpProp):
    def __init__(self, stride=0, landmark = False, network='', dataset='', prefix=''):
        super(get_STR_labelProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['label_STR']

    def list_outputs(self):
        return ['label','bbox_target','bbox_weight','landmark_target','landmark_weight']

    def infer_shape(self, in_shape):
        gt_boxes_shape = in_shape[0]
        #print('in_rpn_ohem', labels_shape, anchor_weight_shape)
        out_shape = (gt_boxes_shape[0], gt_boxes_shape[1],gt_boxes_shape[2])
        return gt_boxes_shape, out_shape

    def create_operator(self, ctx, shapes, dtypes):
        return get_STR_labelOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


