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
from mxnet import gluon, init, nd
import numpy as np
from distutils.util import strtobool
from rcnn.config import config, generate_config


class wh_processOperator(mx.operator.CustomOp):
    def __init__(self, stride=0, network='', landmark = False, prefix=''):
        super(wh_processOperator, self).__init__()
        self.stride = int(stride)
        self.prefix = prefix
        self.landmark = landmark

    def _gather_feat(self,feat, ind, mask=None):
        # K cannot be 1 for this implementation
        K = ind.shape[1]
        batch_size = ind.shape[0]
        attri_dim = feat.shape[2]

        flatten_ind = ind.flatten()
        for i in range(batch_size):
            if i == 0:
                output = feat[i, ind[i]].expand_dims(2)
            else:
                output = nd.concat(output, feat[i, ind[i]].expand_dims(2), dim=2)

        output = output.swapaxes(dim1=1, dim2=2)
        return output

    def _tranpose_and_gather_feat(self,feat, ind):
        feat = nd.transpose(feat, axes=(0, 2, 3, 1))
        feat = nd.reshape(feat, shape=(feat.shape[0], -1, feat.shape[3]))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, is_train, req, in_data, out_data, aux):
        wh_pred = in_data[0]
        ind = in_data[1]
        #mask = in_data[2]
        wh_pred = self._tranpose_and_gather_feat(wh_pred,ind)
        wh_pred = wh_pred.swapaxes(dim1=0, dim2=1)
        #mask = mask.expand_dims(axis=2).broadcast_like(wh_pred).astype('float32')
        self.assign(out_data[0], req[0], wh_pred)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        self.assign(in_grad[0], req[0], 0)
        # self.assign(in_grad[2],req[2], out_grad[1])


@mx.operator.register('wh_process')
class wh_processProp(mx.operator.CustomOpProp):
    def __init__(self, stride=0, landmark = False, network='', dataset='', prefix=''):
        super(wh_processProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['wh_pred','ind']

    def list_outputs(self):
        return ['pred']

    def infer_shape(self, in_shape):
        wh_shape = in_shape[0]
        ind_shape = in_shape[1]
        wh_out = (ind_shape[0],ind_shape[1],2)
        return [wh_shape,ind_shape],[wh_out]

    def create_operator(self, ctx, shapes, dtypes):
        return wh_processOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        deps = []
        if self.need_top_grad_:
            deps.extend(out_grad)
        return deps


