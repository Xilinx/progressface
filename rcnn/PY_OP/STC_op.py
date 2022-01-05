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


STAT = {0:0}
STEP = 28800

class STCOperator(mx.operator.CustomOp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(STCOperator, self).__init__()
        self.stride = int(stride)
        self.prefix = prefix
        generate_config(network, dataset)

    def forward(self, is_train, req, in_data, out_data, aux):
        global STAT

        cls_score    = in_data[0].asnumpy() #BS, 2, ANCHORS
        labels_raw       = in_data[1].asnumpy() # BS, ANCHORS

        A = config.NUM_ANCHORS
        anchor_weight = np.zeros( (labels_raw.shape[0], labels_raw.shape[1],1), dtype=np.float32 )
        valid_count = np.zeros( (labels_raw.shape[0],1), dtype=np.float32 )
        #print('anchor_weight', anchor_weight.shape)

        #assert labels.shape[0]==1
        #assert cls_score.shape[0]==1
        #assert bbox_weight.shape[0]==1
        #print('shape', cls_score.shape, labels.shape, file=sys.stderr)
        #print('bbox_weight 0', bbox_weight.shape, file=sys.stderr)
        #bbox_weight = np.zeros( (labels_raw.shape[0], labels_raw.shape[1], 4), dtype=np.float32)
        for ibatch in xrange(labels_raw.shape[0]):
          _anchor_weight = np.zeros( (labels_raw.shape[1],1), dtype=np.float32)
          labels = labels_raw[ibatch]
          bg_score = cls_score[ibatch,0,:]

          n_fg = np.sum(labels>0)
          fg_inds = np.where(labels>0)[0]
          #num_bg = max(10, num_fg*int(1.0/config.TRAIN.RPN_FG_FRACTION-1))
          #if self.mode==2:
          #  num_bg = num_fg*int(1.0/config.TRAIN.RPN_FG_FRACTION-1)

          bg_inds = np.where(labels == 0)[0]
          origin_num_bg = len(bg_inds)

          sampled_inds = list(set(np.where(bg_score.ravel() < 0.99)[0]).intersection(set(bg_inds)))
          #print(np.max(bg_score))
          #print('sampled_inds_bg', sampled_inds, file=sys.stderr)
          labels[bg_inds] = -1
          labels[sampled_inds] = 0

          if n_fg>0:
            order0_labels = labels.reshape( (1, A, -1) ).transpose( (0, 2, 1) ).reshape( (-1,) )
            bbox_fg_inds = np.where(order0_labels>0)[0]
            #print('bbox_fg_inds, order0 ', bbox_fg_inds, file=sys.stderr)
            _anchor_weight[bbox_fg_inds,:] = 1.0
          anchor_weight[ibatch] = _anchor_weight
          valid_count[ibatch][0] = n_fg

        for ind, val in enumerate([labels_raw, anchor_weight, valid_count]):
            val = mx.nd.array(val)
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('STC_op')
class STCProp(mx.operator.CustomOpProp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(STCProp, self).__init__(need_top_grad=False)
        self.stride = stride
        self.network=network
        self.dataset=dataset
        self.prefix = prefix

    def list_arguments(self):
        return ['cls_score', 'labels']

    def list_outputs(self):
        return ['labels_STC', 'anchor_weight', 'valid_count']

    def infer_shape(self, in_shape):
        labels_shape = in_shape[1]
        #print('in_rpn_ohem', in_shape[0], in_shape[1], in_shape[2], file=sys.stderr)
        anchor_weight_shape = [labels_shape[0], labels_shape[1], 1]
        #print('in_rpn_ohem', labels_shape, anchor_weight_shape)

        return in_shape, \
               [labels_shape, anchor_weight_shape, [labels_shape[0], 1]]

    def create_operator(self, ctx, shapes, dtypes):
        return STCOperator(self.stride, self.network, self.dataset, self.prefix)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


