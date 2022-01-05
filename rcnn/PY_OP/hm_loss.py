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
import numpy as np
import mxnet.autograd as ag
import mxnet.ndarray as nd

class hm_focal_loss(mx.operator.CustomOp):
    def __init__(self):
        super(hm_focal_loss, self).__init__()
        self.grad = None
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        #x = nd.clip(nd.sigmoid(x), a_min=1e-14, a_max=1 - 1e-4)
        label = in_data[1]
        x.attach_grad()
        with ag.record():
            loss = self.function(x,label)
        loss.backward()
        self.grad = x.grad
        self.assign(out_data[0], req[0], mx.nd.array(loss))

    def function(self,pred, gt):
        pos_inds = gt.__eq__(1).astype('float32')
        neg_inds = gt.__lt__(1).astype('float32')
        # print(pos_inds.sum())
        # print(neg_inds.sum())
        #pred = nd.clip(nd.sigmoid(pred), a_min=1e-14, a_max=1 - 1e-4)
        neg_weights = nd.power(1 - gt, 4)

        loss = 0
        pos_loss = nd.log(pred) * nd.power(1 - pred, 2) * pos_inds
        neg_loss = nd.log(1 - pred) * nd.power(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.astype('float32').sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # print('pos_loss%s'%pos_loss)
        # print('neg_loss%s'%neg_loss)
        # print('loss%s'%(((pos_loss + neg_loss) / num_pos)))
        if num_pos == 0:
            # print('00000000000000')
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], self.grad)

@mx.operator.register("hm_focal_loss")
class hm_focal_lossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(hm_focal_lossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [[output_shape[1]]]

    def infer_type(self, in_type):
        return in_type, [in_type[0]]

    def create_operator(self, ctx, shapes, dtypes):
        return hm_focal_loss()