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

# --------------------------------------------------------
# KL loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by jiashu https://github.com/jiashu-zhu/
# --------------------------------------------------------

"""
KL loss
"""

import mxnet as mx
import numpy as np


class KLLossOperator(mx.operator.CustomOp):
    def __init__(self):
        super(KLLossOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        bbox_diff = in_data[0].asnumpy()
        self.bbox_diff = bbox_diff
        #print(np.max(bbox_diff))
        alpha = in_data[1].asnumpy()
        self.alpha = alpha
        #print(np.max(alpha))

        loss = np.zeros((bbox_diff.shape[0],1))
        for ibatch in range(bbox_diff.shape[0]):
            _bbox_diff = bbox_diff[ibatch]
            _alpha = alpha[ibatch]
            bbox_diff_abs = np.abs(_bbox_diff)
            bbox_diff_sqr = np.power(_bbox_diff,2)
            eq10_where = np.where(bbox_diff_abs > 1)
            eq9_where = np.where(bbox_diff_abs <=1)
            eq10_label = np.zeros_like(bbox_diff_abs)
            eq10_label[eq10_where[0], eq10_where[1]] = 1
            eq9_label = np.zeros_like(bbox_diff_abs)
            eq9_label[eq9_where[0], eq9_where[1]] = 1

            eq9_result = bbox_diff_sqr * eq9_label
            eq9_result /= 2
            eq10_result = bbox_diff_abs * 0.5
            eq10_result = eq10_result * eq10_label
            result1 = eq9_result + eq10_result

            #print(_alpha)
            # idx = np.where(_alpha < -1)
            # _alpha[idx[0], idx[1]] = -1
            alpha_log = _alpha * 0.5
            alpha_neg = _alpha * -1
            alpha_exp = np.exp(alpha_neg)
            #print(alpha_exp)
            result1 = result1 * alpha_exp

            alpha_log_meanr = alpha_log - np.mean(alpha_log, axis=1,keepdims=True)
            log_loss = np.sum(alpha_log_meanr)

            result1_meanr = result1 - np.mean(result1, axis=1,keepdims=True)
            mul_loss = np.sum(result1_meanr)

            klloss = log_loss + mul_loss
            loss[ibatch] = klloss

        #print(loss)
        self.assign(out_data[0], req[0], mx.nd.array(loss))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        out0 = np.zeros_like(self.bbox_diff)
        out1 = np.zeros_like(self.alpha)

        for ibatch in range(self.bbox_diff.shape[0]):
            _bbox_diff = self.bbox_diff[ibatch]

            _alpha = self.Normalize(self.alpha[ibatch])
            bbox_diff_abs = np.abs(_bbox_diff)
            bbox_diff_sqr = np.power(_bbox_diff, 2)
            eq10_where = np.where(bbox_diff_abs > 1)
            eq9_where = np.where(bbox_diff_abs <= 1)
            eq10_label = np.zeros_like(bbox_diff_abs)
            eq10_label[eq10_where[0], eq10_where[1]] = 1
            eq9_label = np.zeros_like(bbox_diff_abs)
            eq9_label[eq9_where[0], eq9_where[1]] = 1

            grad_alpha_9 = (-1 * 0.5* np.exp(_alpha * -1) * bbox_diff_sqr + 0.5) * eq9_label
            grad_alpha_10 = (-1 * np.exp(_alpha * -1)*(bbox_diff_abs - 0.5) + 0.5) * eq10_label
            grad_alpha = grad_alpha_9 + grad_alpha_10
            out1[ibatch] = grad_alpha/(grad_alpha.shape[1]*10)

            grad_bbox_9 = (-1 *  np.exp(_alpha * -1) * _bbox_diff) * eq9_label
            grad_bbox_10 = (np.exp(_alpha * -1) * bbox_diff_abs) * eq10_label
            grad_bbox = grad_bbox_9 + grad_bbox_10
            out0[ibatch] = grad_bbox/(grad_bbox.shape[1]*10)
        # print(np.max(out0))
        # print(np.max(out1))
        print(out0.shape)
        print(out1.shape)
        self.assign(in_grad[0], req[0], mx.nd.array(out0))
        self.assign(in_grad[1], req[1], mx.nd.array(out1))

    def Normalize(self,data):
        m = np.mean(data)
        mx = np.max(data)
        mn = np.min(data)
        if mx -mn == 0:
            return data - m
        else:
            return (data - m)/(mx - mn)


@mx.operator.register('KLLoss')
class KLLossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(KLLossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['bbox_diff', 'alpha']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = [in_shape[0][0],1]
        return [data_shape, labels_shape], [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return KLLossOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []