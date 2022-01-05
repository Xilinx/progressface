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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:31:48 2018

@author: Guo Shuangshuang
"""
import mxnet as mx
import numpy as np


class refine_anchor_generator(mx.operator.CustomOp):
    def __init__(self, ):
        super(refine_anchor_generator, self).__init__()


    def forward(self, is_train, req, in_data, out_data, aux):

        print('refine')
        print(in_data[0].shape, in_data[1].shape)
        batch_size = in_data[1].shape[0]
        arm_anchor_boxes = in_data[0]
        arm_loc_preds = in_data[1]
        landmarks_target_preds = in_data[2]

        arm_anchor_boxes = mx.nd.concatenate([arm_anchor_boxes]*batch_size, axis=0)
        #landmarks_target_preds = mx.nd.concatenate([landmarks_target_preds]*batch_size, axis=0)
        #landmarks_target_preds = mx.nd.reshape(data=landmarks_target_preds, shape=(0,-1,5,2))

        arm_anchor_boxes_bs = mx.nd.split(data=arm_anchor_boxes, axis=2, num_outputs=4)

        al = arm_anchor_boxes_bs[0]
        at = arm_anchor_boxes_bs[1]
        ar = arm_anchor_boxes_bs[2]
        ab = arm_anchor_boxes_bs[3]
        aw = ar - al
        ah = ab - at
        ax = (al + ar) / 2.0
        ay = (at + ab) / 2.0
        arm_loc_preds = mx.nd.Reshape(data=arm_loc_preds, shape=(0, -1, 4))
        arm_loc_preds_bs = mx.nd.split(data=arm_loc_preds, axis=2, num_outputs=4)
        ox_preds = arm_loc_preds_bs[0]
        oy_preds = arm_loc_preds_bs[1]
        ow_preds = arm_loc_preds_bs[2]
        oh_preds = arm_loc_preds_bs[3]
        ox = ox_preds * aw + ax
        oy = oy_preds * ah + ay
        ow = mx.nd.exp(ow_preds) * aw/2
        oh = mx.nd.exp(oh_preds) * ah/2

        # pred = landmarks_target_preds.copy()[0,:,:,:]
        # for i in range(5):
        #     pred[:, i, 0] = landmarks_target_preds[0,:,i,0] * aw + ax
        #     pred[:, i, 1] = landmarks_target_preds[0,:,i,1] * ah + ay
        '''
        arm_loc_mask = mx.nd.reshape(data=arm_loc_mask,shape=(0,-1,4))
 #       cond = arm_cls_target >= arm_cls_mask
        cond = arm_loc_mask[:,:,0]
        #print(mx.ndarray.sum(data=cond, axis=1))
#        print(cond.shape)

        cond = mx.nd.reshape(data=cond,shape=(0,0,1))
        out0 = mx.nd.where(condition=cond, x=ox-ow, y=al)
        out1 = mx.nd.where(condition=cond, x=oy - oh, y=at)
        out2 = mx.nd.where(condition=cond, x=ox + ow, y=ar)
        out3 = mx.nd.where(condition=cond, x=oy + oh, y=ab)
        '''
        out0 = ox - ow
        out1 = oy - oh
        out2 = ox + ow
        out3 = oy + oh

        refine_anchor = mx.nd.concat(out0, out1, out2, out3, dim=2)
        #print('arm', arm_anchor_boxes[0,0:10,:])
        #print('odm', refine_anchor[0,0:10,:])

        self.assign(out_data[0], req[0], refine_anchor)
        #self.assign(out_data[1], req[1], pred)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)




@mx.operator.register("refine_anchor_generator")
class refine_anchor_generatorProp(mx.operator.CustomOpProp):
    def __init__(self,):
        super(refine_anchor_generatorProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['arm_anchor_boxes', 'arm_loc_preds','landmarks_target_preds']

    def list_outputs(self):
        return ['refine_anchor_boxes']

    def infer_shape(self, in_shape):
        arm_anchor_boxes_shape = in_shape[0]
        arm_loc_preds_shape = in_shape[1]
        landmarks_target_preds_shape = in_shape[2]

        #arm_loc_mask_shape = in_shape[2]
        out_shape = [arm_anchor_boxes_shape[0],arm_anchor_boxes_shape[1],arm_anchor_boxes_shape[2]]

        return [arm_anchor_boxes_shape, arm_loc_preds_shape,landmarks_target_preds_shape], [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return refine_anchor_generator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
