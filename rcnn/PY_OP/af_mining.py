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
from ce_det_func.decoder import _nms, _topk
from mxnet import nd
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, bbox_transform, landmark_transform, landmark_transform_xyxy, landmark_transform_af

class afminingOperator(mx.operator.CustomOp):
    def __init__(self, stride=8, thresh = 0.7, topk = 3):
        super(afminingOperator, self).__init__()
        self.stride = int(stride)
        self.thresh = thresh
        self.topk = topk
        self.feats = None
        self.num_anchors = None
        # generate_config(network, dataset)

    def anchor_infos(self, isize = 640):
        sstride = str(self.stride)
        self.feats = config.RPN_FEAT_STRIDE
        base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
        ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
        scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
        base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios),
                                        scales=np.array(scales, dtype=np.float32), stride=self.stride,
                                        dense_anchor=config.DENSE_ANCHOR)
        num_anchors = base_anchors.shape[0]
        self.num_anchors = num_anchors
        feat_width = isize // self.stride
        feat_height = feat_width
        feat_stride = self.stride

        A = num_anchors
        K = feat_height * feat_width
        all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors

    # def ind_loc(self,isize = 640): # get start index in labels
    #
    #     ind_tmp = 0
    #     for s in self.feats:
    #         if s != self.stride:
    #             feat_height1= isize // self.stride
    #             feat_width1 = feat_height1
    #             K1 = feat_height1 * feat_width1
    #             ind_tmp += K1 * self.num_anchors
    #         else:
    #             break
    #     return ind_tmp


    def forward(self, is_train, req, in_data, out_data, aux):
        anchors = self.anchor_infos()
        hm_pred = in_data[0].asnumpy()
        wh_pred = in_data[1].asnumpy()
        offset_pred = in_data[2].asnumpy()
        gt_boxes = in_data[3].asnumpy()
        # overlaps = in_data[4].asnumpy()
        labels = in_data[4].asnumpy()
        bbox_weight = in_data[5].transpose((0,2,1)).reshape((0,-1,4)).asnumpy()
        # gtind = [len(np.where(gt_boxes[ibatch,:,4] >= 0)[]) for ibatch in range(hm_pred.shape[0])]
        # print(gtind)
        for ibatch in xrange(hm_pred.shape[0]):
            #process ground truth
            _gt_boxes = gt_boxes[ibatch]
            # _overlaps = overlaps[ibatch]
            _gtind = len(np.where(_gt_boxes[:, 4] >= 0)[0])
            if _gtind > 0:
                _gt_boxes = _gt_boxes[0:_gtind, :]
                # _overlaps = _overlaps[:,0:_gtind]
                _wh_pred = wh_pred[ibatch:(ibatch+1)]
                _hm_pred = hm_pred[ibatch:(ibatch+1)]
                _offset_pred = offset_pred[ibatch:(ibatch+1)]
                _hm_pred = _nms(nd.array(_hm_pred)).asnumpy()
                _label = labels[ibatch]
                _bbox_weight = bbox_weight[ibatch]
                ct_k = 1000
                ct_scores, ct_inds, clses, ys, xs = _topk(nd.array(_hm_pred), K=ct_k)
                ct_inds = ct_inds[0]
                wh_pred_1 = np.zeros((ct_k, 2)).astype(np.float32)
                # offset = np.zeros((ct_k, 2)).astype(np.float32)
                for i in range(ys.shape[0]):
                    wh_pred_1[i, :] = _wh_pred[:, :, ys[i, 0].astype(int), xs[i, 0].astype(int)].reshape(2)
                    # offset[i, :] = _offset_pred[:, :, ys[i, 0].astype(int), xs[i, 0].astype(int)].reshape(2)
                # if config.ct_offset:
                #     ys = ys + offset[:, 0:1]
                #     xs = xs + offset[:, 1:2]

                ct_threshold = self.thresh
                # print(ct_threshold)
                # print(np.where(ct_scores >= ct_threshold))
                area_pred = np.zeros((ct_k, 1)).astype(np.float32)
                area_pred[:,0:1] = wh_pred_1[:,0:1] * wh_pred_1[:,1:2]
                inx0 = np.where(ct_scores >= ct_threshold)[1]
                inx1 = np.where(area_pred <= 128)[0]
                inx = list(set(inx0).intersection(set(inx1)))
                # print(inx)
                if len(inx) > 0:
                    # ct_bboxes = ct_bboxes[inx, :] * self.stride
                    ct_scores = np.reshape(ct_scores, (ct_k, 1))
                    ct_scores = ct_scores[inx, :]
                    ct_inds = ct_inds[inx]
                    count = 0
                    for i in ct_inds:
                        if i >= 81: # handle offset between af and ab
                            i -= 81
                        else:
                            continue
                        # print(i)
                        ind = list()
                        # for j in range(self.num_anchors): # for every anchor, compute its index
                        ind.append(i * self.num_anchors)
                        # print(_label[ind])
                        if _label[ind] == 0:
                            # print('mining a face')
                            # print([ibatch,i+81])
                            count+=1
                        _label[ind] = 1
                        # print(ind)
                        _bbox_weight[ind, 0:4] = 1.0
                    print('mining %d faces in batch %d' %(count, ibatch))
        # bbox_weight = bbox_weight.reshape(0,-1,4*self.num_anchors).transpose(0,2,1)


        for ind, val in enumerate([labels, bbox_weight]):
            val = mx.nd.array(val)
            if ind > 0:
                val = val.reshape((0,-1,4*self.num_anchors)).transpose((0,2,1))
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('af_mining')
class afminingProp(mx.operator.CustomOpProp):
    def __init__(self, stride=8, thresh = 0.7, topk = 3):
        super(afminingProp, self).__init__(need_top_grad=False)
        self.stride = stride
        self.thresh = thresh
        self.topk = topk
        self.feats = None
        self.num_anchors = None

    def list_arguments(self):
        return ['hm_pred', 'wh_pred', 'offset_pred', 'gt_boxes', 'labels', 'bbox_weight']

    def list_outputs(self):
        return ['labels', 'bbox_weight']

    def infer_shape(self, in_shape):
        labels_shape = in_shape[4]
        #print('in_rpn_ohem', in_shape[0], in_shape[1], in_shape[2], file=sys.stderr)
        anchor_weight_shape = in_shape[5]
        #print('in_rpn_ohem', labels_shape, anchor_weight_shape)

        return in_shape, \
               [labels_shape, anchor_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return afminingOperator(self.stride, self.thresh, self.topk)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


