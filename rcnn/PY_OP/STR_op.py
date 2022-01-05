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
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, bbox_transform, landmark_transform
import numpy.random as npr

STAT = {0:0}
STEP = 28800

class STROperator(mx.operator.CustomOp):
    def __init__(self,):
        super(STROperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        landmark = config.FACE_LANDMARK
        bbox_pred_len = 4
        landmark_pred_len = 10
        feat_strides = config.RPN_FEAT_STRIDE
        prefix = 'face'

        anchors = in_data[0].asnumpy()
        labels = np.empty(anchors.shape[0])
        gt_boxes = in_data[1].asnumpy()
        gt_landmarks = in_data[2].asnumpy()
        overlaps = self.bbox_overlaps_py(anchors,gt_boxes)
        print(overlaps.shape)
        argmax_overlaps = overlaps.argmax(axis=1)  # compute max overlaps for every bounding box
        # print('AAA', argmax_overlaps.shape)
        max_overlaps = overlaps[np.arange(anchors.shape[0]), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)  # compute max overmaps for every ground truth
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        if config.TRAIN.RPN_FORCE_POSITIVE:
          labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        if config.TRAIN.RPN_ENABLE_OHEM == 0:
            fg_inds = np.where(labels == 1)[0]
            num_fg = len(fg_inds)
            num_bg = num_fg*int(1.0/config.TRAIN.RPN_FG_FRACTION-1)
            bg_inds = np.where(labels == 0)[0]
            if len(bg_inds) > num_bg:
               disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
               labels[disable_inds] = -1
        else:
            fg_inds = np.where(labels == 1)[0]
            num_fg = len(fg_inds)
            bg_inds = np.where(labels == 0)[0]
            num_bg = len(bg_inds)
        bbox_targets = np.zeros((anchors.shape[0], bbox_pred_len), dtype=np.float32)
        if gt_boxes.size > 0:
            # print('GT', gt_boxes.shape, gt_boxes[argmax_overlaps, :4].shape)
            bbox_targets[:, :] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])
            # bbox_targets[:,4] = gt_blur
        # tb = datetime.datetime.now()
        # self._times[1] += (tb-ta).total_seconds()
        # ta = datetime.datetime.now()

        bbox_weights = np.zeros((anchors.shape[0], bbox_pred_len), dtype=np.float32)
        # bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)
        bbox_weights[labels == 1, 0:4] = 1.0
        if bbox_pred_len > 4:
            bbox_weights[labels == 1, 4:bbox_pred_len] = 0.1

        if config.FACE_LANDMARK:
            landmark_targets = np.zeros((anchors.shape[0], landmark_pred_len), dtype=np.float32)
            # landmark_weights = np.zeros((len(inds_inside), 10), dtype=np.float32)
            landmark_weights = np.zeros((anchors.shape[0], landmark_pred_len), dtype=np.float32)
            # landmark_weights[labels == 1, :] = np.array(config.TRAIN.RPN_LANDMARK_WEIGHTS)
            if landmark_pred_len == 10:
                landmark_weights[labels == 1, :] = 1.0
            elif landmark_pred_len == 15:
                v = [1.0, 1.0, 0.1] * 5
                assert len(v) == 15
                landmark_weights[labels == 1, :] = np.array(v)
            else:
                assert False
            # TODO here
            if gt_landmarks.size > 0:
                # print('AAA',argmax_overlaps)
                a_landmarks = gt_landmarks[argmax_overlaps, :, :]
                landmark_targets[:] = landmark_transform(anchors, a_landmarks)
                invalid = np.where(a_landmarks[:, 0, 2] < 0.0)[0]
                # assert len(invalid)==0
                # landmark_weights[invalid, :] = np.array(config.TRAIN.RPN_INVALID_LANDMARK_WEIGHTS)
                landmark_weights[invalid, :] = 0.0

        anchors_num_list = []
        A_list = []
        feat_infos = []
        for stride in feat_strides:
            sstride = str(stride)
            scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
            A_list.append(len(scales))
            feat_infos.append((640/stride, 640/stride))
            anchors_num_list.append(640/stride * 640/stride * len(scales))
        # resahpe
        label_list = list()
        bbox_target_list = list()
        bbox_weight_list = list()
        if config.FACE_LANDMARK:
            landmark_target_list = list()
            landmark_weight_list = list()
        anchors_num_range = [0] + anchors_num_list
        label = {}
        for i in range(len(feat_strides)):
            stride = feat_strides[i]
            feat_height, feat_width = feat_infos[i]
            A = A_list[i]
            _label = labels[sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[i + 1]]
            # print('_label', _label.shape, select_stride)
            # _fg_inds = np.where(_label == 1)[0]
            # n_fg = len(_fg_inds)
            # STAT[0]+=1
            # STAT[stride]+=n_fg
            # if STAT[0]%100==0:
            #  print('rpn_stat', STAT, file=sys.stderr)
            bbox_target = bbox_targets[
                          sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[i + 1]]
            bbox_weight = bbox_weights[
                          sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[i + 1]]
            if landmark:
                landmark_target = landmark_targets[
                                  sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[
                                      i + 1]]
                landmark_weight = landmark_weights[
                                  sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[
                                      i + 1]]

            _label = _label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
            _label = _label.reshape((1, A * feat_height * feat_width))
            bbox_target = bbox_target.reshape((1, feat_height * feat_width, A * bbox_pred_len)).transpose(0, 2, 1)
            bbox_weight = bbox_weight.reshape((1, feat_height * feat_width, A * bbox_pred_len)).transpose((0, 2, 1))
            label['%s_label_stride%d' % (prefix, stride)] = _label
            label['%s_bbox_target_stride%d' % (prefix, stride)] = bbox_target
            label['%s_bbox_weight_stride%d' % (prefix, stride)] = bbox_weight
            if config.FACE_LANDMARK:
                landmark_target = landmark_target.reshape(
                    (1, feat_height * feat_width, A * landmark_pred_len)).transpose(0, 2, 1)
                landmark_weight = landmark_weight.reshape(
                    (1, feat_height * feat_width, A * landmark_pred_len)).transpose((0, 2, 1))
                label['%s_landmark_target_stride%d' % (prefix, stride)] = landmark_target
                label['%s_landmark_weight_stride%d' % (prefix, stride)] = landmark_weight
            # print('in_rpn', stride,_label.shape, bbox_target.shape, bbox_weight.shape, file=sys.stderr)
            label_list.append(_label)
            # print('DD', _label.shape)
            bbox_target_list.append(bbox_target)
            bbox_weight_list.append(bbox_weight)
            if landmark:
                landmark_target_list.append(landmark_target)
                landmark_weight_list.append(landmark_weight)

        label_concat = np.concatenate(label_list, axis=1)
        bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
        bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)
        # fg_inds = np.where(label_concat[0] == 1)[0]
        # print('fg_inds_in_rpn2', fg_inds, file=sys.stderr)

        label.update({'%s_label' % prefix: label_concat,
                      '%s_bbox_target' % prefix: bbox_target_concat,
                      '%s_bbox_weight' % prefix: bbox_weight_concat}
                     )
        if landmark:
            landmark_target_concat = np.concatenate(landmark_target_list, axis=2)
            landmark_weight_concat = np.concatenate(landmark_weight_list, axis=2)
            label['%s_landmark_target' % prefix] = landmark_target_concat
            label['%s_landmark_weight' % prefix] = landmark_weight_concat
        for key in label:
            label[key] = mx.nd.array(label[key])
        rst = []
        rst.append(label[key] for key in label)
        self.assign(out_data[0], req[0], rst)



    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

    def bbox_overlaps_py(self, boxes, query_boxes):
        """
        determine overlaps between boxes and query_boxes
        :param boxes: n * 4 bounding boxes
        :param query_boxes: k * 4 bounding boxes
        :return: overlaps: n * k overlaps
        """
        n_ = boxes.shape[0]
        k_ = query_boxes.shape[0]
        overlaps = np.zeros((n_, k_), dtype=np.float)
        for k in range(k_):
            query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            for n in range(n_):
                iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
                if iw > 0:
                    ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                    if ih > 0:
                        box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                        all_area = float(box_area + query_box_area - iw * ih)
                        overlaps[n, k] = iw * ih / all_area
        return overlaps


@mx.operator.register('STR_op')
class STRProp(mx.operator.CustomOpProp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(STRProp, self).__init__(need_top_grad=False)
        self.stride = stride
        self.network=network
        self.dataset=dataset
        self.prefix = prefix

    def list_arguments(self):
        return ['anchors', 'gt_boxes', 'gt_landmarks']

    def list_outputs(self):
        return ['rst']

    def infer_shape(self, in_shape):
        labels_shape = in_shape[1]
        #print('in_rpn_ohem', in_shape[0], in_shape[1], in_shape[2], file=sys.stderr)
        anchor_weight_shape = [labels_shape[0], labels_shape[1], 1]
        #print('in_rpn_ohem', labels_shape, anchor_weight_shape)

        return in_shape, \
               [labels_shape, anchor_weight_shape, [labels_shape[0], 1]]

    def create_operator(self, ctx, shapes, dtypes):
        return STROperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


