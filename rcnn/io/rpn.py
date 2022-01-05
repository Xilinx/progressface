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

"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

from __future__ import print_function
import sys
import logging
import datetime
import numpy as np
import numpy.random as npr
import mxnet as mx
import math
from ..logger import logger
from ..config import config
from ..logger import logger
from .image import get_image, tensor_vstack, get_crop_image
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, bbox_transform, landmark_transform, landmark_transform_xyxy, landmark_transform_af
sys.path.append('../../')
from ce_det_func.gaussian import *
from mxnet import nd
STAT = {0:0, 8:0, 16:0, 32:0}

def get_rpn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {}

    return data, label, im_info

def get_rpn_batch(roidb):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    # gt boxes: (x1, y1, x2, y2, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_boxes': gt_boxes}

    return data, label

def get_crop_batch(roidb):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    #assert len(roidb) == 1, 'Single batch only'
    data_list = []
    label_list = []
    imgs, roidb = get_crop_image(roidb)
    assert len(imgs)==len(roidb)
    for i in range(len(imgs)):
      im_array = imgs[i]
      im_info = np.array([roidb[i]['im_info']], dtype=np.float32)

      # gt boxes: (x1, y1, x2, y2, cls)
      if roidb[i]['gt_classes'].size > 0:
          gt_inds = np.where(roidb[i]['gt_classes'] != 0)[0]
          gt_boxes = np.empty((roidb[i]['boxes'].shape[0], 5), dtype=np.float32)
          gt_boxes[:, 0:4] = roidb[i]['boxes'][gt_inds, :]
          gt_boxes[:, 4] = roidb[i]['gt_classes'][gt_inds]
          if config.centernet_branch:
              gt_centers = np.empty((roidb[i]['boxes'].shape[0], 2), dtype=np.float32)
              gt_centers[:,0] = (gt_boxes[:,2] + gt_boxes[:,0])/2
              gt_centers[:,1] = (gt_boxes[:,3] + gt_boxes[:,1])/2
          if config.USE_BLUR:
            gt_blur = roidb[i]['blur']
          if config.FACE_LANDMARK:
            #gt_landmarks = np.empty((roidb[i]['landmarks'].shape[0], 11), dtype=np.float32)
            gt_landmarks = roidb[i]['landmarks'][gt_inds,:,:]
          if config.HEAD_BOX:
            gt_boxes_head = np.empty((roidb[i]['boxes_head'].shape[0], 5), dtype=np.float32)
            gt_boxes_head[:, 0:4] = roidb[i]['boxes_head'][gt_inds, :]
            gt_boxes_head[:, 4] = roidb[i]['gt_classes'][gt_inds]
      else:
          gt_boxes = np.empty((0, 5), dtype=np.float32)
          if config.USE_BLUR:
            gt_blur = np.empty((0,), dtype=np.float32)
          if config.FACE_LANDMARK:
            gt_landmarks = np.empty((0, 5, 3), dtype=np.float32)
          if config.HEAD_BOX:
            gt_boxes_head = np.empty((0, 5), dtype=np.float32)
          if config.centernet_branch:
            gt_centers = np.empty((0,2),dtype=np.float32)

      data = {'data': im_array,
              'im_info': im_info}
      label = {'gt_boxes': gt_boxes}
      if config.USE_BLUR:
        label['gt_blur'] = gt_blur
      if config.FACE_LANDMARK:
        label['gt_landmarks'] = gt_landmarks
      if config.HEAD_BOX:
        label['gt_boxes_head'] = gt_boxes_head
      if config.centernet_branch:
        label['gt_centers'] = gt_centers
      data_list.append(data)
      label_list.append(label)

    return data_list, label_list

def assign_anchor_fpn(feat_shape, gt_label, im_info, landmark=False, prefix='face', select_stride=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :return: tuple
    labels: of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    bbox_targets: of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    bbox_weights: mark the assigned anchors
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    global STAT
    DEBUG = False

    im_info = im_info[0]
    gt_boxes = gt_label['gt_boxes']
    # clean up boxes
    nonneg = np.where(gt_boxes[:, 4] != -1)[0]
    gt_boxes = gt_boxes[nonneg]
    if config.USE_BLUR:
      gt_blur = gt_label['gt_blur']
      gt_blur = gt_blur[nonneg]
    if landmark:
      gt_landmarks = gt_label['gt_landmarks']
      gt_landmarks = gt_landmarks[nonneg]
      assert gt_boxes.shape[0]==gt_landmarks.shape[0]
    #scales = np.array(scales, dtype=np.float32)
    feat_strides = config.RPN_FEAT_STRIDE
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
      gt_boxes[:,4] = gt_blur
      bbox_pred_len = 5
    if config.USE_OCCLUSION:
      landmark_pred_len = 15

    anchors_list = []
    anchors_num_list = []
    inds_inside_list = []
    feat_infos = []
    A_list = []
    for i in range(len(feat_strides)):
        stride = feat_strides[i]
        sstride = str(stride)
        base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
        allowed_border = config.RPN_ANCHOR_CFG[sstride]['ALLOWED_BORDER']
        ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
        scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
        base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, dtype=np.float32), stride = stride, dense_anchor = config.DENSE_ANCHOR)
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shape[i][-2:]
        feat_stride = feat_strides[i]
        feat_infos.append([feat_height, feat_width])

        A = num_anchors
        A_list.append(A)
        K = feat_height * feat_width

        all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))
        #print('anchor0', stride, all_anchors[0])

        total_anchors = int(K * A)
        anchors_num_list.append(total_anchors)
        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        #print('AA', anchors.shape, len(inds_inside))

        anchors_list.append(anchors)
        inds_inside_list.append(inds_inside)

    # Concat anchors from each level
    anchors = np.concatenate(anchors_list)
    for i in range(1, len(inds_inside_list)):
        inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])
    inds_inside = np.concatenate(inds_inside_list)
    total_anchors = sum(anchors_num_list)
    #print('total_anchors', anchors.shape[0], len(inds_inside), file=sys.stderr)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)
    #print('BB', anchors.shape, len(inds_inside))
    #print('gt_boxes', gt_boxes.shape, file=sys.stderr)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        #print('AAA', argmax_overlaps.shape)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
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
    else:
        labels[:] = 0
    fg_inds = np.where(labels == 1)[0]
    #print('fg count', len(fg_inds))

    # subsample positive labels if we have too many
    if config.TRAIN.RPN_ENABLE_OHEM==0:
      fg_inds = np.where(labels == 1)[0]
      num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
      if len(fg_inds) > num_fg:
          disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
          if DEBUG:
              disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
          labels[disable_inds] = -1

      # subsample negative labels if we have too many
      num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
      bg_inds = np.where(labels == 0)[0]
      if len(bg_inds) > num_bg:
          disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
          if DEBUG:
              disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
          labels[disable_inds] = -1

      #fg_inds = np.where(labels == 1)[0]
      #num_fg = len(fg_inds)
      #num_bg = num_fg*int(1.0/config.TRAIN.RPN_FG_FRACTION-1)

      #bg_inds = np.where(labels == 0)[0]
      #if len(bg_inds) > num_bg:
      #    disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
      #    if DEBUG:
      #        disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
      #    labels[disable_inds] = -1
    else:
      fg_inds = np.where(labels == 1)[0]
      num_fg = len(fg_inds)
      bg_inds = np.where(labels == 0)[0]
      num_bg = len(bg_inds)

    #print('anchor stat', num_fg, num_bg)


    bbox_targets = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
    if gt_boxes.size > 0:
        #print('GT', gt_boxes.shape, gt_boxes[argmax_overlaps, :4].shape)
        bbox_targets[:,:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])
        #bbox_targets[:,4] = gt_blur

    bbox_weights = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
    #bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)
    bbox_weights[labels == 1, 0:4] = 1.0
    if bbox_pred_len>4:
      bbox_weights[labels == 1, 4:bbox_pred_len] = 0.1

    if landmark:
      landmark_targets = np.zeros((len(inds_inside), landmark_pred_len), dtype=np.float32)
      #landmark_weights = np.zeros((len(inds_inside), 10), dtype=np.float32)
      landmark_weights = np.zeros((len(inds_inside), landmark_pred_len), dtype=np.float32)
      #landmark_weights[labels == 1, :] = np.array(config.TRAIN.RPN_LANDMARK_WEIGHTS)
      if landmark_pred_len==10:
        landmark_weights[labels == 1, :] = 1.0
      elif landmark_pred_len==15:
        v = [1.0, 1.0, 0.1] * 5
        assert len(v)==15
        landmark_weights[labels == 1, :] = np.array(v)
      else:
        assert False
      #TODO here
      if gt_landmarks.size > 0:
        #print('AAA',argmax_overlaps)
        a_landmarks = gt_landmarks[argmax_overlaps,:,:]
        landmark_targets[:] = landmark_transform(anchors, a_landmarks)

        invalid = np.where(a_landmarks[:,0,2]<0.0)[0]
        #assert len(invalid)==0
        #landmark_weights[invalid, :] = np.array(config.TRAIN.RPN_INVALID_LANDMARK_WEIGHTS)
        landmark_weights[invalid, :] = 0.0

    #if DEBUG:
    #    _sums = bbox_targets[labels == 1, :].sum(axis=0)
    #    _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
    #    _counts = np.sum(labels == 1)
    #    means = _sums / (_counts + 1e-14)
    #    stds = np.sqrt(_squared_sums / _counts - means ** 2)
    #    print 'means', means
    #    print 'stdevs', stds
    # map up to original set of anchors
    #print(labels.shape, total_anchors, inds_inside.shape, inds_inside[0], inds_inside[-1])
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)
    if landmark:
      landmark_targets = _unmap(landmark_targets, total_anchors, inds_inside, fill=0)
      landmark_weights = _unmap(landmark_weights, total_anchors, inds_inside, fill=0)
    #print('CC', anchors.shape, len(inds_inside))

    #if DEBUG:
    #    if gt_boxes.size > 0:
    #        print 'rpn: max max_overlaps', np.max(max_overlaps)
    #    print 'rpn: num_positives', np.sum(labels == 1)
    #    print 'rpn: num_negatives', np.sum(labels == 0)
    #    _fg_sum = np.sum(labels == 1)
    #    _bg_sum = np.sum(labels == 0)
    #    _count = 1
    #    print 'rpn: num_positive avg', _fg_sum / _count
    #    print 'rpn: num_negative avg', _bg_sum / _count

    # resahpe
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    if landmark:
      landmark_target_list = list()
      landmark_weight_list = list()
    anchors_num_range = [0] + anchors_num_list
    label = {}
    for i in range(len(feat_strides)):
        stride = feat_strides[i]
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        _label = labels[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        if select_stride>0 and stride!=select_stride:
          #print('set', stride, select_stride)
          _label[:] = -1
        #print('_label', _label.shape, select_stride)
        #_fg_inds = np.where(_label == 1)[0]
        #n_fg = len(_fg_inds)
        #STAT[0]+=1
        #STAT[stride]+=n_fg
        #if STAT[0]%100==0:
        #  print('rpn_stat', STAT, file=sys.stderr)
        bbox_target = bbox_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        bbox_weight = bbox_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        if landmark:
          landmark_target = landmark_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
          landmark_weight = landmark_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]

        _label = _label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        _label = _label.reshape((1, A * feat_height * feat_width))
        bbox_target = bbox_target.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose(0, 2, 1)
        bbox_weight = bbox_weight.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose((0, 2, 1))
        label['%s_label_stride%d'%(prefix, stride)] = _label
        label['%s_bbox_target_stride%d'%(prefix,stride)] = bbox_target
        label['%s_bbox_weight_stride%d'%(prefix,stride)] = bbox_weight
        if landmark:
          landmark_target = landmark_target.reshape((1, feat_height*feat_width, A * landmark_pred_len)).transpose(0, 2, 1)
          landmark_weight = landmark_weight.reshape((1, feat_height*feat_width, A * landmark_pred_len)).transpose((0, 2, 1))
          label['%s_landmark_target_stride%d'%(prefix,stride)] = landmark_target
          label['%s_landmark_weight_stride%d'%(prefix,stride)] = landmark_weight
        #print('in_rpn', stride,_label.shape, bbox_target.shape, bbox_weight.shape, file=sys.stderr)
        label_list.append(_label)
        #print('DD', _label.shape)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)
        if landmark:
          landmark_target_list.append(landmark_target)
          landmark_weight_list.append(landmark_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)
    #fg_inds = np.where(label_concat[0] == 1)[0]
    #print('fg_inds_in_rpn2', fg_inds, file=sys.stderr)

    label.update({'%s_label'%prefix: label_concat,
            '%s_bbox_target'%prefix: bbox_target_concat,
            '%s_bbox_weight'%prefix: bbox_weight_concat}
            )
    if landmark:
      landmark_target_concat = np.concatenate(landmark_target_list, axis=2)
      landmark_weight_concat = np.concatenate(landmark_weight_list, axis=2)
      label['%s_landmark_target'%prefix] = landmark_target_concat
      label['%s_landmark_weight'%prefix] = landmark_weight_concat
    return label


class AA:
  def __init__(self, feat_shape,sel_stride, flag):
    self.feat_shape = feat_shape
    feat_strides = config.RPN_FEAT_STRIDE
    if config.PROGRESSIVE and config.selective_branch:
        feat_strides = sel_stride
    self.sel_stride = sel_stride
    self.flag = flag
    anchors_list = []
    anchors_num_list = []
    inds_inside_list = []
    feat_infos = []
    A_list = []
    DEBUG = False
    self.debug = False
    anchor_my_list = []
    if not config.ct_reproduce or not config.centernet_branch:
        for i in range(len(feat_strides)):
            stride = feat_strides[i]
            sstride = str(stride)
            base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
            allowed_border = config.RPN_ANCHOR_CFG[sstride]['ALLOWED_BORDER']
            ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
            scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
            base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, dtype=np.float32), stride = stride, dense_anchor = config.DENSE_ANCHOR)
            # base_anchors = base_anchors[0:1,:]
            num_anchors = base_anchors.shape[0]
            feat_height, feat_width = feat_shape[i][-2:]
            feat_stride = feat_strides[i]
            feat_infos.append([feat_height, feat_width])

            A = num_anchors
            A_list.append(A)
            K = feat_height * feat_width
            all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
            all_anchors = all_anchors.reshape((K * A, 4))
            # if stride == 8:
            #     img = cv2.imread('./14_Traffic_Traffic_14_29_640_ori.jpg')
            #     for ii in [0,1,4,5,8,9]:
            #             box = all_anchors[ii].astype(np.int)
            #             color = (0, 255,0)
            #             # cv2.circle(img,((box[0]+box[2])/2,(box[1]+box[3])/2),2,color)
            #             cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 1)
            #             print(box)
            #             # font = cv2.FONT_HERSHEY_SIMPLEX
            #             # cv2.putText(img, "%d"%ii, (box[0], box[1]), font, 0.01 * (box[2] - box[0]),
            #             #             (255, 255, 255), 1)
            #     filename = './visualization/test11.jpg'
            #     print('writing', filename)
            #     cv2.imwrite(filename, img)

            #print('anchor0', stride, all_anchors[0])

            total_anchors = int(K * A)
            anchors_num_list.append(total_anchors)
            # only keep anchors inside the image
            inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                                   (all_anchors[:, 1] >= -allowed_border) &
                                   (all_anchors[:, 2] < config.SCALES[0][1] + allowed_border) &
                                   (all_anchors[:, 3] < config.SCALES[0][1] + allowed_border))[0]
            if DEBUG:
                print('total_anchors', total_anchors)
                print('inds_inside', len(inds_inside))

            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]
            #print('AA', anchors.shape, len(inds_inside))

            anchors_list.append(anchors)
            inds_inside_list.append(inds_inside)
        anchors = np.concatenate(anchors_list)

        for i in range(1, len(inds_inside_list)):
            inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])
        inds_inside = np.concatenate(inds_inside_list)
    #self.anchors_list = anchors_list
    #self.inds_inside_list = inds_inside_list
        self.anchors = anchors
        self.inds_inside = inds_inside
    self.anchors_num_list = anchors_num_list
    self.feat_infos = feat_infos
    self.A_list = A_list
    self._times = [0.0, 0.0, 0.0, 0.0]

  @staticmethod
  def _unmap(data, count, inds, fill=0):
      """" unmap a subset inds of data into original data of size count """
      if len(data.shape) == 1:
          ret = np.empty((count,), dtype=np.float32)
          ret.fill(fill)
          ret[inds] = data
      else:
          ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
          ret.fill(fill)
          ret[inds, :] = data
      return ret

  def get_wh_scale(self):
      feat_stridess = config.ct_stride[:]
      feat_stridess.reverse()
      wh_scale = {}
      for i,stride in enumerate(feat_stridess):
          sstride = str(stride)
          if len(feat_stridess) != 1:
              basesize = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
              scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
              if i == 0:
                  min_scale = 0
              else:
                  min_scale = max_scale
              if i == len(feat_stridess) - 1:
                  max_scale = 999999
              else:
                  max_scale = max(scales) * int(basesize)

              wh_scale[stride] = [min_scale, max_scale]
          else:
              if stride == 4:
                  wh_scale[stride] = [0, 25]
              elif stride == 8:
                  wh_scale[stride] = [0,64]
              elif stride == 16:
                  wh_scale[stride] = [64,256]
              #logger.info('anchor free range from %s to %s'%(wh_scale[stride][0],wh_scale[stride][1]))
      return wh_scale

  def assign_anchor_fpn(self, gt_label, im_info, landmark=False, prefix='face', select_stride=0):

    DEBUG = False
    #ta = datetime.datetime.now()
    im_info = im_info[0]
    if prefix == 'head':
        gt_boxes = gt_label['gt_boxes_head']
    else:
        gt_boxes = gt_label['gt_boxes']
    # clean up boxes
    nonneg = np.where(gt_boxes[:, 4] != -1)[0]
    gt_boxes = gt_boxes[nonneg]
    if config.USE_BLUR:
      gt_blur = gt_label['gt_blur']
      gt_blur = gt_blur[nonneg]
    if landmark:
      gt_landmarks = gt_label['gt_landmarks']
      gt_landmarks = gt_landmarks[nonneg]
      assert gt_boxes.shape[0]==gt_landmarks.shape[0]
    if config.centernet_branch:
        gt_centers = gt_label['gt_centers']
        gt_centers = gt_centers[nonneg]
    #scales = np.array(scales, dtype=np.float32)
    feat_strides = config.RPN_FEAT_STRIDE
    if config.PROGRESSIVE and config.selective_branch:
        if not self.flag:
            feat_strides = self.sel_stride
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
      gt_boxes[:,4] = gt_blur
      bbox_pred_len = 5
    if config.USE_OCCLUSION:
      landmark_pred_len = 15

    #anchors_list = self.anchors_list
    #inds_inside_list = self.inds_inside_list
    if config.ct_reproduce and config.centernet_branch:
        label = {}
    if not config.ct_reproduce or not config.centernet_branch:
        anchors = self.anchors
        inds_inside = self.inds_inside
        anchors_num_list = self.anchors_num_list
        feat_infos = self.feat_infos
        A_list = self.A_list

        total_anchors = sum(anchors_num_list)
        #print('total_anchors', anchors.shape[0], len(inds_inside), file=sys.stderr)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)
        #print('BB', anchors.shape, len(inds_inside))
        #print('gt_boxes', gt_boxes.shape, file=sys.stderr)
        #tb = datetime.datetime.now()
        #self._times[0] += (tb-ta).total_seconds()
        #ta = datetime.datetime.now()

        if gt_boxes.size > 0:
            # overlap between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))

            argmax_overlaps = overlaps.argmax(axis=1) #compute max overlaps for every bounding box
            #print('AAA', argmax_overlaps.shape)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0) #compute max overmaps for every ground truth
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

            '''if config.TRAIN.SCALE_COMPENSATION:
                labels[max_overlaps >= 0.35] = 1
    
    
                sorted_index = overlaps.argsort(axis=0)
                sorted_overlaps = overlaps[sorted_index,np.arange(overlaps.shape[1])]
                if sorted_overlaps.shape[0] > config.TRAIN.SCALE_COMPENSATION_N:
                    sorted_overlaps = sorted_overlaps[:config.TRAIN.SCALE_COMPENSATION_N, :]'''




        else:
            labels[:] = 0
        fg_inds = np.where(labels == 1)[0]
        #print('fg count', len(fg_inds))

        # subsample positive labels if we have too many
        if config.TRAIN.RPN_ENABLE_OHEM==0:
          '''fg_inds = np.where(labels == 1)[0]
          num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
          if len(fg_inds) > num_fg:
              disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
              if DEBUG:
                  disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
              labels[disable_inds] = -1
    
          # subsample negative labels if we have too many
          num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
          bg_inds = np.where(labels == 0)[0]
          if len(bg_inds) > num_bg:
              disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
              if DEBUG:
                  disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
              labels[disable_inds] = -1'''

          #fg_inds = np.where(labels == 1)[0]
          #num_fg = len(fg_inds)
          #num_bg = num_fg*int(1.0/config.TRAIN.RPN_FG_FRACTION-1)

          #bg_inds = np.where(labels == 0)[0]
          #if len(bg_inds) > num_bg:
          #    disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
          #    if DEBUG:
          #        disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
          #    labels[disable_inds] = -1
        else:
          fg_inds = np.where(labels == 1)[0]
          num_fg = len(fg_inds)
          bg_inds = np.where(labels == 0)[0]
          num_bg = len(bg_inds)

        #print('anchor stat', num_fg, num_bg)


        bbox_targets = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
        if gt_boxes.size > 0:
            #print('GT', gt_boxes.shape, gt_boxes[argmax_overlaps, :4].shape)
            bbox_targets[:,:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])
            #bbox_targets[:,4] = gt_blur
        #tb = datetime.datetime.now()
        #self._times[1] += (tb-ta).total_seconds()
        #ta = datetime.datetime.now()

        bbox_weights = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
        #bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)
        bbox_weights[labels == 1, 0:4] = 1.0
        if bbox_pred_len>4:
          bbox_weights[labels == 1, 4:bbox_pred_len] = 0.1

        if landmark:
          landmark_targets = np.zeros((len(inds_inside), landmark_pred_len), dtype=np.float32)
          #landmark_weights = np.zeros((len(inds_inside), 10), dtype=np.float32)
          landmark_weights = np.zeros((len(inds_inside), landmark_pred_len), dtype=np.float32)
          #landmark_weights[labels == 1, :] = np.array(config.TRAIN.RPN_LANDMARK_WEIGHTS)
          if landmark_pred_len==10:
            landmark_weights[labels == 1, :] = 1.0
          elif landmark_pred_len==15:
            v = [1.0, 1.0, 0.1] * 5
            assert len(v)==15
            landmark_weights[labels == 1, :] = np.array(v)
          else:
            assert False
          #TODO here
          if gt_landmarks.size > 0:
            #print('AAA',argmax_overlaps)
            a_landmarks = gt_landmarks[argmax_overlaps,:,:]
            landmark_targets[:] = landmark_transform(anchors, a_landmarks)
            if config.landmark_klloss:
                landmark_targets[:] = landmark_transform_xyxy(anchors, a_landmarks)
            invalid = np.where(a_landmarks[:,0,2]<0.0)[0]
            #assert len(invalid)==0
            #landmark_weights[invalid, :] = np.array(config.TRAIN.RPN_INVALID_LANDMARK_WEIGHTS)
            landmark_weights[invalid, :] = 0.0
        #tb = datetime.datetime.now()
        #self._times[2] += (tb-ta).total_seconds()
        #ta = datetime.datetime.now()

        #if DEBUG:
        #    _sums = bbox_targets[labels == 1, :].sum(axis=0)
        #    _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        #    _counts = np.sum(labels == 1)
        #    means = _sums / (_counts + 1e-14)
        #    stds = np.sqrt(_squared_sums / _counts - means ** 2)
        #    print 'means', means
        #    print 'stdevs', stds
        # map up to original set of anchors
        #print(labels.shape, total_anchors, inds_inside.shape, inds_inside[0], inds_inside[-1])
        labels = AA._unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = AA._unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_weights = AA._unmap(bbox_weights, total_anchors, inds_inside, fill=0)
        if landmark:
          landmark_targets = AA._unmap(landmark_targets, total_anchors, inds_inside, fill=0)
          landmark_weights = AA._unmap(landmark_weights, total_anchors, inds_inside, fill=0)
        #print('CC', anchors.shape, len(inds_inside))

        #if DEBUG:
        #    if gt_boxes.size > 0:
        #        print 'rpn: max max_overlaps', np.max(max_overlaps)
        #    print 'rpn: num_positives', np.sum(labels == 1)
        #    print 'rpn: num_negatives', np.sum(labels == 0)
        #    _fg_sum = np.sum(labels == 1)
        #    _bg_sum = np.sum(labels == 0)
        #    _count = 1
        #    print 'rpn: num_positive avg', _fg_sum / _count
        #    print 'rpn: num_negative avg', _bg_sum / _count

        # resahpe
        label_list = list()
        bbox_target_list = list()
        bbox_weight_list = list()
        if landmark:
          landmark_target_list = list()
          landmark_weight_list = list()
        anchors_num_range = [0] + anchors_num_list
        label = {}
        for i in range(len(feat_strides)):
            stride = feat_strides[i]
            feat_height, feat_width = feat_infos[i]
            A = A_list[i]
            _label = labels[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
            if select_stride>0 and stride!=select_stride:
              #print('set', stride, select_stride)
              _label[:] = -1
            #print('_label', _label.shape, select_stride)
            #_fg_inds = np.where(_label == 1)[0]
            #n_fg = len(_fg_inds)
            #STAT[0]+=1
            #STAT[stride]+=n_fg
            #if STAT[0]%100==0:
            #  print('rpn_stat', STAT, file=sys.stderr)
            bbox_target = bbox_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
            bbox_weight = bbox_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
            if landmark:
              landmark_target = landmark_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
              landmark_weight = landmark_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]

            _label = _label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
            _label = _label.reshape((1, A * feat_height * feat_width))
            bbox_target = bbox_target.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose(0, 2, 1)
            bbox_weight = bbox_weight.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose((0, 2, 1))
            label['%s_label_stride%d'%(prefix, stride)] = _label
            label['%s_bbox_target_stride%d'%(prefix,stride)] = bbox_target
            label['%s_bbox_weight_stride%d'%(prefix,stride)] = bbox_weight
            if landmark:
              landmark_target = landmark_target.reshape((1, feat_height*feat_width, A * landmark_pred_len)).transpose(0, 2, 1)
              landmark_weight = landmark_weight.reshape((1, feat_height*feat_width, A * landmark_pred_len)).transpose((0, 2, 1))
              label['%s_landmark_target_stride%d'%(prefix,stride)] = landmark_target
              label['%s_landmark_weight_stride%d'%(prefix,stride)] = landmark_weight
            #print('in_rpn', stride,_label.shape, bbox_target.shape, bbox_weight.shape, file=sys.stderr)
            label_list.append(_label)
            #print('DD', _label.shape)
            bbox_target_list.append(bbox_target)
            bbox_weight_list.append(bbox_weight)
            if landmark:
              landmark_target_list.append(landmark_target)
              landmark_weight_list.append(landmark_weight)

        ############# Progressive v2 ######################
        valid_stride_idx = 2
        if config.Progressive_v2:
            for i in range(len(feat_strides)):
                if i <= valid_stride_idx:
                    continue
                label_list[i][:] = -1
                bbox_target_list[i][:] = 0
                bbox_weight_list[i][:] = 0
                landmark_target_list[i][:] = 0
                landmark_weight_list[i][:] = 0
        ###################################################

        label_concat = np.concatenate(label_list, axis=1)
        bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
        bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)
        #fg_inds = np.where(label_concat[0] == 1)[0]
        #print('fg_inds_in_rpn2', fg_inds, file=sys.stderr)

        label.update({'%s_label'%prefix: label_concat,
                '%s_bbox_target'%prefix: bbox_target_concat,
                '%s_bbox_weight'%prefix: bbox_weight_concat}
                )
        if landmark:
          landmark_target_concat = np.concatenate(landmark_target_list, axis=2)
          landmark_weight_concat = np.concatenate(landmark_weight_list, axis=2)
          label['%s_landmark_target'%prefix] = landmark_target_concat
          label['%s_landmark_weight'%prefix] = landmark_weight_concat

    if config.centernet_branch: #for a image
        #if gt_centers.shape[0] > 0:
            for stride in config.ct_stride:
                input_h = config.SCALES[0][0]
                input_w = config.SCALES[0][0]
                output_h = input_h // stride
                output_w = input_w // stride
                hm = np.zeros((1, output_h, output_w), dtype=np.float32)
                if config.ct_wh_inds:
                    max_objs = config.max_objs
                    wh = np.zeros((max_objs, 2), dtype=np.float32)
                    ind = np.zeros((max_objs), dtype=np.int64)
                    reg = np.zeros((max_objs,2), dtype=np.float32)
                    reg_mask = np.zeros((max_objs), dtype=np.uint8)
                else:
                    wh = np.zeros((2,output_h,output_w),dtype=np.float32)
                    reg = np.zeros((2,output_h,output_w), dtype=np.float32)
                    reg_mask = np.zeros((1,output_h,output_w), dtype=np.uint8)
                if config.ct_landmarks:
                    ct_landmarks = np.zeros((output_h,output_w,5,2),dtype=np.float32)
                # TODO ind reg_mask
                draw_gaussian = draw_umich_gaussian
                if config.ct_wh_inds:
                    num_faces = min(gt_boxes.shape[0], max_objs)
                else:
                    num_faces = min(gt_boxes.shape[0], output_h*output_w)
                for k in range(num_faces):
                    bbox = gt_boxes[k,0:4] / stride
                    if config.ct_landmarks:
                        ldks = gt_landmarks[k,:,0:2] / stride
                    # bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
                    # bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if h > 0 and w > 0:
                        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                        ct_int = ct.astype(np.int32)  # ct = (x,y)

                        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                        radius = max(0, int(radius))
                        # if radius == 2:
                        #     print(radius)

                        if ct_int[0] >= output_w or ct_int[1] >= output_h: #some center points are outside the feature map
                            continue
                        if config.ct_scale: # do scale to anchor-free branch
                            wh_scale = self.get_wh_scale()
                            #wh_scale = {8:[0,64],16:[64,256],32:[256,999999]}
                            if h * stride > wh_scale[stride][1] or h * stride< wh_scale[stride][0] or w * stride> wh_scale[stride][1] or w * stride< wh_scale[stride][0]:
                                if config.ct_ignore:
                                    hm[0, ct_int[1], ct_int[0]] = 2
                                if config.ct_hmclasses:
                                    hm[0, ct_int[1], ct_int[0]] = -1
                                continue
                        #draw_gaussian(hm[0], ct_int, 2)
                        draw_gaussian(hm[0], ct_int, radius)
                        if config.ct_wh_log:
                            h, w = np.log(bbox[3] - bbox[1]), np.log(bbox[2] - bbox[0])
                        if config.ct_wh_inds:
                            wh[k] = 1. * w, 1. * h
                            ind[k] = ct_int[1] * output_w + ct_int[0]
                            reg[k] = ct - ct_int
                            reg_mask[k] = 1
                        else:
                            wh[:,ct_int[1],ct_int[0]] = 1. * h, 1. * w
                            tmp = ct - ct_int
                            reg[:,ct_int[1],ct_int[0]] = tmp[1], tmp[0]
                            reg_mask[:,ct_int[1],ct_int[0]] = 1
                        if config.ct_landmarks:
                            bbox_ldk = np.reshape(bbox,(1,4))
                            ldks = ldks[np.newaxis,:]
                            ct_landmarks[ct_int[1],ct_int[0],:] = landmark_transform_af(ct_int, ldks,bbox_ldk, output_h,output_w)
                # if num_faces > 100:
                #     print(num_faces)
                # hm0 = hm[0]
                # img = cv2.imread('./visualization/test11.jpg')
                # for ii in [1]:
                #     for jj in [1,3,5]:
                #         color = (255, 0,0)
                #         cv2.circle(img, (jj*stride,ii*stride), 1, color)
                # filename = './visualization/test11.jpg'
                #
                # cv2.imwrite(filename,img)
                # print('writing', filename)

                if prefix == 'face':
                    label['ct_hm_stride_%s'%stride] = hm[np.newaxis,:]

                    label['ct_wh_stride_%s' % stride] = wh[np.newaxis,:]
                    label['ct_wh_mask_stride_%s'%stride] = reg_mask[np.newaxis,:]
                    if config.ct_offset:
                        label['ct_offset_stride_%s'%stride] = reg[np.newaxis,:]
                    if config.ct_wh_inds:
                        label['ct_ind_stride_%s' % stride] = ind[np.newaxis,:]
                    if config.ct_landmarks:
                        ct_landmarks = ct_landmarks.reshape((ct_landmarks.shape[0],ct_landmarks.shape[1], -1))
                        ct_landmarks = ct_landmarks.transpose(2,0,1)
                        label['ct_landmark_stride_%s' % stride] = ct_landmarks[np.newaxis, :]
                elif prefix == 'head' and config.ct_head:
                    label['ct_head_hm_stride_%s' % stride] = hm[np.newaxis, :]

                    label['ct_head_wh_stride_%s' % stride] = wh[np.newaxis, :]
                    label['ct_head_wh_mask_stride_%s' % stride] = reg_mask[np.newaxis, :]
                    if config.ct_offset:
                        label['ct_head_offset_stride_%s' % stride] = reg[np.newaxis, :]
                    if config.ct_wh_inds:
                        label['ct_head_ind_stride_%s' % stride] = ind[np.newaxis, :]
                    if config.ct_landmarks:
                        ct_landmarks = ct_landmarks.reshape((ct_landmarks.shape[0], ct_landmarks.shape[1], -1))
                        ct_landmarks = ct_landmarks.transpose(2, 0, 1)
                        label['ct_head_landmark_stride_%s' % stride] = ct_landmarks[np.newaxis, :]
                # if config.ct_mining:
                #     if gt_boxes.size > 0:
                #         label['overlaps'] = overlaps
                #     else:
                #         label['overlaps'] = np.zeros((anchors.shape[0],0))

            # if not label.has_key('ct_hm_stride_8') or not label.has_key('ct_hm_stride_16') or not label.has_key('ct_hm_stride_32'):
            #     print('sss')
    #tb = datetime.datetime.now()
    #self._times[3] += (tb-ta).total_seconds()
    #ta = datetime.datetime.now()
    #print(self._times)
    return label
