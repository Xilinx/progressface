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

import numpy as np
from ..cython.bbox import bbox_overlaps_cython
from ..config import config
#from rcnn.config import config


def bbox_overlaps(boxes, query_boxes):
    return bbox_overlaps_cython(boxes, query_boxes)


def bbox_overlaps_py(boxes, query_boxes):
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


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def nonlinear_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    if gt_rois.shape[1]<=4:
      targets = np.vstack(
          (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
      return targets
    else:
      targets = [targets_dx, targets_dy, targets_dw, targets_dh]
      #if config.USE_BLUR:
      #  for i in range(4, gt_rois.shape[1]):
      #    t = gt_rois[:,i]
      #    targets.append(t)
      targets = np.vstack(targets).transpose()
      return targets

def landmark_transform(ex_rois, gt_rois):

    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    
    targets = []
    for i in range(gt_rois.shape[1]):
      for j in range(gt_rois.shape[2]):
        #if not config.USE_OCCLUSION and j==2:
        #  continue
        if j==2:
          continue
        if j==0: #w
          target = (gt_rois[:,i,j] - ex_ctr_x) / (ex_widths + 1e-14)
        elif j==1: #h
          target = (gt_rois[:,i,j] - ex_ctr_y) / (ex_heights + 1e-14)
        else: #visibile
          target = gt_rois[:,i,j]
        targets.append(target)


    targets = np.vstack(targets).transpose()
    return targets


def landmark_transform_af(ct_int,gt_roi,gt_box,output_h,output_w):

    gt_widths = gt_box[0, 2] - gt_box[0, 0]
    gt_heights = gt_box[0, 3] - gt_box[0, 1]

    targets = np.zeros((5,2),dtype=np.float32)
    for i in range(gt_roi.shape[1]):
        for j in range(gt_roi.shape[2]):
            # if not config.USE_OCCLUSION and j==2:
            #  continue
            if j == 2:
                continue
            if j == 0:  # w
                #target = (gt_roi[:, i, j] - ct_int[0]) / (gt_widths + 1e-14)
                targets[i,j] = (gt_roi[0, i, j] - ct_int[0]) / (gt_widths + 1e-14)
            elif j == 1:  # h
                #target = (gt_roi[:, i, j] - ct_int[1]) / (gt_heights + 1e-14)
                targets[i,j] = (gt_roi[0, i, j] - ct_int[1]) / (gt_heights + 1e-14)

    #targets = np.vstack(targets).transpose()
    return targets
def nonlinear_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes

def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    preds = []
    for i in range(landmark_deltas.shape[1]):
      if i%2==0:
        pred = (landmark_deltas[:,i]*widths + ctr_x)
      else:
        pred = (landmark_deltas[:,i]*heights + ctr_y)
      preds.append(pred)
    preds = np.vstack(preds).transpose()
    return preds

def iou_transform(ex_rois, gt_rois):
    """ return bbox targets, IoU loss uses gt_rois as gt """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'
    return gt_rois


def iou_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    dx1 = box_deltas[:, 0::4]
    dy1 = box_deltas[:, 1::4]
    dx2 = box_deltas[:, 2::4]
    dy2 = box_deltas[:, 3::4]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = dx1 + x1[:, np.newaxis]
    # y1
    pred_boxes[:, 1::4] = dy1 + y1[:, np.newaxis]
    # x2
    pred_boxes[:, 2::4] = dx2 + x2[:, np.newaxis]
    # y2
    pred_boxes[:, 3::4] = dy2 + y2[:, np.newaxis]

    return pred_boxes

def bbox_transform_xyxy(ex_rois, gt_rois, weights=(1.0, 1.0, 1.0, 1.0)):
    #  0 1 2 3
    #  < v > ^

    gt_l = gt_rois[:, 0]
    gt_r = gt_rois[:, 2]
    gt_d = gt_rois[:, 1]
    gt_u = gt_rois[:, 3]

    ex_l = ex_rois[:, 0]
    ex_r = ex_rois[:, 2]
    ex_d = ex_rois[:, 1]
    ex_u = ex_rois[:, 3]

    ex_widths = ex_r - ex_l + 1.0
    ex_heights = ex_u - ex_d + 1.0

    wx, wy, ww, wh = weights
    targets_dl = wx * (gt_l - ex_l) / ex_widths
    targets_dr = wy * (gt_r - ex_r) / ex_widths
    targets_dd = wx * (gt_d - ex_d) / ex_heights
    targets_du = wy * (gt_u - ex_u) / ex_heights

    targets = np.vstack(
        (targets_dl, targets_dr, targets_dd, targets_du)).transpose()
    return targets


def landmark_transform_xyxy(ex_rois, gt_rois):
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_l = ex_rois[:, 0]
    ex_r = ex_rois[:, 2]
    ex_d = ex_rois[:, 1]
    ex_u = ex_rois[:, 3]

    ex_widths = ex_r - ex_l + 1.0
    ex_heights = ex_u - ex_d + 1.0

    targets = []
    for i in range(gt_rois.shape[1]):
        for j in range(gt_rois.shape[2]):
            # if not config.USE_OCCLUSION and j==2:
            #  continue
            if j == 2:
                continue
            if j == 0:  # w
                if i % 3 == 0 or i % 3 == 2:
                    target = (gt_rois[:, i, j] - ex_l) / (ex_widths + 1e-14)
                elif i % 3 == 1:
                    target = (gt_rois[:, i, j] - ex_r) / (ex_widths + 1e-14)

            elif j == 1:  # h
                if i % 3 == 0 or i % 3 == 2:
                    target = (gt_rois[:, i, j] - ex_d) / (ex_heights + 1e-14)
                elif i % 3 == 1:
                    target = (gt_rois[:, i, j] - ex_u) / (ex_heights + 1e-14)
            else:  # visibile
                target = gt_rois[:, i, j]
            targets.append(target)

    targets = np.vstack(targets).transpose()
    return targets

def bbox_pred_xyxy(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    #  0 1 2 3
    #  < v > ^

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0

    l = boxes[:, 0]
    r = boxes[:, 2]
    d = boxes[:, 1]
    u = boxes[:, 3]

    wx, wy, ww, wh = weights
    dl = deltas[:, 0::4] / wx
    dr = deltas[:, 1::4] / wy
    dd = deltas[:, 2::4] / wx
    du = deltas[:, 3::4] / wy
    # Prevent sending too large values into np.exp()
    BBOX_XFORM_CLIP = np.log(1000. / 16.)
    BBOX_XFORM_CLIPe = 1000. / 16.
    dl = np.maximum(np.minimum(dl, BBOX_XFORM_CLIPe), -BBOX_XFORM_CLIPe)
    dr = np.maximum(np.minimum(dr, BBOX_XFORM_CLIPe), -BBOX_XFORM_CLIPe)
    dd = np.maximum(np.minimum(dd, BBOX_XFORM_CLIPe), -BBOX_XFORM_CLIPe)
    du = np.maximum(np.minimum(du, BBOX_XFORM_CLIPe), -BBOX_XFORM_CLIPe)

    pred_l = dl * widths[:, np.newaxis] + l[:, np.newaxis]
    pred_r = dr * widths[:, np.newaxis] + r[:, np.newaxis]
    pred_d = dd * heights[:, np.newaxis] + d[:, np.newaxis]
    pred_u = du * heights[:, np.newaxis] + u[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_l
    # y1
    pred_boxes[:, 1::4] = pred_d
    # x2
    pred_boxes[:, 2::4] = pred_r
    # y2
    pred_boxes[:, 3::4] = pred_u

    return pred_boxes

# define bbox_transform and bbox_pred
bbox_transform = nonlinear_transform
if config.USE_KLLOSS:
    bbox_transform = bbox_transform_xyxy
bbox_pred = nonlinear_pred
if config.USE_KLLOSS:
    bbox_pred = bbox_pred_xyxy

