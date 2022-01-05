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
import os
import datetime
import time
import numpy as np
sys.path.insert(0,'mxnet_v1.3/python/')
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from rcnn.config import config
from rcnn.logger import logger
#from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes, landmark_pred
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps
from ce_det_func.decoder import _nms, _topk

class ProgressFace:
  def __init__(self, prefix, epoch, ctx_id=0, network='net3', nms=0.4, nocrop=False, decay4 = 0.5, vote=False):
    self.ctx_id = ctx_id
    self.network = network
    self.decay4 = decay4
    self.nms_threshold = nms
    self.vote = vote
    self.nocrop = nocrop
    self.debug = False
    self.fpn_keys = []
    self.anchor_cfg = None
    pixel_means=[0.0, 0.0, 0.0]
    pixel_stds=[1.0, 1.0, 1.0]
    #pixel_means = [123.68, 116.779, 103.939]
    #pixel_stds = [58.393, 57.13, 57.375]
    pixel_scale = 1.0
    self.preprocess = False
    _ratio = (1.,)
    fmc = 3
    if network=='ssh' or network=='vgg':
      #pixel_means=[103.939, 116.779, 123.68]
      self.preprocess = True
    elif network=='net3':
      _ratio = (1.,)
    elif network=='net3a':
      _ratio = (1.,1.5)
    elif network=='net6': #like pyramidbox or s3fd
      fmc = 6
    elif network=='net5': #retinaface
      fmc = 5
    elif network=='net5a':
      fmc = 5
      _ratio = (1.,1.5)
    elif network=='net4':
      fmc = 4
    elif network=='net2':
      fmc = 2
    elif network=='net4a':
      fmc = 4
      _ratio = (1.,1.5)
    else:
      assert False, 'network setting error %s'%network

    if fmc==3:
      self._feat_stride_fpn = [32, 16, 8]
      self.anchor_cfg = {
          '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
      }

      '''self.anchor_cfg = {
          '32': {'SCALES': (12.70, 10.08, 8), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (6.35, 5.04, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (3.17, 2.52, 2), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
      }'''
    elif fmc==4:
      self._feat_stride_fpn = [32, 16, 8, 4]
      self.anchor_cfg = {
          '32': {'SCALES': (8,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (4,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (2,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '4': {'SCALES': (1,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
      }
    elif fmc==6:
      self._feat_stride_fpn = [128, 64, 32, 16, 8, 4]
      self.anchor_cfg = {
          '128': {'SCALES': (32,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '64': {'SCALES': (16,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '32': {'SCALES': (8,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (4,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (2,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '4': {'SCALES': (1,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
      }
    elif fmc==5:
      self._feat_stride_fpn = [64, 32, 16, 8, 4]
      self.anchor_cfg = {}
      _ass = 2.0**(1.0/3)
      _basescale = 1.0
      for _stride in [4, 8, 16, 32, 64]:
        key = str(_stride)
        value = {'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999}
        scales = []
        for _ in range(3):
          scales.append(_basescale)
          _basescale *= _ass
        value['SCALES'] = tuple(scales)
        self.anchor_cfg[key] = value
        # if _stride == 4:
        #     value1 = value
        #     value1['SCALES'] = (0.6, 1, 1.25)
        #     self.anchor_cfg[key] = value1
    elif fmc == 2:
        self._feat_stride_fpn = [32, 16]
        self.anchor_cfg = {
            '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},

        }

    print(self._feat_stride_fpn, self.anchor_cfg)

    for s in self._feat_stride_fpn:
        self.fpn_keys.append('stride%s'%s)


    dense_anchor = False
    #self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_base_size, scales=self._scales, ratios=self._ratios)))
    self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=dense_anchor, cfg=self.anchor_cfg)))
    for k in self._anchors_fpn:
      v = self._anchors_fpn[k].astype(np.float32)
      self._anchors_fpn[k] = v

    self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
    #self._bbox_pred = nonlinear_pred
    #self._landmark_pred = landmark_pred
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    if self.ctx_id>=0:
      self.ctx = mx.gpu(self.ctx_id)
      self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
    else:
      self.ctx = mx.cpu()
      self.nms = cpu_nms_wrapper(self.nms_threshold)
    self.pixel_means = np.array(pixel_means, dtype=np.float32)
    self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
    self.pixel_scale = float(pixel_scale)
    print('means', self.pixel_means)
    self.use_landmarks = True
    if len(sym)//len(self._feat_stride_fpn)==3:
      self.use_landmarks = True
    if config.USE_KLLOSS:
        if len(sym)//len(self._feat_stride_fpn)==4:
            self.use_landmarks = True
    print('use_landmarks', self.use_landmarks)

    if self.debug:
      c = len(sym)//len(self._feat_stride_fpn)
      sym = sym[(c*0):]
      self._feat_stride_fpn = [32,16,8]
    print('sym size:', len(sym))

    image_size = (640, 640)
    # image_size = (800, 800)
    self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
    self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
    self.model.set_params(arg_params, aux_params)

  def get_input(self, img):
    im = img.astype(np.float32)
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = (im[:, :, 2 - i]/self.pixel_scale - self.pixel_means[2 - i])/self.pixel_stds[2-i]
    #if self.debug:
    #  timeb = datetime.datetime.now()
    #  diff = timeb - timea
    #  print('X2 uses', diff.total_seconds(), 'seconds')
    data = nd.array(im_tensor)
    return data

  def detect(self, img, threshold=0.5, scales=[1.0], do_flip=False):
    #print('in_detect', threshold, scales, do_flip, do_nms)
    proposals_list = []
    scores_list = []
    landmarks_list = []
    if config.centernet_branch:
        ct_bbox_list = []
        ct_score_list = []
        ct_bbox_flip_list = []
        ct_score_flip_list = []
        if config.ct_landmarks:
            ct_landmarks_list = []
    if config.USE_KLLOSS:
        alpha_list = []
    timea = datetime.datetime.now()
    flips = [0]
    if do_flip:
      flips = [0, 1]

    for im_scale in scales:
      for flip in flips:
        if im_scale!=1.0:
          im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        else:
          im = img.copy()
        if flip:
          im = im[:,::-1,:]
        if self.nocrop:
          if im.shape[0]%32==0:
            h = im.shape[0]
          else:
            h = (im.shape[0]//32+1)*32
          if im.shape[1]%32==0:
            w = im.shape[1]
          else:
            w = (im.shape[1]//32+1)*32
          _im = np.zeros( (h, w, 3), dtype=np.float32 )
          _im[0:im.shape[0], 0:im.shape[1], :] = im
          im = _im
        else:
          im = im.astype(np.float32)
        if self.debug:
          timeb = datetime.datetime.now()
          diff = timeb - timea
          print('X1 uses', diff.total_seconds(), 'seconds')
        #self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
        #im_info = [im.shape[0], im.shape[1], im_scale]

        im_info = [im.shape[0], im.shape[1]]
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i]/self.pixel_scale - self.pixel_means[2 - i])/self.pixel_stds[2-i]
        if self.debug:
          timeb = datetime.datetime.now()
          diff = timeb - timea
          print('X2 uses', diff.total_seconds(), 'seconds')
        data = nd.array(im_tensor)
        db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
        if self.debug:
          timeb = datetime.datetime.now()
          diff = timeb - timea
          print('X3 uses', diff.total_seconds(), 'seconds')
        tic = time.time()
        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        toc = time.time()
        #post_nms_topN = self._rpn_post_nms_top_n
        #min_size_dict = self._rpn_min_size_fpn

        if not config.ct_reproduce or not config.centernet_branch:
            for _idx,s in enumerate(self._feat_stride_fpn):
                # if im_scale <= 1:
                    #if len(scales)>1 and s==32 and im_scale==scales[-1]:
                    #  continue
                    _key = 'stride%s'%s
                    stride = int(s)
                    #if self.vote and stride==4 and len(scales)>2 and (im_scale==scales[0]):
                    #  continue
                    if self.use_landmarks:
                      idx = _idx*3
                    else:
                      idx = _idx*2
                    if config.USE_KLLOSS:
                      idx = _idx*4
                      if config.landmark_klloss:
                          idx = _idx*5
                    if config.STR:
                        if stride == 64:
                            idx = 0
                        elif stride == 32:
                            idx = 5
                        elif stride == 16:
                            idx = 10
                        elif stride == 8:
                            idx = 13
                        elif stride == 4:
                            idx = 16
                    # if config.centernet_branch:
                    #     if stride == 32:
                    #         idx = 0
                    #     elif stride == 16:
                    #         idx = 5
                    #     else:
                    #         idx = 10

                    #print('getting', im_scale, stride, idx, len(net_out), data.shape, file=sys.stderr)
                    scores = net_out[idx].asnumpy()
                    if config.STR_cls and stride in config.STR_FPN_STRIDE:# if do refinement to cls
                        scores = net_out[idx+3]
                    if self.debug:
                      timeb = datetime.datetime.now()
                      diff = timeb - timea
                      print('A uses', diff.total_seconds(), 'seconds')
                    #print(scores.shape)
                    #print('scores',stride, scores.shape, file=sys.stderr)
                    scores = scores[:, self._num_anchors['stride%s'%s]:, :, :] #only need positive score

                    idx+=1
                    if config.STR and stride in config.STR_FPN_STRIDE:
                        idx += 3
                    bbox_deltas = net_out[idx].asnumpy()

                    #if DEBUG:
                    #    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                    #    print 'scale: {}'.format(im_info[2])

                    #_height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
                    height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

                    A = self._num_anchors['stride%s'%s]
                    K = height * width
                    anchors_fpn = self._anchors_fpn['stride%s'%s]
                    min_stride = min(self._feat_stride_fpn)
                    if config.USE_OHEGS:
                        anchors = anchors_plane(height, width, min_stride, anchors_fpn)
                    else:
                        anchors = anchors_plane(height, width, stride, anchors_fpn)
                    #print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
                    anchors = anchors.reshape((K * A, 4))
                    #print('num_anchors', self._num_anchors['stride%s'%s], file=sys.stderr)
                    #print('HW', (height, width), file=sys.stderr)
                    #print('anchors_fpn', anchors_fpn.shape, file=sys.stderr)
                    #print('anchors', anchors.shape, file=sys.stderr)
                    #print('bbox_deltas', bbox_deltas.shape, file=sys.stderr)
                    #print('scores', scores.shape, file=sys.stderr)


                    scores = self._clip_pad(scores, (height, width))
                    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

                    #print('pre', bbox_deltas.shape, height, width)
                    bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
                    #print('after', bbox_deltas.shape, height, width)
                    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                    bbox_pred_len = bbox_deltas.shape[3]//A
                    #print(bbox_deltas.shape)
                    bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

                    #print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
                    if config.USE_KLLOSS:
                        proposals = self.bbox_pred_xyxy(anchors, bbox_deltas)
                    else:
                        proposals = self.bbox_pred(anchors, bbox_deltas)
                    proposals = clip_boxes(proposals, im_info[:2])

                    #if self.vote:
                    #  if im_scale>1.0:
                    #    keep = self._filter_boxes2(proposals, 160*im_scale, -1)
                    #  else:
                    #    keep = self._filter_boxes2(proposals, -1, 100*im_scale)
                    #  if stride==4:
                    #    keep = self._filter_boxes2(proposals, 12*im_scale, -1)
                    #    proposals = proposals[keep, :]
                    #    scores = scores[keep]

                    #keep = self._filter_boxes(proposals, min_size_dict['stride%s'%s] * im_info[2])
                    #proposals = proposals[keep, :]
                    #scores = scores[keep]
                    #print('333', proposals.shape)

                    scores_ravel = scores.ravel()
                    #print('__shapes', proposals.shape, scores_ravel.shape)
                    #print('max score', np.max(scores_ravel))
                    order = np.where(scores_ravel>=threshold)[0]
                      #_scores = scores_ravel[order]
                      #_order = _scores.argsort()[::-1]
                      #order = order[_order]
                    proposals = proposals[order, :]
                    scores = scores[order]
                    if stride==4 and self.decay4<1.0:
                      scores *= self.decay4
                    if flip:
                      oldx1 = proposals[:, 0].copy()
                      oldx2 = proposals[:, 2].copy()
                      proposals[:, 0] = im.shape[1] - oldx2 - 1
                      proposals[:, 2] = im.shape[1] - oldx1 - 1

                    # if config.ct_discard and config.centernet_branch:
                    #     discard_idx = np.where(((proposals[2]-proposals[0]) > 64) and ((proposals[3]-proposals[1]) > 64))[0]
                    #     proposals = proposals[discard_idx,:]
                    #     scores = scores[discard_idx]
                    proposals[:,0:4] /= im_scale

                    proposals_list.append(proposals)
                    scores_list.append(scores)

                    if not self.vote and self.use_landmarks:
                      idx+=1
                      landmark_deltas = net_out[idx].asnumpy()
                      landmark_deltas = self._clip_pad(landmark_deltas, (height, width))
                      landmark_pred_len = landmark_deltas.shape[1]//A
                      landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
                      #print(landmark_deltas.shape, landmark_deltas)
                      landmarks = self.landmark_pred(anchors, landmark_deltas)
                      landmarks = landmarks[order, :]

                      if flip:
                        landmarks[:,:,0] = im.shape[1] - landmarks[:,:,0] - 1
                        #for a in range(5):
                        #  oldx1 = landmarks[:, a].copy()
                        #  landmarks[:,a] = im.shape[1] - oldx1 - 1
                        order1 = [1,0,2,4,3]
                        flandmarks = landmarks.copy()
                        for tmp_idx, a in enumerate(order1):
                          flandmarks[:,tmp_idx,:] = landmarks[:,a,:]
                          #flandmarks[:, idx*2] = landmarks[:,a*2]
                          #flandmarks[:, idx*2+1] = landmarks[:,a*2+1]
                        landmarks = flandmarks
                      landmarks[:,:,0:2] /= im_scale
                      #landmarks /= im_scale
                      #landmarks = landmarks.reshape( (-1, landmark_pred_len) )
                      landmarks_list.append(landmarks)
                      #proposals = np.hstack((proposals, landmarks))

                    if config.USE_KLLOSS and config.var_voting:
                        if not self.vote and self.use_landmarks:
                            idx += 1
                        else:
                            idx+=2
                        alpha_pred = net_out[idx].asnumpy()
                        alpha_pred = alpha_pred.transpose((0,2,1))
                        alpha_pred = alpha_pred.reshape((-1,bbox_pred_len))
                        alpha_pred = alpha_pred[order,:]

                        if flip:
                            alpha_pred[:,0] = -alpha_pred[:,0]
                            alpha_pred[:,2] = -alpha_pred[:,2]
                        alpha_list.append(alpha_pred)

                    # if config.centernet_branch and not self.vote:
                    #     # if stride != 32:
                    #     #     continue
                    #     if stride == 32:
                    #         idx = 3
                    #     elif stride == 16:
                    #         idx = 8
                    #     else:
                    #         idx = 13
                    #     hm_pred = nd.sigmoid(net_out[idx])
                    #     wh_pred = net_out[idx + 1]
                    #     hm_pred = _nms(hm_pred)
                    #     ct_scores, ct_inds, clses, ys, xs = _topk(hm_pred, K=1000)
                    #     # xs = nd.reshape(xs, (batch, 1000, 1)) + 0.5
                    #     # ys = nd.reshape(ys, (batch, 1000, 1)) + 0.5
                    #     wh_pred_1 = nd.zeros((1000,2),ctx=wh_pred.context)
                    #     xs += 1
                    #     ys += 1
                    #     for i in range(ys.shape[0]):
                    #         wh_pred_1[i,:] = wh_pred[:,:,ys[i,0].astype(int),xs[i,0].astype(int)].reshape(2)
                    #     ct_bboxes = nd.concat(xs - wh_pred_1[:,1:2] / 2,
                    #                        ys - wh_pred_1[:,0:1] / 2,
                    #                        xs + wh_pred_1[:,1:2] / 2,
                    #                        ys + wh_pred_1[:,0:1] / 2,
                    #                        dim=1)
                    #
                    #     ct_threshold = 0.3
                    #     inx = np.where(ct_scores >= ct_threshold)[1]
                    #     if len(inx) > 0:
                    #         ct_bboxes = ct_bboxes[inx, :] * stride
                    #         ct_bbox_list.append(ct_bboxes.asnumpy())
                # else:
                #     idx = 5
        if config.centernet_branch: #and im_scale >= 1:
            if config.ct_reproduce:
                anchor_idx = 15
            else:
                anchor_idx = idx + 1
                if self.vote:  # because landmarks don't add 1
                    anchor_idx += 1
            for _idx, s in enumerate(config.ct_stride):
                if config.ct_offset:
                    idx = _idx * 3
                    if config.ct_landmarks:
                        idx = _idx * 4
                else:
                    idx = _idx * 2
                stride = int(s)
                index = anchor_idx + idx
                hm_pred = nd.sigmoid(net_out[index]).asnumpy()
                index += 1
                wh_pred = net_out[index].asnumpy()
                index += 1
                if config.ct_offset:
                    offset_pred = net_out[index].asnumpy()
                    index += 1
                if config.ct_landmarks:
                    landmark_pred = net_out[index].asnumpy()
                    landmark_pred = landmark_pred.reshape((landmark_pred.shape[0],5,2,landmark_pred.shape[2],landmark_pred.shape[3]))
                hm_pred = _nms(nd.array(hm_pred)).asnumpy()
                num_temp = hm_pred.shape[2] * hm_pred.shape[3]
                ct_k = 1000
                if num_temp < ct_k:
                    ct_k = num_temp
                ct_scores, ct_inds, clses, ys, xs = _topk(nd.array(hm_pred), K=ct_k)
                wh_pred_1 = np.zeros((ct_k, 2)).astype(np.float32)
                offset = np.zeros((ct_k, 2)).astype(np.float32)
                if config.ct_landmarks:
                    landmark_pred_1 = np.zeros((ct_k,5,2))
                for i in range(ys.shape[0]):
                    wh_pred_1[i, :] = wh_pred[:, :, ys[i, 0].astype(int), xs[i, 0].astype(int)].reshape(2)
                    offset[i, :] = offset_pred[:, :, ys[i, 0].astype(int), xs[i, 0].astype(int)].reshape(2)
                    if config.ct_landmarks:
                        landmark_pred_1[i,:,0:1] = landmark_pred[:,:,0,ys[i, 0].astype(int),
                                                 xs[i, 0].astype(int)].reshape((5,1)) * wh_pred_1[i,1] + xs[i,0:1]
                        landmark_pred_1[i, :,1:2] = landmark_pred[:, :, 1, ys[i, 0].astype(int),
                                                   xs[i, 0].astype(int)].reshape((5, 1)) * wh_pred_1[i, 0] + ys[i,0:1]
                if config.ct_offset:
                    ys = ys + offset[:, 0:1]
                    # if flip:
                    #     xs = xs - offset[:, 1:2]
                    # else:
                    xs = xs + offset[:, 1:2]
                if flip:
                    xs = im.shape[1] // stride - xs
                if config.ct_wh_log:
                    wh_pred_1 = np.exp(wh_pred_1)
                ct_bboxes = np.concatenate((xs - wh_pred_1[:, 1:2] / 2,
                                      ys - wh_pred_1[:, 0:1] / 2,
                                      xs + wh_pred_1[:, 1:2] / 2,
                                      ys + wh_pred_1[:, 0:1] / 2),
                                      axis=1).astype(np.float32)

                ct_threshold = 0.36
                if self.vote:
                    ct_threshold = 0.02
                if config.ct_restriction:
                    res_inx = np.where(ct_scores < config.ct_restriction)[1]
                    ct_scores[:]= ct_scores[:] * 1.4
                    ct_scores[:,res_inx] = ct_scores[:,res_inx] / 5
                inx = np.where(ct_scores >= ct_threshold)[1]
                if len(inx) > 0:
                    ct_bboxes = ct_bboxes[inx, :] * stride
                    ct_scores = np.reshape(ct_scores, (ct_k, 1))
                    ct_scores = ct_scores[inx, :]
                    if config.ct_discard:
                        discard_idx = np.where(wh_pred_1 * stride > 64)[0]
                        ct_bboxes = np.delete(ct_bboxes, discard_idx, 0)

                        ct_scores = np.delete(ct_scores, discard_idx, 0)

                    if config.ct_landmarks:
                        ct_landmarks = landmark_pred_1[inx,:] * stride
                        ct_landmarks = np.delete(ct_landmarks, discard_idx, 0)
                        ct_landmarks_list.append(ct_landmarks / im_scale)
                    # if not flip:
                    ct_bbox_list.append(ct_bboxes / im_scale)
                    ct_score_list.append(ct_scores)
                    # if flip:
                    #     ct_bbox_flip_list.append(ct_bboxes / im_scale)
                    #     ct_score_flip_list.append(ct_scores)


    #toc = time.time()
    print((toc - tic)*1000)
    if self.debug:
      timeb = datetime.datetime.now()
      diff = timeb - timea
      print('B uses', diff.total_seconds(), 'seconds')
    if config.ct_reproduce:
        proposals = np.zeros((0, 5))
        scores = np.zeros((0, 1))
    else:
        proposals = np.vstack(proposals_list)
        scores = np.vstack(scores_list)
    if config.centernet_branch:
        if len(ct_bbox_list) > 0:
            ct_result = np.vstack(ct_bbox_list)
            ct_scores = np.vstack(ct_score_list)
            # if flip:
            #     ct_flip_result = np.vstack(ct_bbox_flip_list)
            #     ct_flip_scores = np.vstack(ct_score_flip_list)
            if config.ct_landmarks:
                ct_landmarks = np.vstack(ct_landmarks_list)
        else:
            ct_result = np.empty((0, 4)).astype(np.float32)
            ct_scores = np.empty((0, 1)).astype(np.float32)
            if config.ct_landmarks:
                ct_landmarks = np.empty((0,0,0)).astype(np.float32)
    else:
        ct_result = np.zeros((0,4))
    landmarks = None
    if proposals.shape[0]==0:
        proposals = np.zeros( (0,5) )
      # if self.use_landmarks:
      #   landmarks = np.zeros( (0,5,2) )
      # return np.zeros( (0,5) ), landmarks

    #print('shapes', proposals.shape, scores.shape)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    #np.partition()
    #if config.TEST.SCORE_THRESH>0.0:
    #  _count = np.sum(scores_ravel>config.TEST.SCORE_THRESH)
    #  order = order[:_count]
    proposals = proposals[order, :] # sort the proposals
    scores = scores[order]
    if not self.vote and self.use_landmarks:
      if config.ct_reproduce:
          landmarks = np.zeros((0,10))
      else:
          landmarks = np.vstack(landmarks_list)
          landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
    # if pre_det.shape[0] == 0:
    #     print('pre_det.shape[0] == 0')
    if config.USE_KLLOSS and config.var_voting:
        confidence = np.vstack(alpha_list)
        confidence = np.exp(confidence / 2.)
    if not self.vote:
      if pre_det.shape[0] >= 1:
          keep = self.nms(pre_det)
          #af_det, keep = self.soft_uncertainty(pre_det, threshold=threshold)
          if config.USE_SOFTNMS:
              #keep = self.softnms(pre_det[:, 0:4], pre_det[:, 4], thresh=threshold, method=2)
              keep = self.cpu_soft_nms(pre_det,threshold=threshold,method=2)
          if config.USE_KLLOSS:
            #keep = self.softnms(pre_det[:,0:4], pre_det[:,4],thresh=threshold,method=2)
            #print('soft')
            #keep = self.cpu_soft_nms(pre_det,threshold=threshold,method=2)
            af_det, keep = self.soft_uncertainty(pre_det, confidence=confidence, threshold=threshold,method=2)
          # keep = self.cpu_soft_nms(pre_det)
          det = np.hstack( (pre_det, proposals[:,4:]) )
          det = det[keep, :].astype(np.float32)
          # det = self.bbox_vote(det)
      else:
          det = pre_det
          keep = []
      if config.centernet_branch:
          ct_proposals = np.hstack((ct_result, ct_scores)).astype(np.float32)
          # ct_proposals = np.hstack((ct_flip_result, ct_flip_scores)).astype(np.float32)
          if ct_proposals.shape[0] > 0:
              ct_scores_ravel = ct_scores.ravel()
              ct_order = ct_scores_ravel.argsort()[::-1]
              ct_proposals = ct_proposals[ct_order, :]
              ct_keep = self.nms(ct_proposals)
              ct_proposals = ct_proposals[ct_keep,:]
              # ct_proposals = self.bbox_vote(ct_proposals)
              # cnt = 0
              # for ii in range(ct_proposals.shape[0]):
              #     bbox = ct_proposals[ii]
              #     if bbox[2] - bbox[0] <= 64:
              #         print(bbox[4], bbox[2] - bbox[0], bbox[3] - bbox[1])
              #         cnt+=1
              # print(cnt)


              #ct_result = ct_result[ct_keep,:]
          if config.ct_ab_concat_af:
              if ct_proposals.shape[0] == 0:
                  # print("ct_proposals.shape[0] == 0")
                  det = np.zeros( (0,5) )
              else:
                  det = np.vstack((det,ct_proposals)).astype(np.float32)
                  tmp_scores = det[:, 4]
                  tmp_scores_ravel = tmp_scores.ravel()
                  tmp_order = tmp_scores_ravel.argsort()[::-1]
                  det = det[tmp_order, :]
                  all_keep = self.nms(det)
                  det = det[all_keep, :]
                  #det = self.bbox_vote(det)
      else:
          ct_proposals = np.zeros((0,5))
      if config.USE_KLLOSS and config.var_voting:
          det = af_det
      if self.use_landmarks:
        landmarks = landmarks[keep]
    else:
      #print('vote')
      det = np.hstack( (pre_det, proposals[:,4:]) )
      if config.centernet_branch:
          #ct_scores = np.sqrt(ct_scores)
          ct_proposals = np.hstack((ct_result, ct_scores))
          ct_scores_ravel = ct_scores.ravel()
          ct_order = ct_scores_ravel.argsort()[::-1]
          ct_proposals = ct_proposals[ct_order,:]
          if config.ct_landmarks:
              ct_landmarks = ct_landmarks[ct_order,:]
          if ct_proposals.shape[0] > 0:
            # ct_keep = self.nms(ct_proposals)
            # # print(len(ct_keep))
            # ct_proposals = ct_proposals[ct_keep,:]
            # if config.ct_landmarks:
            #     ct_landmarks = ct_landmarks[ct_keep,:]
            # ct_result = ct_result[ct_keep,:]
            ct_proposals = self.bbox_vote(ct_proposals)
          # ct_scores_ravel = ct_scores.ravel()
          # ct_order = ct_scores_ravel.argsort()[::-1]
          # max_ct_score = ct_scores_ravel[ct_order[0]]
          # max_score = det[0,4]
          # ratio = max_score / max_ct_score
          # ct_proposals[:,4] = ct_proposals[:,4] * ratio
          # det = np.vstack((det, ct_proposals))
      else:
          ct_proposals = np.zeros((0,5))
      if config.ct_ab_concat_af:
        if det.shape[0] == 0:
            det = ct_proposals
        else:
            det = np.vstack((det, ct_proposals))
            tmp_scores = det[:,4]
            tmp_scores_ravel = tmp_scores.ravel()
            tmp_order = tmp_scores_ravel.argsort()[::-1]
            det = det[tmp_order,:]
      det = self.bbox_vote(det)
      if config.USE_KLLOSS and config.var_voting:
          det = np.hstack((pre_det, proposals[:, 4:]))
          det = self.bbox_vote_uncertainty(det,confidence)
    #if self.use_landmarks:
    #  det = np.hstack((det, landmarks))

    if self.debug:
      timeb = datetime.datetime.now()
      diff = timeb - timea
      print('C uses', diff.total_seconds(), 'seconds')
    #print(ct_proposals.shape)
    if config.ct_landmarks:
        return det,landmarks,ct_proposals,ct_landmarks
    else:
        return det, landmarks, ct_proposals

  def detect_center(self, img, threshold=0.5, scales=[1.0], do_flip=False):
    det, landmarks = self.detect(img, threshold, scales, do_flip)
    if det.shape[0]==0:
      return None, None
    bindex = 0
    if det.shape[0]>1:
      img_size = np.asarray(img.shape)[0:2]
      bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
      img_center = img_size / 2
      offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
      offset_dist_squared = np.sum(np.power(offsets,2.0),0)
      bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
    bbox = det[bindex,:]
    landmark = landmarks[bindex, :, :]
    return bbox, landmark

  @staticmethod
  def check_large_pose(landmark, bbox):
    assert landmark.shape==(5,2)
    assert len(bbox)==4
    def get_theta(base, x, y):
      vx = x-base
      vy = y-base
      vx[1] *= -1
      vy[1] *= -1
      tx = np.arctan2(vx[1], vx[0])
      ty = np.arctan2(vy[1], vy[0])
      d = ty-tx
      d = np.degrees(d)
      #print(vx, tx, vy, ty, d)
      #if d<-1.*math.pi:
      #  d+=2*math.pi
      #elif d>math.pi:
      #  d-=2*math.pi
      if d<-180.0:
        d+=360.
      elif d>180.0:
        d-=360.0
      return d
    landmark = landmark.astype(np.float32)

    theta1 = get_theta(landmark[0], landmark[3], landmark[2])
    theta2 = get_theta(landmark[1], landmark[2], landmark[4])
    #print(va, vb, theta2)
    theta3 = get_theta(landmark[0], landmark[2], landmark[1])
    theta4 = get_theta(landmark[1], landmark[0], landmark[2])
    theta5 = get_theta(landmark[3], landmark[4], landmark[2])
    theta6 = get_theta(landmark[4], landmark[2], landmark[3])
    theta7 = get_theta(landmark[3], landmark[2], landmark[0])
    theta8 = get_theta(landmark[4], landmark[1], landmark[2])
    #print(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8)
    left_score = 0.0
    right_score = 0.0
    up_score = 0.0
    down_score = 0.0
    if theta1<=0.0:
      left_score = 10.0
    elif theta2<=0.0:
      right_score = 10.0
    else:
      left_score = theta2/theta1
      right_score = theta1/theta2
    if theta3<=10.0 or theta4<=10.0:
      up_score = 10.0
    else:
      up_score = max(theta1/theta3, theta2/theta4)
    if theta5<=10.0 or theta6<=10.0:
      down_score = 10.0
    else:
      down_score = max(theta7/theta5, theta8/theta6)
    mleft = (landmark[0][0]+landmark[3][0])/2
    mright = (landmark[1][0]+landmark[4][0])/2
    box_center = ( (bbox[0]+bbox[2])/2,  (bbox[1]+bbox[3])/2 )
    ret = 0
    if left_score>=3.0:
      ret = 1
    if ret==0 and left_score>=2.0:
      if mright<=box_center[0]:
        ret = 1
    if ret==0 and right_score>=3.0:
      ret = 2
    if ret==0 and right_score>=2.0:
      if mleft>=box_center[0]:
        ret = 2
    if ret==0 and up_score>=2.0:
      ret = 3
    if ret==0 and down_score>=5.0:
      ret = 4
    return ret, left_score, right_score, up_score, down_score

  @staticmethod
  def _filter_boxes(boxes, min_size):
      """ Remove all boxes with any side smaller than min_size """
      ws = boxes[:, 2] - boxes[:, 0] + 1
      hs = boxes[:, 3] - boxes[:, 1] + 1
      keep = np.where((ws >= min_size) & (hs >= min_size))[0]
      return keep

  @staticmethod
  def _filter_boxes2(boxes, max_size, min_size):
      """ Remove all boxes with any side smaller than min_size """
      ws = boxes[:, 2] - boxes[:, 0] + 1
      hs = boxes[:, 3] - boxes[:, 1] + 1
      if max_size>0:
        keep = np.where( np.minimum(ws, hs)<max_size )[0]
      elif min_size>0:
        keep = np.where( np.maximum(ws, hs)>min_size )[0]
      return keep

  @staticmethod
  def _clip_pad(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
      H, W = tensor.shape[2:]
      h, w = pad_shape

      if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

      return tensor

  @staticmethod
  def bbox_pred(boxes, box_deltas):
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

      dx = box_deltas[:, 0:1]
      dy = box_deltas[:, 1:2]
      dw = box_deltas[:, 2:3]
      dh = box_deltas[:, 3:4]

      pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
      pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
      pred_w = np.exp(dw) * widths[:, np.newaxis]
      pred_h = np.exp(dh) * heights[:, np.newaxis]

      pred_boxes = np.zeros(box_deltas.shape)
      # x1
      pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
      # y1
      pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
      # x2
      pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
      # y2
      pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

      if box_deltas.shape[1]>4:
        pred_boxes[:,4:] = box_deltas[:,4:]

      return pred_boxes

  @staticmethod
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

  @staticmethod
  def landmark_pred(boxes, landmark_deltas):
      if boxes.shape[0] == 0:
          return np.zeros((0, landmark_deltas.shape[1]))
      boxes = boxes.astype(np.float, copy=False)
      widths = boxes[:, 2] - boxes[:, 0] + 1.0
      heights = boxes[:, 3] - boxes[:, 1] + 1.0
      ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
      ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
      pred = landmark_deltas.copy()
      for i in range(5):
        pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
        pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
      return pred

      #preds = []
      #for i in range(landmark_deltas.shape[1]):
      #  if i%2==0:
      #    pred = (landmark_deltas[:,i]*widths + ctr_x)
      #  else:
      #    pred = (landmark_deltas[:,i]*heights + ctr_y)
      #  preds.append(pred)
      #preds = np.vstack(preds).transpose()
      #return preds

  @staticmethod
  def landmark_pred_xyxy(boxes, landmark_deltas):
      if boxes.shape[0] == 0:
          return np.zeros((0, landmark_deltas.shape[1]))
      ex_l = boxes[:, 0]
      ex_r = boxes[:, 2]
      ex_d = boxes[:, 1]
      ex_u = boxes[:, 3]

      ex_widths = ex_r - ex_l + 1.0
      ex_heights = ex_u - ex_d + 1.0
      pred = landmark_deltas.copy()
      for i in range(5):
            if i % 3 == 0 or i % 3 == 2:
                pred[:,i,0] = landmark_deltas[:,i,0]*ex_widths + ex_l
                pred[:,i,1] = landmark_deltas[:,i,1]*ex_heights + ex_d
            elif i % 3 == 1:
                pred[:,i,0] = landmark_deltas[:,i,0]*ex_widths + ex_r
                pred[:, i, 1] = landmark_deltas[:, i, 1] * ex_heights + ex_u


      return pred

  @staticmethod
  def landmark_pred_af(boxes, landmark_deltas, xs,ys):
      if boxes.shape[0] == 0:
          return np.zeros((0, landmark_deltas.shape[1]))
      boxes = boxes.astype(np.float, copy=False)
      widths = boxes[:, 2] - boxes[:, 0] + 1.0
      heights = boxes[:, 3] - boxes[:, 1] + 1.0
      ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
      ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
      pred = landmark_deltas.copy()
      for i in range(5):
          pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
          pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
      return pred

  def bbox_vote(self, det):
      #order = det[:, 4].ravel().argsort()[::-1]
      #det = det[order, :]
      if det.shape[0] == 0:
          dets = np.array([[10, 10, 20, 20, 0.002]])
          det = np.empty(shape=[0, 5])
      while det.shape[0] > 0:
          # IOU
          area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
          xx1 = np.maximum(det[0, 0], det[:, 0])
          yy1 = np.maximum(det[0, 1], det[:, 1])
          xx2 = np.minimum(det[0, 2], det[:, 2])
          yy2 = np.minimum(det[0, 3], det[:, 3])
          w = np.maximum(0.0, xx2 - xx1 + 1)
          h = np.maximum(0.0, yy2 - yy1 + 1)
          inter = w * h
          o = inter / (area[0] + area[:] - inter)

          # nms
          merge_index = np.where(o >= self.nms_threshold)[0]
          det_accu = det[merge_index, :]
          det = np.delete(det, merge_index, 0)
          if merge_index.shape[0] <= 1:
              if det.shape[0] == 0:
                  try:
                      dets = np.row_stack((dets, det_accu))
                  except:
                      dets = det_accu
              continue
          det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
          max_score = np.max(det_accu[:, 4])
          det_accu_sum = np.zeros((1, 5))
          det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                        axis=0) / np.sum(det_accu[:, -1:])
          det_accu_sum[:, 4] = max_score
          try:
              dets = np.row_stack((dets, det_accu_sum))
          except:
              dets = det_accu_sum
      dets = dets[0:750, :]
      return dets

  def bbox_vote_uncertainty(self, det, confidence):
      #order = det[:, 4].ravel().argsort()[::-1]
      #det = det[order, :]
      if det.shape[0] == 0:
          dets = np.array([[10, 10, 20, 20, 0.002]])
          det = np.empty(shape=[0, 5])
      while det.shape[0] > 0:
          # IOU
          area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
          xx1 = np.maximum(det[0, 0], det[:, 0])
          yy1 = np.maximum(det[0, 1], det[:, 1])
          xx2 = np.minimum(det[0, 2], det[:, 2])
          yy2 = np.minimum(det[0, 3], det[:, 3])
          w = np.maximum(0.0, xx2 - xx1 + 1)
          h = np.maximum(0.0, yy2 - yy1 + 1)
          inter = w * h
          o = inter / (area[0] + area[:] - inter)

          # nms
          merge_index = np.where(o >= self.nms_threshold)[0]
          det_accu = det[merge_index, :]
          det = np.delete(det, merge_index, 0)

          ious = o[merge_index]#for var voting

          if merge_index.shape[0] <= 1:
              if det.shape[0] == 0:
                  try:
                      dets = np.row_stack((dets, det_accu))
                  except:
                      dets = det_accu
              continue
          det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
          if config.var_voting:
              p = np.exp(-(1 - ious) ** 2 / 0.01)
              #max_score = np.max(det_accu[:, 4])
              #det_accu_sum = np.zeros((1, 5))
              det_accu[:,0:4] = p.dot(det_accu[:,0:4] / (confidence[merge_index]) ** 2) / p.dot(
                  1. / (confidence[merge_index]) ** 2)
              #det_accu_sum[:, 4] = max_score
          max_score = np.max(det_accu[:, 4])
          det_accu_sum = np.zeros((1, 5))
          det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                        axis=0) / np.sum(det_accu[:, -1:])
          det_accu_sum[:, 4] = max_score


          try:
              dets = np.row_stack((dets, det_accu_sum))
          except:
              dets = det_accu_sum
      dets = dets[0:750, :]
      return dets

  @staticmethod
  def softnms(det, sc, Nt=0.4, sigma=0.5, thresh=0.02, method=2, confidence = None):

    # indexes concatenate boxes with the last column
    N = det.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((det, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        if config.USE_KLLOSS:
            tconfidence = confidence[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            if config.USE_KLLOSS:
                confidence[i, :] = confidence[maxpos +i +1, :]
                confidence[maxpos + i + 1, :] = tconfidence
                tconfidence = confidence[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # variance voting
        if config.USE_KLLOSS:
            p = np.exp(-(1 - ovr[i, np.arange(pos,N)]) ** 2 / 0.01)
            dets[i, :4] = p.dot(dets[np.arange(pos,N), :4] / confidence[np.arange(pos,N)] ** 2) / p.dot(1. / confidence[np.arange(pos,N)] ** 2)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = np.where(scores > thresh)[0]
    keep = inds.astype(int)
    #print(keep)

    return keep

  def cpu_soft_nms(self,boxes, sigma = 0.5, Nt = 0.4, threshold = 0.02,method = 2):
      N = boxes.shape[0]
      maxscore = 0
      maxpos = 0
      for i in range(N):
          maxscore = boxes[i, 4]
          maxpos = i

          tx1 = boxes[i, 0]
          ty1 = boxes[i, 1]
          tx2 = boxes[i, 2]
          ty2 = boxes[i, 3]
          ts = boxes[i, 4]

          pos = i + 1
          # get max box
          while pos < N:
              if maxscore < boxes[pos, 4]:
                  maxscore = boxes[pos, 4]
                  maxpos = pos
              pos = pos + 1

          # add max box as a detection
          boxes[i, 0] = boxes[maxpos, 0]
          boxes[i, 1] = boxes[maxpos, 1]
          boxes[i, 2] = boxes[maxpos, 2]
          boxes[i, 3] = boxes[maxpos, 3]
          boxes[i, 4] = boxes[maxpos, 4]

          # swap ith box with position of max box
          boxes[maxpos, 0] = tx1
          boxes[maxpos, 1] = ty1
          boxes[maxpos, 2] = tx2
          boxes[maxpos, 3] = ty2
          boxes[maxpos, 4] = ts

          tx1 = boxes[i, 0]
          ty1 = boxes[i, 1]
          tx2 = boxes[i, 2]
          ty2 = boxes[i, 3]
          ts = boxes[i, 4]

          pos = i + 1
          # NMS iterations, note that N changes if detection boxes fall below threshold
          while pos < N:
              x1 = boxes[pos, 0]
              y1 = boxes[pos, 1]
              x2 = boxes[pos, 2]
              y2 = boxes[pos, 3]
              s = boxes[pos, 4]

              area = (x2 - x1 + 1) * (y2 - y1 + 1)
              iw = (min(tx2, x2) - max(tx1, x1) + 1)
              if iw > 0:
                  ih = (min(ty2, y2) - max(ty1, y1) + 1)
                  if ih > 0:
                      ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                      ov = iw * ih / ua  # iou between max box and detection box

                      if method == 1:  # linear
                          if ov > Nt:
                              weight = 1 - ov
                          else:
                              weight = 1
                      elif method == 2:  # gaussian
                          weight = np.exp(-(ov * ov) / sigma)
                      else:  # original NMS
                          if ov > Nt:
                              weight = 0
                          else:
                              weight = 1

                      boxes[pos, 4] = weight * boxes[pos, 4]

                      # if box score falls below threshold, discard the box by swapping with last box
                      # update N
                      if boxes[pos, 4] < threshold:
                          boxes[pos, 0] = boxes[N - 1, 0]
                          boxes[pos, 1] = boxes[N - 1, 1]
                          boxes[pos, 2] = boxes[N - 1, 2]
                          boxes[pos, 3] = boxes[N - 1, 3]
                          boxes[pos, 4] = boxes[N - 1, 4]
                          N = N - 1
                          pos = pos - 1

              pos = pos + 1

      keep = [i for i in range(N)]
      return keep

  @staticmethod
  def soft_uncertainty(dets, confidence = None, iou_thresh = 0.4, threshold = 0.02, method = 2, ax=None):
      sigma = .5
      N = len(dets)
      x1 = dets[:, 0].copy()
      y1 = dets[:, 1].copy()
      x2 = dets[:, 2].copy()
      y2 = dets[:, 3].copy()
      scores = dets[:, 4].copy()
      areas = (x2 - x1 + 1) * (y2 - y1 + 1)
      ious = np.zeros((N, N))
      kls = np.zeros((N, N))
      for i in range(N):
          xx1 = np.maximum(x1[i], x1)
          yy1 = np.maximum(y1[i], y1)
          xx2 = np.minimum(x2[i], x2)
          yy2 = np.minimum(y2[i], y2)

          w = np.maximum(0.0, xx2 - xx1 + 1.)
          h = np.maximum(0.0, yy2 - yy1 + 1.)
          inter = w * h
          ovr = inter / (areas[i] + areas - inter)
          ious[i, :] = ovr.copy()

      i = 0
      while i < N:
          maxpos = dets[i:N, 4].argmax()
          maxpos += i
          dets[[maxpos, i]] = dets[[i, maxpos]]
          if confidence != None:
            confidence[[maxpos, i]] = confidence[[i, maxpos]]
          ious[[maxpos, i]] = ious[[i, maxpos]]
          ious[:, [maxpos, i]] = ious[:, [i, maxpos]]

          ovr_bbox = np.where((ious[i, i:N] > iou_thresh))[0] + i
          # if len(ovr_bbox) > 10:
          #     print('asd')
          if config.var_voting:
              p = np.exp(-(1 - ious[i, ovr_bbox]) ** 2 / 0.01)
              dets[i, :4] = p.dot(dets[ovr_bbox, :4] / confidence[ovr_bbox] ** 2) / p.dot(
                  1. / confidence[ovr_bbox] ** 2)

          pos = i + 1
          while pos < N:
              if ious[i, pos] > 0:
                  ovr = ious[i, pos]
                  if method > 2:
                      if ious[i, pos] > iou_thresh:
                          dets[pos, 4] = 0
                  elif method == 2:#Gaussian
                      dets[pos, 4] *= np.exp(-(ovr * ovr) / sigma)
                  else:#linear
                      if ovr > iou_thresh:
                          weight = 1 - ovr
                      else:
                          weight = 1
                      dets[pos,4] *= weight
                  if dets[pos, 4] < threshold:
                      dets[[pos, N - 1]] = dets[[N - 1, pos]]
                      if confidence != None:
                        confidence[[pos, N - 1]] = confidence[[N - 1, pos]]
                      ious[[pos, N - 1]] = ious[[N - 1, pos]]
                      ious[:, [pos, N - 1]] = ious[:, [N - 1, pos]]
                      N -= 1
                      pos -= 1
              pos += 1
          i += 1
      keep = [i for i in range(N)]
      return dets[keep], keep
