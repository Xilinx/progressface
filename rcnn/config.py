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
from easydict import EasyDict as edict

config = edict()

# network related params
#config.PIXEL_MEANS = np.array([123.68,116.779,103.939])
#config.PIXEL_STDS = np.array([58.393,57.13,57.375])
config.PIXEL_MEANS = np.array([0.0, 0.0, 0.0])
#config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
config.PIXEL_STDS = np.array([1.0, 1.0, 1.0])
config.PIXEL_SCALE = 1.0
config.IMAGE_STRIDE = 0

# dataset related params
config.NUM_CLASSES = 2
config.PRE_SCALES = [(1200, 1600)]  # first is scale (the shorter side); second is max size
config.SCALES = [(640, 640)]  # first is scale (the shorter side); second is max size
# config.SCALES = [(800, 800)]  # first is scale (the shorter side); second is max size
config.ORIGIN_SCALE = False

_ratio = (1.,)

'''RAC_SSH = {
    '32': {'SCALES': (12.70, 10.08, 8), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '16': {'SCALES': (6.35, 5.04, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '8': {'SCALES': (3.17, 2.52, 2), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
}'''
RAC_SSH = {
    '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
}
# RAC_SSH = {
#     '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
#     '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
# }
RAC_GP = {
    '32': {'SCALES': (8,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '16': {'SCALES': (4,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '8': {'SCALES': (2,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '4': {'SCALES': (1,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
}


_ratio = (1.,1.5)
RAC_SSH2 = {
    '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
}

_ratio = (1.,1.5)
RAC_SSH3 = {
    '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
    '4': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
}

RAC_RETINA = {}
_ratios = (1.0,)
_ass = 2.0**(1.0/3)
_basescale = 1.0
for _stride in [4, 8, 16, 32, 64]:
  key = str(_stride)
  value = {'BASE_SIZE': 16, 'RATIOS': _ratios, 'ALLOWED_BORDER': 9999}
  scales = []
  for _ in range(3):
    scales.append(_basescale)
    _basescale *= _ass
  value['SCALES'] = tuple(scales)
  RAC_RETINA[key] = value
  # if _stride == 4:
  #     value1 = value
  #     value1['SCALES'] = (0.6, 1, 1.25)
  #     RAC_RETINA[key] = value1

RAC_SRN = {}
_ratios = (1.0,)
_ass = 2.0**(1.0/3)
_basescale = 0.5
for _stride in [4, 8, 16, 32, 64, 128]:
  key = str(_stride)
  value = {'BASE_SIZE': 16, 'RATIOS': _ratios, 'ALLOWED_BORDER': 9999}
  scales = []
  for _ in range(3):
    scales.append(_basescale)
    _basescale *= _ass
  value['SCALES'] = tuple(scales)
  RAC_SRN[key] = value


config.RPN_ANCHOR_CFG = RAC_SSH #default

config.STR = False
config.STR_cls = False
config.STR_landmark = False
config.STC = False
config.STC_FPN_STRIDE = (64,32)
config.STR_FPN_STRIDE = (64,32)
config.head_module_after = 1 # if not head_module after get_sym_conv
config.use_gn = False

config.USE_FOCALLOSS = False
config.USE_SEBLOCK = False
config.USE_OHEGS = False # Online Hard Example Group Sampling
config.USE_KLLOSS = False
config.ft = False
config.landmark_klloss = False
config.USE_SOFTNMS = False
config.PROGRESSIVE = False
config.inherent = False
config.progressive_warm_up = False
config.selective_branch = False
config.USE_SERESNET = False
config.Progressive_v2 = False
config.progressive_manual = False
config.var_voting = False
config.try_end2end = False
config.bfp = False # Balanced Feature Pyramid from Libra R-CNN

config.cpm = False
config.deephead = False

config.centernet_branch = False
config.ct_scale = False # if ct_reproduce is true, this must be false
config.ct_stride = [8]
config.ct_wh_log =  False
config.ct_wh_inds = False # Use ind to process wh
config.max_objs = 1200
config.ct_context_module = False
config.ct_offset = False # add offset regression
config.ct_reproduce = False # screen anchor-based branch
config.ct_newtype = False
config.ct_landmarks = False
config.ct_ab_concat_af = False  # concat af to ab in retinaface.py
config.ct_discard = False # discard proposals which > 64
config.ct_restriction = False # float
config.ct_hmclasses = False # int
config.ct_ignore = False
config.ct_mining = False
config.ct_mining_stride = [8]
config.ct_head = False

# if config.ct_reproduce:
#     config.SCALES = [(800, 800)]
config.non_local = False
config.CBAM = False

config.NET_MODE = 2
config.HEAD_MODULE = 'SSH'
#config.HEAD_MODULE = 'RF'
#config.HEAD_MODULE = 'RFE'
#config.HEAD_MODULE = 'SSH_RFE'
config.LR_MODE = 0
config.LANDMARK_LR_MULT = 2.0
config.HEAD_FILTER_NUM = 256
config.CONTEXT_FILTER_RATIO = 1
config.max_feat_channel = 9999

config.USE_CROP = True
config.USE_DCN = 2
config.FACE_LANDMARK = True
config.USE_OCCLUSION = False
config.USE_BLUR = False
config.MORE_SMALL_BOX = True

config.LAYER_FIX = False

config.HEAD_BOX = False
config.DENSE_ANCHOR = False
config.USE_MAXOUT = 2
config.SHARE_WEIGHT_BBOX = False
config.SHARE_WEIGHT_LANDMARK = False
if config.deephead:
    config.SHARE_WEIGHT_BBOX = True
    config.SHARE_WEIGHT_LANDMARK = True

config.RANDOM_FEAT_STRIDE = False
config.NUM_CPU = 4
config.MIXUP = 0.0
config.USE_3D = False

#config.BBOX_MASK_THRESH = 0
config.COLOR_MODE = 2
config.COLOR_JITTERING = 0.125
#config.COLOR_JITTERING = 0
#config.COLOR_JITTERING = 0.2


config.TRAIN = edict()

config.TRAIN.IMAGE_ALIGN = 0
config.TRAIN.MIN_BOX_SIZE = 0
config.BBOX_MASK_THRESH = config.TRAIN.MIN_BOX_SIZE
# R-CNN and RPN
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 8
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = True
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = False

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_ENABLE_OHEM = 2
config.TRAIN.CASCADE_OVERLAP = [0.4,0.5]
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.25
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.5
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
if config.STR == True:
    config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_CLOBBER_POSITIVES = False
config.TRAIN.RPN_FORCE_POSITIVE = False
config.TRAIN.SCALE_COMPENSATION = False
config.TRAIN.SCALE_COMPENSATION_N = 1000
config.TRAIN.MAX_BBOX_PER_IMAGE = 2000
# rpn bounding box regression params
#config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
#config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
#config.TRAIN.RPN_LANDMARK_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#config.TRAIN.RPN_INVALID_LANDMARK_WEIGHTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# used for end2end training
# RPN proposal
#config.TRAIN.CXX_PROPOSAL = True
#config.TRAIN.RPN_NMS_THRESH = 0.7
#config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
#config.TRAIN.RPN_POST_NMS_TOP_N = 2000
#config.TRAIN.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
#config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
#config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
#config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 4

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.3
config.TEST.RPN_PRE_NMS_TOP_N = 1000
config.TEST.RPN_POST_NMS_TOP_N = 3000
#config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
#config.TEST.RPN_MIN_SIZE = [0,0,0]

# RCNN nms
config.TEST.NMS = 0.3

config.TEST.SCORE_THRESH = 0.05
config.TEST.IOU_THRESH = 0.5


# network settings
network = edict()

network.ssh = edict()

network.mnet = edict()
#network.mnet.pretrained = 'model/mnasnet'
#network.mnet.pretrained = 'model/mobilenetv2_0_5'
#network.mnet.pretrained = 'model/mobilenet_0_5'
#network.mnet.MULTIPLIER = 0.5
#network.mnet.pretrained = 'model/mobilenet_0_25'
#network.mnet.pretrained_epoch = 0
#network.mnet.PIXEL_MEANS = np.array([0.406, 0.456, 0.485])
#network.mnet.PIXEL_STDS = np.array([0.225, 0.224, 0.229])
#network.mnet.PIXEL_SCALE = 255.0
network.mnet.FIXED_PARAMS = ['^stage1', '^.*upsampling']
# if config.ct_reproduce:
#     network.mnet.FIXED_PARAMS = []
network.mnet.BATCH_IMAGES = 8
network.mnet.HEAD_FILTER_NUM = 64
network.mnet.CONTEXT_FILTER_RATIO = 1

#network.mnet.PIXEL_MEANS = np.array([123.68,116.779,103.939])
#network.mnet.PIXEL_STDS = np.array([58.393,57.13,57.375])
network.mnet.PIXEL_MEANS = np.array([0.0, 0.0, 0.0])
network.mnet.PIXEL_STDS = np.array([1.0, 1.0, 1.0])

network.mnet.PIXEL_SCALE = 1.0
#network.mnet.pretrained = 'model/mobilenetfd_0_25' #78
#network.mnet.pretrained = 'model/mobilenetfd2' #75
network.mnet.pretrained = 'model/mobilenet_0_25' #78
#network.mnet.pretrained = 'model/mobilenet025fd1' #75
#network.mnet.pretrained = 'model/mobilenet025fd2' #
network.mnet.pretrained_epoch = 0
network.mnet.max_feat_channel = 8888
network.mnet.COLOR_MODE = 1
network.mnet.USE_CROP = True
network.mnet.RPN_ANCHOR_CFG = RAC_SSH
if config.STR == True:
    network.mnet.RPN_ANCHOR_CFG = RAC_RETINA
network.mnet.LAYER_FIX = True
if config.ct_reproduce:
    network.mnet.LAYER_FIX = False
network.mnet.LANDMARK_LR_MULT = 2.5


network.resnet = edict()
#network.resnet.pretrained = 'model/ResNet50_v1d'
#network.resnet.pretrained = 'model/resnet-50'
network.resnet.pretrained = 'model/resnet-152'
#network.resnet.pretrained = 'model/senet154'
#network.resnet.pretrained = 'model/densenet161'
network.resnet.pretrained_epoch = 0
#network.mnet.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
#network.mnet.PIXEL_STDS = np.array([57.375, 57.12, 58.393])
#network.resnet.PIXEL_MEANS = np.array([0.406, 0.456, 0.485])
#network.resnet.PIXEL_STDS = np.array([0.225, 0.224, 0.229])
#network.resnet.PIXEL_SCALE = 255.0
network.resnet.lr_step = '1,2,3,4,5,55,68,80'
network.resnet.lr = 0.001
network.resnet.PIXEL_MEANS = np.array([0.0, 0.0, 0.0])
network.resnet.PIXEL_STDS = np.array([1.0, 1.0, 1.0])
network.resnet.PIXEL_SCALE = 1.0
network.resnet.FIXED_PARAMS = ['^stage1', '^.*upsampling']
network.resnet.BATCH_IMAGES = 4
network.resnet.HEAD_FILTER_NUM = 256
network.resnet.CONTEXT_FILTER_RATIO = 1
network.resnet.USE_DCN = 2
network.resnet.RPN_BATCH_SIZE = 256
network.resnet.RPN_ANCHOR_CFG = RAC_RETINA

'''network.resnet.USE_DCN = 1
network.resnet.pretrained = 'model/resnet-50'
network.resnet.RPN_ANCHOR_CFG = RAC_RETINA'''


# dataset settings
dataset = edict()

dataset.widerface = edict()
dataset.widerface.dataset = 'widerface'
dataset.widerface.image_set = 'train'
dataset.widerface.test_image_set = 'val'
dataset.widerface.root_path = 'data'
dataset.widerface.dataset_path = 'data/widerface'
dataset.widerface.NUM_CLASSES = 2

dataset.retinaface = edict()
dataset.retinaface.dataset = 'retinaface'
dataset.retinaface.image_set = 'train'
dataset.retinaface.test_image_set = 'val'
dataset.retinaface.root_path = 'data'
dataset.retinaface.dataset_path = 'data/retinaface'
dataset.retinaface.NUM_CLASSES = 2

# default settings
default = edict()

config.FIXED_PARAMS = ['^conv1', '^conv2', '^conv3', '^.*upsampling']
# if config.ct_reproduce:
#     config.FIXED_PARAMS = []
#config.FIXED_PARAMS = ['^.*upsampling']
#config.FIXED_PARAMS = ['^conv1', '^conv2', '^conv3']
#config.FIXED_PARAMS = ['^conv0', '^stage1', 'gamma', 'beta']  #for resnet

# default network
default.network = 'resnet'
default.pretrained = 'model/resnet-152'
#default.network = 'resnetssh'
default.pretrained_epoch = 0
# default dataset
default.dataset = 'retinaface'
default.image_set = 'train'
default.test_image_set = 'val'
default.root_path = 'data'
default.dataset_path = 'data/retinaface'
# default training
default.frequent = 20
default.kvstore = 'device'
# default e2e
default.prefix = 'model/retinaface'
default.end_epoch = 100
default.lr_step = '55,68,80'

default.lr = 0.01


def generate_config(_network, _dataset):
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
        if k in config.TRAIN:
          config.TRAIN[k] = v
    for k, v in dataset[_dataset].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
        if k in config.TRAIN:
          config.TRAIN[k] = v
    config.network = _network
    config.dataset = _dataset
    config.RPN_FEAT_STRIDE = []
    num_anchors = []
    for k in config.RPN_ANCHOR_CFG:
      config.RPN_FEAT_STRIDE.append( int(k) )
      _num_anchors = len(config.RPN_ANCHOR_CFG[k]['SCALES'])*len(config.RPN_ANCHOR_CFG[k]['RATIOS'])
      if config.DENSE_ANCHOR:
        _num_anchors *= 2
      config.RPN_ANCHOR_CFG[k]['NUM_ANCHORS'] = _num_anchors
      num_anchors.append(_num_anchors)
    config.RPN_FEAT_STRIDE = sorted(config.RPN_FEAT_STRIDE, reverse=True)
    for j in range(1,len(num_anchors)):
      assert num_anchors[0]==num_anchors[j]
    config.NUM_ANCHORS = num_anchors[0]

