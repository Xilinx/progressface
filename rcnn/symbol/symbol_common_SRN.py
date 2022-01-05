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
import mxnet.ndarray as nd
import numpy as np
from rcnn.config import config
from rcnn.PY_OP import rpn_fpn_ohem3
from rcnn.PY_OP import STC_op
from rcnn.symbol.focal_loss_optimizedversion import *
from rcnn.PY_OP.refine_anchor_generator import *
from rcnn.PY_OP.gt_boxes_reshape import *
from rcnn.PY_OP.STR_op import *
from rcnn.PY_OP.get_STR_label import *
from rcnn.config import default

isize = 640
batch_size = 1

def conv_only(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), bias_wd_mult=0.0, shared_weight=None, shared_bias = None):
  if shared_weight is None:
    weight = mx.symbol.Variable(name="{}_weight".format(name),
        init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    bias = mx.symbol.Variable(name="{}_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
  else:
    weight = shared_weight
    bias = shared_bias
    print('reuse shared var in', name)
  conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
      stride=stride, num_filter=num_filter, name="{}".format(name), weight = weight, bias=bias)
  return conv

def conv_deformable(net, num_filter, num_group=1, act_type='relu',name=''):
  if config.USE_DCN==1:
    f = num_group*18
    conv_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = net,
                        num_filter=f, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
    net = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=net, offset=conv_offset,
                        num_filter=num_filter, pad=(1,1), kernel=(3, 3), num_deformable_group=num_group, stride=(1, 1), no_bias=False)
  else:
    print('use dcnv2 at', name)
    lr_mult = 0.1
    weight_var = mx.sym.Variable(name=name+'_conv2_offset_weight', init=mx.init.Zero(), lr_mult=lr_mult)
    bias_var = mx.sym.Variable(name=name+'_conv2_offset_bias', init=mx.init.Zero(), lr_mult=lr_mult)
    conv2_offset = mx.symbol.Convolution(name=name + '_conv2_offset', data=net, num_filter=27,
      pad=(1, 1), kernel=(3, 3), stride=(1,1), weight=weight_var, bias=bias_var, lr_mult=lr_mult)
    conv2_offset_t = mx.sym.slice_axis(conv2_offset, axis=1, begin=0, end=18)
    conv2_mask =  mx.sym.slice_axis(conv2_offset, axis=1, begin=18, end=None)
    conv2_mask = 2 * mx.sym.Activation(conv2_mask, act_type='sigmoid')

    conv2 = mx.contrib.symbol.ModulatedDeformableConvolution(name=name + '_conv2', data=net, offset=conv2_offset_t, mask=conv2_mask,
        num_filter=num_filter, pad=(1, 1), kernel=(3, 3), stride=(1,1),
        num_deformable_group=num_group, no_bias=True)
    net = conv2
  net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
  if len(act_type)>0:
    net = mx.symbol.Activation(data=net, act_type=act_type, name=name+'_act')
  return net

def conv_act_layer_dw(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", bias_wd_mult=0.0):
    assert kernel[0]==3
    weight = mx.symbol.Variable(name="{}_weight".format(name),
        init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    bias = mx.symbol.Variable(name="{}_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, num_group=num_filter, name="{}".format(name), weight=weight, bias=bias)
    conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
    if len(act_type)>0:
      relu = mx.symbol.Activation(data=conv, act_type=act_type, \
          name="{}_{}".format(name, act_type))
    else:
      relu = conv
    return relu

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", bias_wd_mult=0.0, separable=False, filter_in = -1):

    if config.USE_DCN>1 and kernel==(3,3) and pad==(1,1) and stride==(1,1) and not separable:
      return conv_deformable(from_layer, num_filter, num_group=1, act_type = act_type, name=name)

    if separable:
      assert kernel[0]>1
      assert filter_in>0
    if not separable:
      weight = mx.symbol.Variable(name="{}_weight".format(name),
          init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
      bias = mx.symbol.Variable(name="{}_bias".format(name),
          init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
      conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
          stride=stride, num_filter=num_filter, name="{}".format(name), weight=weight, bias=bias)
      conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
    else:
      if filter_in<0:
        filter_in = num_filter
      conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
          stride=stride, num_filter=filter_in, num_group=filter_in, name="{}_sep".format(name))
      conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_sep_bn')
      conv = mx.symbol.Activation(data=conv, act_type='relu', \
          name="{}_sep_bn_relu".format(name))
      conv = mx.symbol.Convolution(data=conv, kernel=(1,1), pad=(0,0), \
          stride=(1,1), num_filter=num_filter, name="{}".format(name))
      conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
    if len(act_type)>0:
      relu = mx.symbol.Activation(data=conv, act_type=act_type, \
          name="{}_{}".format(name, act_type))
    else:
      relu = conv
    return relu

def ssh_context_module(body, num_filter, filter_in, name):
  conv_dimred = conv_act_layer(body, name+'_conv1',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False, filter_in = filter_in)
  conv5x5 = conv_act_layer(conv_dimred, name+'_conv2',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=False)
  conv7x7_1 = conv_act_layer(conv_dimred, name+'_conv3_1',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False)
  conv7x7 = conv_act_layer(conv7x7_1, name+'_conv3_2',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=False)
  return (conv5x5, conv7x7)


def ssh_detection_module(body, num_filter, filter_in, name):
  assert num_filter%4==0
  conv3x3 = conv_act_layer(body, name+'_conv1',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=False, filter_in=filter_in)
  #_filter = max(num_filter//4, 16)
  _filter = num_filter//4
  conv5x5, conv7x7 = ssh_context_module(body, _filter, filter_in, name+'_context')
  ret = mx.sym.concat(*[conv3x3, conv5x5, conv7x7], dim=1, name = name+'_concat')
  ret = mx.symbol.Activation(data=ret, act_type='relu', name=name+'_concat_relu')
  out_filter = num_filter//2+_filter*2
  if config.USE_DCN>0:
    ret = conv_deformable(ret, num_filter = out_filter, name = name+'_concat_dcn')
  return ret

def ssh_detection_RFE_module(body, num_filter, filter_in, name):
  assert num_filter%4==0
  conv3x3 = conv_act_layer(body, name+'_conv1',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=False, filter_in=filter_in)
  #_filter = max(num_filter//4, 16)
  conv3x3_1 = RFE_module(conv3x3,num_filter//2,filter_in,name+'_conv1_RFE')
  _filter = num_filter//4
  conv5x5, conv7x7 = ssh_context_module(body, _filter, filter_in, name+'_context')
  conv5x5_1 = RFE_module(conv5x5,_filter,filter_in,name+'_context_RFE1')
  conv7x7_1 = RFE_module(conv7x7,_filter,filter_in,name+'_context_RFE2')
  ret = mx.sym.concat(*[conv3x3_1, conv5x5_1, conv7x7_1], dim=1, name = name+'_concat')
  ret = mx.symbol.Activation(data=ret, act_type='relu', name=name+'_concat_relu')
  out_filter = num_filter//2+_filter*2
  if config.USE_DCN>0:
    ret = conv_deformable(ret, num_filter = out_filter, name = name+'_concat_dcn')
  return ret

def RFE_module(body, num_filter, filter_in, name):
    conv1 = conv_act_layer(body, name + '_RFE_conv1', num_filter // 4, kernel=(1,1), pad=(0,0), stride=(1,1))
    conv2 = conv_act_layer(body, name + '_RFE_conv2', num_filter // 4, kernel=(1, 1), pad=(0,0), stride=(1, 1))
    conv3 = conv_act_layer(body, name + '_RFE_conv3', num_filter // 4, kernel=(1, 1), pad=(0,0), stride=(1, 1))
    conv4 = conv_act_layer(body, name + '_RFE_conv4', num_filter // 4, kernel=(1, 1), pad=(0,0), stride=(1, 1))
    conv1_1 = conv_act_layer(conv1, name + '_RFE_conv1_1', num_filter // 4, kernel=(3, 1), pad=(1,0), stride=(1, 1))
    conv1_2 = conv_act_layer(conv1_1, name + '_RFE_conv1_2', num_filter // 4, kernel=(1,1), pad=(0,0), stride=(1,1))
    conv2_1 = conv_act_layer(conv2, name + '_RFE_conv2_1', num_filter // 4, kernel=(1, 3), pad=(0,1), stride=(1, 1))
    conv2_2 = conv_act_layer(conv2_1, name + '_RFE_conv2_2', num_filter // 4, kernel=(1, 1), pad=(0,0), stride=(1, 1))
    conv3_1 = conv_act_layer(conv3, name + '_RFE_conv3_1', num_filter // 4, kernel=(5, 1), pad=(2,0), stride=(1, 1))
    conv3_2 = conv_act_layer(conv3_1, name + '_RFE_conv3_2', num_filter // 4, kernel=(1, 1), pad=(0,0), stride=(1, 1))
    conv4_1 = conv_act_layer(conv4, name + '_RFE_conv4_1', num_filter // 4, kernel=(1, 5), pad=(0,2), stride=(1, 1))
    conv4_2 = conv_act_layer(conv4_1, name + '_RFE_conv4_2', num_filter // 4, kernel=(1, 1), pad=(0,0), stride=(1, 1))
    conv = mx.sym.concat(*[conv1_2, conv2_2, conv3_2, conv4_2], dim=1, name=name + '_concat')
    ret0 = conv_act_layer(conv, name + '_RFE', num_filter, kernel=(1,1), pad=(0,0), stride=(1,1))
    ret0 = mx.symbol.Crop(*[ret0, body])
    ret = ret0 + body
    return ret


#def retina_context_module(body, kernel, num_filter, filter_in, name):
#  conv_dimred = conv_act_layer(body, name+'_conv0',
#      num_filter, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False, filter_in = filter_in)
#  conv1 = conv_act_layer(conv_dimred, name+'_conv1',
#      num_filter*6, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False, filter_in = filter_in)
#  conv2 = conv_act_layer(conv1, name+'_conv2',
#      num_filter*6, kernel=kernel, pad=((kernel[0]-1)//2, (kernel[1]-1)//2), stride=(1, 1), act_type='relu', separable=True, filter_in = num_filter*6)
#  conv3 = conv_act_layer(conv2, name+'_conv3',
#      num_filter, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False)
#  conv3 = conv3 + conv_dimred
#  return conv3

def retina_detection_module(body, num_filter, filter_in, name):
  assert num_filter%4==0
  conv1 = conv_act_layer(body, name+'_conv1',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False, filter_in=filter_in)
  conv2 = conv_act_layer(conv1, name+'_conv2',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False, filter_in=num_filter//2)
  conv3 = conv_act_layer(conv2, name+'_conv3',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False, filter_in=num_filter//2)
  conv4 = conv2 + conv3
  body = mx.sym.concat(*[conv1, conv4], dim=1, name = name+'_concat')
  if config.USE_DCN>0:
    body = conv_deformable(body, num_filter = num_filter, name = name+'_concat_dcn')
  return body

def retina_detection_RFE_module(body, num_filter, filter_in, name):
    assert num_filter % 4 == 0
    conv1 = conv_act_layer(body, name + '_conv1',
                           num_filter // 2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False,
                           filter_in=filter_in)
    conv1_1 = RFE_module(conv1, num_filter//2, filter_in, name + '_conv1_1')
    conv2 = conv_act_layer(conv1, name + '_conv2',
                           num_filter // 2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False,
                           filter_in=num_filter // 2)
    conv2_1 = RFE_module(conv2, num_filter // 2, filter_in//2, name + '_conv2_1')
    conv3 = conv_act_layer(conv2, name + '_conv3',
                           num_filter // 2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False,
                           filter_in=num_filter // 2)
    conv3_1 = RFE_module(conv3, num_filter // 2, filter_in//2, name + '_conv3_1')
    conv4 = conv2_1 + conv3_1
    body = mx.sym.concat(*[conv1_1, conv4], dim=1, name=name + '_concat')
    if config.USE_DCN > 0:
        body = conv_deformable(body, num_filter=num_filter, name=name + '_concat_dcn')
    return body


def head_module(body, num_filter, filter_in, name):
  if config.HEAD_MODULE=='SSH':
    return ssh_detection_module(body, num_filter, filter_in, name)
  elif config.HEAD_MODULE == 'RFE':
    return  retina_detection_RFE_module(body,num_filter,filter_in,name)
  elif config.HEAD_MODULE == 'SSH_RFE':
      return ssh_detection_RFE_module(body, num_filter, filter_in, name)
  else:
    return retina_detection_module(body, num_filter, filter_in, name)


def upsampling(data, num_filter, name):
    #ret = mx.symbol.Deconvolution(data=data, num_filter=num_filter, kernel=(4,4),  stride=(2, 2), pad=(1,1),
    #    num_group = num_filter, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
    #    name=name)
    #ret = mx.symbol.Deconvolution(data=data, num_filter=num_filter, kernel=(2,2),  stride=(2, 2), pad=(0,0),
    #    num_group = num_filter, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
    #    name=name)
    ret = mx.symbol.UpSampling(data, scale=2, sample_type='nearest', workspace=512, name=name, num_args=1)
    return ret

def get_sym_conv(data, sym):
    all_layers = sym.get_internals()

    _, out_shape, _ = all_layers.infer_shape(data = (1,3,isize,isize))
    last_entry = None
    c1 = None
    c2 = None
    c3 = None
    c1_name = None
    c2_name = None
    c3_name = None
    c1_filter = -1
    c2_filter = -1
    c3_filter = -1
    #print(len(all_layers), len(out_shape))
    #print(all_layers.__class__)
    outputs = all_layers.list_outputs()
    #print(outputs.__class__, len(outputs))
    count = len(outputs)
    stride2name = {}
    stride2layer = {}
    stride2shape = {}
    for i in range(count):
      name = outputs[i]
      shape = out_shape[i]
      if not name.endswith('_output'):
        continue
      if len(shape)!=4:
        continue
      #assert isize%shape[2]==0
      if shape[1]>config.max_feat_channel:
        break
      stride = isize//shape[2]
      stride2name[stride] = name
      stride2layer[stride] = all_layers[name]
      stride2shape[stride] = shape
      #print(name, shape)
      #if c1 is None and shape[2]==isize//16:
      #  cname = last_entry[0]
      #  #print('c1', last_entry)
      #  c1 = all_layers[cname]
      #  c1_name = cname
      #if c2 is None and shape[2]==isize//32:
      #  cname = last_entry[0]
      #  #print('c2', last_entry)
      #  c2 = all_layers[cname]
      #  c2_name = cname
      #if shape[2]==isize//32:
      #  c3 = all_layers[name]
      #  #print('c3', name, shape)
      #  c3_name = name

      #last_entry = (name, shape)

    F1 = config.HEAD_FILTER_NUM
    F2 = F1
    if config.SHARE_WEIGHT_BBOX or config.SHARE_WEIGHT_LANDMARK:
      F2 = F1
    strides = sorted(stride2name.keys())
    for stride in strides:
      print('stride', stride, stride2name[stride], stride2shape[stride])
    print('F1_F2', F1, F2)
    #print('cnames', c1_name, c2_name, c3_name, F1, F2)
    _bwm = 1.0
    c0 = stride2layer[4]
    c1 = stride2layer[8]
    c2 = stride2layer[16]
    c3 = stride2layer[32]

    ########define layers of STC & STR first stage##############
    STC_out = {}
    STR_out = {}
    if config.STC:
        for stride in config.STC_FPN_STRIDE:
            STC_out[stride] = stride2layer[stride]

    ###########################################################

    c3 = conv_act_layer(c3, 'rf_c3_lateral',
        F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    #c3_up = mx.symbol.UpSampling(c3, scale=2, sample_type='nearest', workspace=512, name='ssh_c3_up', num_args=1)
    c3_up = upsampling(c3, F2, 'rf_c3_upsampling')
    c2_lateral = conv_act_layer(c2, 'rf_c2_lateral',
        F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    if config.USE_CROP:
      c3_up = mx.symbol.Crop(*[c3_up, c2_lateral])
    c2 = c2_lateral+c3_up
    c2 = conv_act_layer(c2, 'rf_c2_aggr',
        F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    c1_lateral = conv_act_layer(c1, 'rf_c1_red_conv',
        F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    #c2_up = mx.symbol.UpSampling(c2, scale=2, sample_type='nearest', workspace=512, name='ssh_m2_red_up', num_args=1)
    c2_up = upsampling(c2, F2, 'rf_c2_upsampling')
    #conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
    if config.USE_CROP:
      c2_up = mx.symbol.Crop(*[c2_up, c1_lateral])
    c1 = c1_lateral+c2_up
    c1 = conv_act_layer(c1, 'rf_c1_aggr',
        F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    m1 = head_module(c1, F2*config.CONTEXT_FILTER_RATIO, F2, 'rf_c1_det')
    m2 = head_module(c2, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c2_det')
    m3 = head_module(c3, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c3_det')
    if len(config.RPN_ANCHOR_CFG)==3:
      ret = {8: m1, 16:m2, 32: m3}
    elif len(config.RPN_ANCHOR_CFG)==1:
      ret = {16:m2}
    elif len(config.RPN_ANCHOR_CFG)==2:
      ret = {8: m1, 16:m2}
    elif len(config.RPN_ANCHOR_CFG)==4:
      c0_lateral = conv_act_layer(c0, 'rf_c0_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_up = upsampling(c1, F2, 'rf_c1_upsampling')
      if config.USE_CROP:
        c1_up = mx.symbol.Crop(*[c1_up, c0_lateral])
      c0 = c0_lateral+c1_up
      c0 = conv_act_layer(c0, 'rf_c0_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      m0 = head_module(c0, F2*config.CONTEXT_FILTER_RATIO, F2, 'rf_c0_det')
      ret = {4: m0, 8: m1, 16:m2, 32: m3}
    elif len(config.RPN_ANCHOR_CFG)==5:
      c0_lateral = conv_act_layer(c0, 'rf_c0_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_up = upsampling(c1, F2, 'rf_c1_upsampling')
      if config.USE_CROP:
        c1_up = mx.symbol.Crop(*[c1_up, c0_lateral])
      c0 = c0_lateral+c1_up
      c0 = conv_act_layer(c0, 'rf_c0_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      c4 = conv_act_layer(c3, 'rf_c4',
          F2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu', bias_wd_mult=_bwm)

      if config.STR:
          for stride in config.STR_FPN_STRIDE:
              if stride == 32:
                  STR_out[stride] = c3
              elif stride == 64:
                  STR_out[stride] = c4


      m0 = head_module(c0, F2*config.CONTEXT_FILTER_RATIO, F2, 'rf_c0_det')
      m4 = head_module(c4, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c4_det')
      ret = {4: m0, 8: m1, 16:m2, 32: m3, 64: m4}
    elif len(config.RPN_ANCHOR_CFG)==6:
      c0_lateral = conv_act_layer(c0, 'rf_c0_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_up = upsampling(c1, F2, 'rf_c1_upsampling')
      if config.USE_CROP:
        c1_up = mx.symbol.Crop(*[c1_up, c0_lateral])
      c0 = c0_lateral+c1_up
      c0 = conv_act_layer(c0, 'rf_c0_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      c4 = conv_act_layer(c3, 'rf_c4',
          F2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu', bias_wd_mult=_bwm)
      c5 = conv_act_layer(c4, 'rf_c5',
          F2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu', bias_wd_mult=_bwm)

      if config.STR:
          for stride in config.STR_FPN_STRIDE:
              if stride == 32:
                  STR_out[stride] = c3
              elif stride == 64:
                  STR_out[stride] = c4
              elif stride == 128:
                  STR_out[stride] = c5

      m0 = head_module(c0, F2*config.CONTEXT_FILTER_RATIO, F2, 'rf_c0_det')
      m4 = head_module(c4, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c4_det')
      m5 = head_module(c5, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c5_det')
      ret = {4: m0, 8: m1, 16:m2, 32: m3, 64: m4, 128: m5}

    #return {8: m1, 16:m2, 32: m3}
    return ret, STC_out, STR_out

def STC(conv_fpn_fs, prefix, stride, landmark = False, lr_mult=1.0, shared_vars = None):
    A = config.NUM_ANCHORS
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
        bbox_pred_len = 5
    if config.USE_OCCLUSION:
        landmark_pred_len = 15
    ret_group = []
    num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
    label = mx.symbol.Variable(name='%s_label_stride%d' % (prefix, stride))
    bbox_weight = mx.symbol.Variable(name='%s_bbox_weight_stride%d' % (prefix, stride))
    if landmark:
        landmark_weight = mx.symbol.Variable(name='%s_landmark_weight_stride%d' % (prefix, stride))
    rpn_relu = conv_fpn_fs[stride]

    rpn_cls_score = conv_only(rpn_relu, 'STC_%s_rpn_cls_score_stride%d' % (prefix, stride), 2 * num_anchors,
                                  kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=shared_vars[0][0],
                                  shared_bias=shared_vars[0][1])
    '''arg_shapes, out_shapes, aux_shapes = rpn_cls_score.infer_shape(data=(1, 3, 640, 640))
    print("rpn_cls_")
    print(out_shapes)'''

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                              shape=(0, 2, -1),
                                              name="STC_rpn_cls_score_reshape_stride%s" %stride)
    if config.TRAIN.RPN_ENABLE_OHEM >= 2:
        label1, anchor_weight1, valid_count1 = mx.sym.Custom(op_type='rpn_fpn_ohem3', stride=int(stride),
                                                          network=config.network, dataset=config.dataset, prefix=prefix,
                                                          cls_score=rpn_cls_score_reshape, labels=label)
    # cls loss
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape,
                                           label=label1,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           grad_scale=lr_mult,
                                           name='STC_%s' % stride)

    ret_group.append(rpn_cls_prob)
    ret_group.append(mx.sym.BlockGrad(label1))

    '''anchors = mx.contrib.symbol.MultiBoxPrior(rpn_relu, sizes=[0.2,0.5], ratios=1, clip=0, \
                                              name="{}_anchors".format(rpn_relu))
    arg_shapes, out_shapes, aux_shapes = rpn_relu.infer_shape(data=(1, 3, 640, 640))
    print("rpn_relu_")
    print(out_shapes)
    arg_shapes, out_shapes, aux_shapes = anchors.infer_shape(data=(1, 3, 640, 640))
    print("anchor_")
    print(out_shapes)'''

    label, anchor_weight, valid_count = mx.sym.Custom(op_type='STC_op', stride=int(stride),
                                                      network=config.network, dataset=config.dataset, prefix=prefix,
                                                      cls_score=rpn_cls_prob, labels=label)
    _bbox_weight = mx.sym.tile(anchor_weight, (1, 1, bbox_pred_len))
    _bbox_weight = _bbox_weight.reshape((0, -1, A * bbox_pred_len)).transpose((0, 2, 1))
    bbox_weight = mx.sym.elemwise_mul(bbox_weight, _bbox_weight,
                                      name='%s_bbox_weight_mul_stride%s' % (prefix, stride))
    if landmark:
        _landmark_weight = mx.sym.tile(anchor_weight, (1, 1, landmark_pred_len))
        _landmark_weight = _landmark_weight.reshape((0, -1, A * landmark_pred_len)).transpose((0, 2, 1))
        landmark_weight = mx.sym.elemwise_mul(landmark_weight, _landmark_weight,
                                              name='%s_landmark_weight_mul_stride%s' % (prefix, stride))

    return label,bbox_weight,landmark_weight,ret_group

def STR(conv_fpn_fs, prefix, stride, landmark = False, lr_mult=1.0, shared_vars = None):
    print("use STR")
    A = config.NUM_ANCHORS
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
        bbox_pred_len = 5
    if config.USE_OCCLUSION:
        landmark_pred_len = 15
    ret_group = []

    num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
    #label = mx.symbol.Variable(name='%s_label_stride%d' % (prefix, stride))
    bbox_target = mx.symbol.Variable(name='%s_bbox_target_stride%d' % (prefix, stride))
    bbox_weight = mx.symbol.Variable(name='%s_bbox_weight_stride%d' % (prefix, stride))
    if landmark:
        landmark_target = mx.symbol.Variable(name='%s_landmark_target_stride%d' % (prefix, stride))
        landmark_weight = mx.symbol.Variable(name='%s_landmark_weight_stride%d' % (prefix, stride))
    rpn_relu = conv_fpn_fs[stride]

    rpn_bbox_pred = conv_only(rpn_relu, 'STR_%s_rpn_bbox_pred_stride%d' % (prefix, stride), bbox_pred_len * num_anchors,
                              kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=shared_vars[1][0],
                              shared_bias=shared_vars[1][1])
    if landmark:
      rpn_landmark_pred = conv_only(rpn_relu, 'STR_%s_rpn_landmark_pred_stride%d'%(prefix,stride), landmark_pred_len*num_anchors,
          kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[2][0], shared_bias = shared_vars[2][1])
      rpn_landmark_pred_reshape = mx.symbol.Reshape(data=rpn_landmark_pred,
                                              shape=(0, 0, -1),
                                              name="STR_%s_rpn_landmark_pred_reshape_stride%s" % (prefix,stride))

    rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                              shape=(0, 0, -1),
                                              name="STR_%s_rpn_bbox_pred_reshape_stride%s" % (prefix, stride))
    #arg_shapes, out_shapes, aux_shapes = rpn_bbox_pred_reshape.infer_shape(data=(batch_size, 3, 640, 640))#(1,12,400)
    #arg_shapes, out_shapes, aux_shapes = rpn_landmark_pred_reshape.infer_shape(data=(batch_size, 3, 640, 640))#(1,30,400)
    ##refine anchors###
    offset = 0
    sstride = str(stride)
    base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
    allowed_border = config.RPN_ANCHOR_CFG[sstride]['ALLOWED_BORDER']
    ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
    scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
    feat_height = isize/stride
    feat_width = isize/stride
    #a = mx.nd.random.uniform(shape=(1, 3, feat_height, feat_width))
    scalesmy = []
    for i in range(len(scales)):
        scalesmy.append(scales[i] * 16 / 640)
    ##reproduce initial anchors###
    anchors_nd = mx.contrib.symbol.MultiBoxPrior(data=rpn_relu, sizes=scalesmy,
                                             steps=(float(1 / feat_height), float(1 / feat_width)), \
                                             offsets=(offset, offset))
    arg_shapes, out_shapes, aux_shapes = rpn_bbox_pred_reshape.infer_shape(data=(batch_size, 3, 640, 640))
    anchors_nd.__add__(0.01171875) # 7.5/640
    anchors_no_norm = anchors_nd
    anchors_no_norm = anchors_no_norm.__mul__(isize)

    refined_anchors = mx.symbol.Custom(arm_anchor_boxes=anchors_no_norm, arm_loc_preds=rpn_bbox_pred_reshape, landmarks_target_preds=rpn_landmark_pred_reshape, \
                                        op_type='refine_anchor_generator')

    # refined_anchors_norm = refined_anchors.__div__(isize)
    #
    # #arg_shapes, out_shapes = refined_anchors.infer_shape(arm_anchor_boxes_shape=(1,A*feat_height*feat_width,4),arm_loc_preds_shape = out_shapes[0])
    # cls_pred = mx.symbol.random.uniform(shape=(batch_size, 2, A*feat_width*feat_height))
    # gt_boxes = mx.symbol.Variable(name="gt_boxes")
    # gt_output = mx.symbol.Custom(gt_boxes=gt_boxes, op_type='gt_boxes_reshape')
    #
    # refined_anchors_norm_bs = mx.sym.split(data=refined_anchors_norm, axis=0, num_outputs=batch_size)
    # tmp_bbox_target = []
    # tmp_bbox_weight = []
    # tmp_label = []
    # label_bs = mx.sym.split(data=gt_output, axis=0, num_outputs=batch_size)
    # cls_preds_bs = mx.sym.split(data=cls_pred, axis=0, num_outputs=batch_size)
    # for i in range(batch_size):
    #     tmp = mx.contrib.symbol.MultiBoxTarget(*[refined_anchors_norm_bs[i], label_bs[i], cls_preds_bs[i]],
    #                                            overlap_threshold=0.5)
    #     tmp_bbox_target.append(tmp[0])
    #     tmp_bbox_weight.append(tmp[1])
    #     tmp_label.append(tmp[2])
    #
    # tmp_bbox_target = mx.symbol.concat(*tmp_bbox_target, num_args=len(tmp_bbox_target), dim=0)
    # tmp_bbox_weight = mx.symbol.concat(*tmp_bbox_weight, num_args=len(tmp_bbox_weight), dim=0)
    # tmp_label = mx.symbol.concat(*tmp_label, num_args=len(tmp_label), dim=0)
    #
    # tmp_bbox_target_reshape = mx.symbol.reshape(data=tmp_bbox_target,shape=(batch_size,A*bbox_pred_len,feat_height*feat_width),\
    #                                           name='%s_bbox_target_stride%d' % (prefix, stride))
    # tmp_bbox_weight_reshape = mx.symbol.reshape(data=tmp_bbox_weight,
    #                                           shape=(batch_size, A * bbox_pred_len, feat_height * feat_width), \
    #                                           name='%s_bbox_weight_stride%d' % (prefix, stride))
    # tmp_label_reshape = mx.symbol.reshape(data=tmp_label,
    #                                             shape=(batch_size, A * feat_height * feat_width), \
    #                                             name='%s_label_stride%d' % (prefix, stride))





    '''if landmark:
        rpn_landmark_pred = conv_only(rpn_relu, '%s_rpn_landmark_pred_stride%d' % (prefix, stride),
                                      landmark_pred_len * num_anchors,
                                      kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=shared_vars[2][0],
                                      shared_bias=shared_vars[2][1])
        rpn_landmark_pred_reshape = mx.symbol.Reshape(data=rpn_landmark_pred,
                                                      shape=(0, 0, -1),
                                                      name="%s_rpn_landmark_pred_reshape_stride%s" % (prefix, stride))'''

    # if config.TRAIN.RPN_ENABLE_OHEM >= 2:
    #     label, anchor_weight, valid_count = mx.sym.Custom(op_type='rpn_fpn_ohem3', stride=int(stride),
    #                                                       network=config.network, dataset=config.dataset, prefix=prefix,
    #                                                       cls_score=rpn_cls_score_reshape, labels=label)
    #
    #     _bbox_weight = mx.sym.tile(anchor_weight, (1, 1, bbox_pred_len))
    #     _bbox_weight = _bbox_weight.reshape((0, -1, A * bbox_pred_len)).transpose((0, 2, 1))
    #     bbox_weight = mx.sym.elemwise_mul(bbox_weight, _bbox_weight,
    #                                       name='%s_bbox_weight_mul_stride%s' % (prefix, stride))
    #
    #     if landmark:
    #         _landmark_weight = mx.sym.tile(anchor_weight, (1, 1, landmark_pred_len))
    #         _landmark_weight = _landmark_weight.reshape((0, -1, A * landmark_pred_len)).transpose((0, 2, 1))
    #         landmark_weight = mx.sym.elemwise_mul(landmark_weight, _landmark_weight,
    #                                               name='%s_landmark_weight_mul_stride%s' % (prefix, stride))
    #     # if not config.FACE_LANDMARK:
    #     #  label, bbox_weight = mx.sym.Custom(op_type='rpn_fpn_ohem', stride=int(stride), cls_score=rpn_cls_score_reshape, bbox_weight = bbox_weight , labels = label)
    #     # else:
    #     #  label, bbox_weight, landmark_weight = mx.sym.Custom(op_type='rpn_fpn_ohem2', stride=int(stride), cls_score=rpn_cls_score_reshape, bbox_weight = bbox_weight, landmark_weight=landmark_weight, labels = label)
    #
    # valid_count = mx.symbol.sum(valid_count)
    # valid_count = valid_count + 0.001  # avoid zero

    # bbox loss
    bbox_diff = rpn_bbox_pred_reshape - bbox_target
    bbox_diff = bbox_diff * bbox_weight
    rpn_bbox_loss_ = mx.symbol.smooth_l1(name='%s_rpn_bbox_loss_stride%d_' % (prefix, stride), scalar=3.0,
                                         data=bbox_diff)

    if config.LR_MODE == 0:
        rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d' % (prefix, stride), data=rpn_bbox_loss_,
                                        grad_scale=1.0 * lr_mult / (config.TRAIN.RPN_BATCH_SIZE))
    else:
        #rpn_bbox_loss_ = mx.symbol.broadcast_div(rpn_bbox_loss_, valid_count)
        rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d' % (prefix, stride), data=rpn_bbox_loss_,
                                        grad_scale=0.25 * lr_mult)
    ret_group.append(rpn_bbox_loss)
    ret_group.append(mx.sym.BlockGrad(bbox_weight))

    # landmark loss
    if landmark:
        landmark_diff = rpn_landmark_pred_reshape - landmark_target
        landmark_diff = landmark_diff * landmark_weight
        rpn_landmark_loss_ = mx.symbol.smooth_l1(name='%s_rpn_landmark_loss_stride%d_' % (prefix, stride), scalar=3.0,
                                                 data=landmark_diff)
        if config.LR_MODE == 0:
            rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d' % (prefix, stride),
                                                data=rpn_landmark_loss_,
                                                grad_scale=0.4 * config.LANDMARK_LR_MULT * lr_mult / (
                                                    config.TRAIN.RPN_BATCH_SIZE))
        else:
            #rpn_landmark_loss_ = mx.symbol.broadcast_div(rpn_landmark_loss_, valid_count)
            rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d' % (prefix, stride),
                                                data=rpn_landmark_loss_,
                                                grad_scale=0.1 * config.LANDMARK_LR_MULT * lr_mult)
        ret_group.append(rpn_landmark_loss)
        ret_group.append(mx.sym.BlockGrad(landmark_weight))
    return refined_anchors, ret_group


def get_out(conv_fpn_feat, prefix, stride, STC_para, STR_para, label_STR, landmark=False,  lr_mult=1.0, shared_vars = None):
    A = config.NUM_ANCHORS
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
      bbox_pred_len = 5
    if config.USE_OCCLUSION:
      landmark_pred_len = 15
    ret_group = []
    num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']

    ###### use STC#######
    if config.STC:
        if stride in config.STC_FPN_STRIDE:
            ret_group += STC_para[stride][3]
            label = STC_para[stride][0]
            bbox_weight = STC_para[stride][1]
            if landmark:
                landmark_weight = STC_para[stride][2]
            bbox_target = mx.symbol.Variable(name='%s_bbox_target_stride%d' % (prefix, stride))
            if landmark:
                landmark_target = mx.symbol.Variable(name='%s_landmark_target_stride%d' % (prefix, stride))
    elif stride in config.STC_FPN_STRIDE:
        label = mx.symbol.Variable(name='%s_label_stride%d' % (prefix, stride))
        bbox_weight = mx.symbol.Variable(name='%s_bbox_weight_stride%d' % (prefix, stride))
        if landmark:
            landmark_weight = mx.symbol.Variable(name='%s_landmark_weight_stride%d' % (prefix, stride))
        bbox_target = mx.symbol.Variable(name='%s_bbox_target_stride%d' % (prefix, stride))
        if landmark:
            landmark_target = mx.symbol.Variable(name='%s_landmark_target_stride%d' % (prefix, stride))

    ######## use STR ########
    if config.STR:
        if stride in config.STR_FPN_STRIDE:
            ret_group += STR_para[stride]
            label, bbox_target,bbox_weight,landmark_target,landmark_weight = mx.symbol.Custom(prefix=prefix,stride=stride,landmark=True,label_STR=label_STR,op_type='get_STR_label')
    elif stride in config.STR_FPN_STRIDE:
        label = mx.symbol.Variable(name='%s_label_stride%d' % (prefix, stride))
        bbox_weight = mx.symbol.Variable(name='%s_bbox_weight_stride%d' % (prefix, stride))
        bbox_target = mx.symbol.Variable(name='%s_bbox_target_stride%d' % (prefix, stride))
        # if landmark:
        #     landmark_weight = mx.symbol.Variable(name='%s_landmark_weight_stride%d' % (prefix, stride))
        if landmark:
            landmark_target = mx.symbol.Variable(name='%s_landmark_target_stride%d' % (prefix, stride))
            landmark_weight = mx.symbol.Variable(name='%s_landmark_weight_stride%d' % (prefix, stride))

    rpn_relu = conv_fpn_feat[stride]
    maxout_stat = 0
    if config.USE_MAXOUT>=1 and stride==config.RPN_FEAT_STRIDE[-1]:
      maxout_stat = 1
    if config.USE_MAXOUT>=2 and stride!=config.RPN_FEAT_STRIDE[-1]:
      maxout_stat = 2

    if maxout_stat==0:
      rpn_cls_score = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d'%(prefix, stride), 2*num_anchors,
          kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[0][0], shared_bias = shared_vars[0][1])
    elif maxout_stat==1:
      cls_list = []
      for a in range(num_anchors):
        rpn_cls_score_bg = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d_anchor%d_bg'%(prefix,stride,a), 3,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        rpn_cls_score_bg = mx.sym.max(rpn_cls_score_bg, axis=1, keepdims=True)
        cls_list.append(rpn_cls_score_bg)
        rpn_cls_score_fg = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d_anchor%d_fg'%(prefix,stride,a), 1,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        cls_list.append(rpn_cls_score_fg)
      rpn_cls_score = mx.sym.concat(*cls_list, dim=1, name='%s_rpn_cls_score_stride%d'%(prefix,stride))
    else:
      cls_list = []
      for a in range(num_anchors):
        rpn_cls_score_bg = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d_anchor%d_bg'%(prefix,stride,a), 1,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        cls_list.append(rpn_cls_score_bg)
        rpn_cls_score_fg = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d_anchor%d_fg'%(prefix,stride,a), 3,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        rpn_cls_score_fg = mx.sym.max(rpn_cls_score_fg, axis=1, keepdims=True)
        cls_list.append(rpn_cls_score_fg)
      rpn_cls_score = mx.sym.concat(*cls_list, dim=1, name='%s_rpn_cls_score_stride%d'%(prefix,stride))

    rpn_bbox_pred = conv_only(rpn_relu, '%s_rpn_bbox_pred_stride%d'%(prefix,stride), bbox_pred_len*num_anchors,
        kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[1][0], shared_bias = shared_vars[1][1])

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                              shape=(0, 2, -1),
                                              name="%s_rpn_cls_score_reshape_stride%s" % (prefix,stride))

    rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                              shape=(0, 0, -1),
                                              name="%s_rpn_bbox_pred_reshape_stride%s" % (prefix,stride))
    if landmark:
      rpn_landmark_pred = conv_only(rpn_relu, '%s_rpn_landmark_pred_stride%d'%(prefix,stride), landmark_pred_len*num_anchors,
          kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[2][0], shared_bias = shared_vars[2][1])
      rpn_landmark_pred_reshape = mx.symbol.Reshape(data=rpn_landmark_pred,
                                              shape=(0, 0, -1),
                                              name="%s_rpn_landmark_pred_reshape_stride%s" % (prefix,stride))

    if config.USE_KLLOSS:
        alpha_pred = conv_only(rpn_relu, '%s_alpha_pred_stride%d'%(prefix,stride), bbox_pred_len*num_anchors,
                               kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[1][0], shared_bias = shared_vars[1][1])
        alpha_pred_reshape = mx.symbol.Reshape(data=alpha_pred, shape=(0,0,-1), name="%s_alpha_pred_reshape_stride%s"%(prefix, stride))

    if config.TRAIN.RPN_ENABLE_OHEM>=2:
      label, anchor_weight, valid_count = mx.sym.Custom(op_type='rpn_fpn_ohem3', stride=int(stride), network=config.network, dataset=config.dataset, prefix=prefix, cls_score=rpn_cls_score_reshape, labels = label)

      _bbox_weight = mx.sym.tile(anchor_weight, (1,1,bbox_pred_len))
      _bbox_weight = _bbox_weight.reshape((0, -1, A * bbox_pred_len)).transpose((0,2,1))
      bbox_weight = mx.sym.elemwise_mul(bbox_weight, _bbox_weight, name='%s_bbox_weight_mul_stride%s'%(prefix,stride))

      if landmark:
        _landmark_weight = mx.sym.tile(anchor_weight, (1,1,landmark_pred_len))
        _landmark_weight = _landmark_weight.reshape((0, -1, A * landmark_pred_len)).transpose((0,2,1))
        landmark_weight = mx.sym.elemwise_mul(landmark_weight, _landmark_weight, name='%s_landmark_weight_mul_stride%s'%(prefix,stride))
      #if not config.FACE_LANDMARK:
      #  label, bbox_weight = mx.sym.Custom(op_type='rpn_fpn_ohem', stride=int(stride), cls_score=rpn_cls_score_reshape, bbox_weight = bbox_weight , labels = label)
      #else:
      #  label, bbox_weight, landmark_weight = mx.sym.Custom(op_type='rpn_fpn_ohem2', stride=int(stride), cls_score=rpn_cls_score_reshape, bbox_weight = bbox_weight, landmark_weight=landmark_weight, labels = label)
    #cls loss
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape,
                                           label=label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           grad_scale = lr_mult,
                                           name='%s_rpn_cls_prob_stride%d'%(prefix,stride))
    # _, out_shape1, _ = rpn_cls_score.infer_shape(data=(1, 3, 640, 640))
    # _, out_shape, _ = rpn_cls_score_reshape.infer_shape(data=(1, 3, 640, 640))
    #rpn_cls_prob = mx.symbol.Custom(data=rpn_cls_score_reshape, op_type='FocalLoss', labels = label, name='%s_rpn_cls_prob_stride%d'%(prefix,stride), alpha=0.25, gamma=2)
    #print(rpn_cls_prob.list_arguments())
    ret_group.append(rpn_cls_prob)
    ret_group.append(mx.sym.BlockGrad(label))

    valid_count = mx.symbol.sum(valid_count)
    valid_count = valid_count + 0.001 #avoid zero

    #bbox loss
    bbox_diff = rpn_bbox_pred_reshape-bbox_target
    bbox_diff = bbox_diff * bbox_weight
    rpn_bbox_loss_ = mx.symbol.smooth_l1(name='%s_rpn_bbox_loss_stride%d_'%(prefix,stride), scalar=3.0, data=bbox_diff)
    if config.LR_MODE==0:
      rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d'%(prefix,stride), data=rpn_bbox_loss_, grad_scale=1.0*lr_mult / (config.TRAIN.RPN_BATCH_SIZE))
    else:
      rpn_bbox_loss_ = mx.symbol.broadcast_div(rpn_bbox_loss_, valid_count)
      rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d'%(prefix,stride), data=rpn_bbox_loss_, grad_scale=0.25*lr_mult)
    ret_group.append(rpn_bbox_loss)
    ret_group.append(mx.sym.BlockGrad(bbox_weight))

    #bbox KL loss
    if config.USE_KLLOSS:
        bbox_diff_abs = mx.symbol.abs(data=bbox_diff, name='abs_bbox_diff')
        bbox_diff_sq = mx.symbol.pow(bbox_diff, 2)
        eq10_label = mx.symbol.broadcast_greater(bbox_diff_abs, 1)
        eq9_label = mx.symbol.broadcast_lesser_equal(bbox_diff_abs, 1)

        eq9_result = mx.symbol.elemwise_mul(bbox_diff_sq, eq9_label)
        eq9_result = mx.symbol.elemwise_mul(eq9_result,0.5)
        eq10_result = mx.symbol.elemwise_sub(bbox_diff_abs, 0.5)
        eq10_result = mx.symbol.elemwise_mul(eq10_result, eq10_label)
        result_1 = mx.symbol.elemwise_add(eq9_result, eq10_result)

        alpha_pred_reshape_log = mx.symbol.elemwise_mul(alpha_pred_reshape, 0.5)
        alpha_pred_reshape_negative = mx.symbol.negative(alpha_pred_reshape)
        alpha_pred_reshape_exp = mx.symbol.exp(alpha_pred_reshape_negative)

        result_1 = mx.symbol.elemwise_mul(result_1, alpha_pred_reshape_exp)

        alpha_pred_reshape_log_mean = mx.symbol.mean(alpha_pred_reshape_log, 0)
        alpha_pred_reshape_log_meanr = mx.symbol.elemwise_sub(alpha_pred_reshape_log, alpha_pred_reshape_log_mean)

        log_loss = mx.symbol.sum(alpha_pred_reshape_log_meanr)

        #not scale result1 line165 in fast_rcnn_heads
        result_1_mean = mx.symbol.mean(result_1, 0)
        result_1_meanr = mx.symbol.elemwise_sub(result_1, result_1_mean)
        mul_loss = mx.symbol.sum(result_1_meanr)

        KL_loss = log_loss + mul_loss
        KL_loss = mx.symbol.make_loss(KL_loss, name='%s_KL_loss_stride%d'%(prefix,stride), grad_scale=0.25*lr_mult)
        ret_group.append(KL_loss)




    #landmark loss
    if landmark:
      landmark_diff = rpn_landmark_pred_reshape-landmark_target
      landmark_diff = landmark_diff * landmark_weight
      rpn_landmark_loss_ = mx.symbol.smooth_l1(name='%s_rpn_landmark_loss_stride%d_'%(prefix,stride), scalar=3.0, data=landmark_diff)
      if config.LR_MODE==0:
        rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d'%(prefix,stride), data=rpn_landmark_loss_, grad_scale=0.4*config.LANDMARK_LR_MULT*lr_mult / (config.TRAIN.RPN_BATCH_SIZE))
      else:
        rpn_landmark_loss_ = mx.symbol.broadcast_div(rpn_landmark_loss_, valid_count)
        rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d'%(prefix,stride), data=rpn_landmark_loss_, grad_scale=0.1*config.LANDMARK_LR_MULT*lr_mult)
      ret_group.append(rpn_landmark_loss)
      ret_group.append(mx.sym.BlockGrad(landmark_weight))
    if config.USE_3D:
      from rcnn.PY_OP import rpn_3d_mesh
      pass
    return ret_group

def get_sym_train(sym, stride, flag):
    data = mx.symbol.Variable(name="data")

    # shared convolutional layers
    conv_fpn_feat, STC_convs, STR_convs = get_sym_conv(data, sym)
    if config.STC:
        assert len(STC_convs) == len(config.STC_FPN_STRIDE)
    if config.STR:
        assert len(STR_convs) == len(config.STR_FPN_STRIDE)

    ret_group = []
    shared_vars = []
    if config.SHARE_WEIGHT_BBOX:
      assert config.USE_MAXOUT==0
      _name = 'face_rpn_cls_score_share'
      shared_weight = mx.symbol.Variable(name="{}_weight".format(_name),
          init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
      shared_bias = mx.symbol.Variable(name="{}_bias".format(_name),
          init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(0.0)})
      shared_vars.append( [shared_weight, shared_bias] )
      _name = 'face_rpn_bbox_pred_share'
      shared_weight = mx.symbol.Variable(name="{}_weight".format(_name),
          init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
      shared_bias = mx.symbol.Variable(name="{}_bias".format(_name),
          init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(0.0)})
      shared_vars.append( [shared_weight, shared_bias] )
    else:
      shared_vars.append( [None, None] )
      shared_vars.append( [None, None] )
    if config.SHARE_WEIGHT_LANDMARK:
      _name = 'face_rpn_landmark_pred_share'
      shared_weight = mx.symbol.Variable(name="{}_weight".format(_name),
          init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
      shared_bias = mx.symbol.Variable(name="{}_bias".format(_name),
          init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(0.0)})
      shared_vars.append( [shared_weight, shared_bias] )
    else:
      shared_vars.append( [None, None] )

    STC_para = {}
    if config.STC:
        for stride in config.STC_FPN_STRIDE:
            label_stc, bbweight_stc, ldmweight_stc, ret_stc = STC(STC_convs, 'face', stride, config.FACE_LANDMARK, lr_mult=1.0,shared_vars = shared_vars)
            STC_para[stride] = [label_stc, bbweight_stc, ldmweight_stc, ret_stc]

    STR_para = {}
    if config.STR:
        refined_anchor_list = []
        for stride in config.RPN_FEAT_STRIDE:
            if stride in config.STR_FPN_STRIDE:
                refined_anchor, ret_str = STR(STR_convs, 'face', stride, config.FACE_LANDMARK, lr_mult=1.0,
                                              shared_vars=shared_vars)
                STR_para[stride] = ret_str
                refined_anchor_list.append(refined_anchor)
            else:
                refined_anchor_list.append(construct_anchor(conv_fpn_feat[stride], stride, 0, 640))
        refined_anchor = mx.sym.concat(*[a for a in refined_anchor_list],dim=1)
        #refined_anchor = mx.sym.concat(refined_anchor_list,ori_anchors, dim=1)
        gt_boxes = mx.symbol.Variable(name='gt_boxes')
        gt_landmarks = mx.symbol.Variable(name='gt_landmarks')
        gt_boxes_reshape, gt_landmarks_reshape = mx.symbol.Custom(gt_boxes=gt_boxes,gt_landmarks=gt_landmarks, op_type='gt_boxes_reshape')
        #STR_op(refined_anchor, gt_boxes_reshape, gt_landmarks_reshape)
        label_STR = mx.symbol.Custom(anchors = refined_anchor ,gt_boxes=gt_boxes_reshape, gt_landmarks=gt_landmarks_reshape,op_type='STR_op')

        #STR_para[stride] = [refined_anchor, refined_landmarks, ret_str]

    for stride in config.RPN_FEAT_STRIDE:
      ret = get_out(conv_fpn_feat, 'face', stride, STC_para,STR_para, label_STR, config.FACE_LANDMARK, lr_mult=1.0, shared_vars = shared_vars)
      ret_group += ret
      if config.HEAD_BOX:
        assert not config.SHARE_WEIGHT_BBOX and not config.SHARE_WEIGHT_LANDMARK
        shared_vars = [ [None, None], [None, None], [None, None] ]
        ret = get_out(conv_fpn_feat, 'head', stride, False, lr_mult=0.5, shared_vars = shared_vars)
        ret_group += ret

    return mx.sym.Group(ret_group)

def construct_anchor(data, stride, offset, isize):
    offset = offset
    sstride = str(stride)
    base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
    allowed_border = config.RPN_ANCHOR_CFG[sstride]['ALLOWED_BORDER']
    ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
    scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
    feat_height = isize / stride
    feat_width = isize / stride
    # a = mx.nd.random.uniform(shape=(1, 3, feat_height, feat_width))
    scalesmy = []
    for i in range(len(scales)):
        scalesmy.append(scales[i] * 16 / 640)
    ##reproduce initial anchors###
    anchors_nd = mx.contrib.symbol.MultiBoxPrior(data=data, sizes=scalesmy,
                                                 steps=(float(1 / feat_height), float(1 / feat_width)), \
                                                 offsets=(offset, offset))
    anchors_nd.__add__(0.01171875)  # 7.5/640
    anchors_no_norm = anchors_nd
    anchors_no_norm = anchors_no_norm.__mul__(isize)

    return anchors_no_norm

def STR_op(anchors ,gt_boxes, gt_landmarks):
    landmark = config.FACE_LANDMARK
    bbox_pred_len = 4
    landmark_pred_len = 10
    feat_strides = config.RPN_FEAT_STRIDE
    prefix = 'face'
    overlaps = bbox_overlaps_symbol(anchors,gt_boxes)



def bbox_overlaps_symbol(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = 102400
    k_ = 5
    overlaps = np.zeros((n_, k_), dtype=np.float)
    a = query_boxes[1,2]
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

