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
from rcnn.symbol.focal_loss_optimizedversion import *
from rcnn.PY_OP.cascade_refine import *
from rcnn.PY_OP.KL_Loss import *
from rcnn.symbol.nonlocal_net import *
from ce_det_func.wh_process import *
from rcnn.PY_OP.hm_loss import *
from rcnn.PY_OP.af_mining import *

def GroupNorm(data, in_channel, name, num_groups=32, eps=1e-5):
    """
    If the batch size is small, it's better to use GroupNorm instead of BatchNorm.
    GroupNorm achieves good results even at small batch sizes.
    Reference:
      https://arxiv.org/pdf/1803.08494.pdf
    """
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1] # G: number of groups for GN
    C = in_channel
    G = num_groups
    G = min(G, C)
    x_group= mx.sym.reshape(data = data, shape = (1, G, C//G, 0, -1))
    mean = mx.sym.mean(x_group, axis= (2, 3, 4), keepdims = True)
    differ = mx.sym.broadcast_minus(lhs = x_group, rhs = mean)
    var = mx.sym.mean(mx.sym.square(differ), axis = (2, 3, 4), keepdims =True)
    x_groupnorm = mx.sym.broadcast_div(lhs = differ, rhs = mx.sym.sqrt(var + eps))
    #x_out = mx.sym.reshape(x_groupnorm, shape = (0, -3, -2))
    x_out = mx.sym.reshape_like(x_groupnorm, data)
    gamma = mx.sym.Variable(name = name + '_gamma',shape = (1,C,1,1), dtype='float32')
    beta = mx.sym.Variable(name = name + '_beta', shape=(1,C,1,1), dtype='float32')
    gn_x = mx.sym.broadcast_mul(lhs = x_out, rhs = gamma)
    gn_x = mx.sym.broadcast_plus(lhs = gn_x, rhs = beta)
    return gn_x

def SE_Block(from_layer, name, num_filter, ratio, kernel_pool,pool_type = 'avg'):
    squeeze = mx.sym.Pooling(data=from_layer, global_pool=True, kernel=(kernel_pool, kernel_pool), pool_type=pool_type, name=name + '_squeeze')
    squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
    excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter * ratio), name=name + '_excitation1')
    excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
    excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
    excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
    conv3 = mx.symbol.broadcast_mul(from_layer, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))
    return conv3

def Act(data, act_type='prelu', name=None):
    body = mx.sym.LeakyReLU(data = data, act_type=act_type, name = '%s_%s' %(name, act_type))
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def CBAM(data, name, num_filter, ratio, act_type,stride):
    # Channel attention module
    module_input = data
    avg = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name='%s_ca_avg_pool1_%s' % (name,stride))
    ma = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='max', name='%s_ca_max_pool1_%s' % (name,stride))

    avg = conv_only(avg,'%s_ca_avg_fc1_%s'%(name,stride), int(num_filter*ratio),kernel=(1,1),pad=(0,0),stride=(1,1))
    ma = conv_only(ma,'%s_ca_max_fc1_%s'%(name,stride), int(num_filter*ratio),kernel=(1,1),pad=(0,0),stride=(1,1))

    avg = Act(avg, act_type=act_type, name='%s_ca_avg_%s_%s' % (name, act_type,stride))
    ma = Act(ma, act_type=act_type, name='%s_ca_max_%s_%s' % (name, act_type,stride))
    avg = conv_only(avg,'%s_ca_avg_fc2_%s'%(name,stride), num_filter,kernel=(1,1),pad=(0,0),stride=(1,1))
    ma = conv_only(ma, '%s_ca_max_fc2_%s' % (name,stride), num_filter, kernel=(1, 1), pad=(0, 0), stride=(1, 1))

    body = avg + ma
    body = mx.symbol.Activation(data=body, act_type='sigmoid', name='%s_ca_sigmoid_%s' % (name,stride))
    # Spatial attention module
    body = mx.symbol.broadcast_mul(module_input, body)

    module_input = body
    avg = mx.symbol.mean(data=body, axis=1, keepdims=True, name='%s_sa_mean_%s' % (name,stride))
    ma = mx.symbol.max(data=body, axis=1, keepdims=True, name='%s_sa_max_%s' % (name,stride))

    body = mx.symbol.Concat(avg, ma, dim=1, name='%s_sa_concat_%s' % (name,stride))
    body = conv_only(body,name = '%s_sa_conv_%s' % (name,stride), num_filter=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    body = mx.symbol.Activation(data=body, act_type='sigmoid', name='%s_sa_sigmoid_%s' % (name,stride))
    body = mx.symbol.broadcast_mul(module_input, body)
    return body

def cpm_residual(body, num_filter, filter_in, name):
    # branch1 = conv_bn(input, 1024, 1, 1, 0, None)
    # branch2a = conv_bn(input, 256, 1, 1, 0, act='relu')
    # branch2b = conv_bn(branch2a, 256, 3, 1, 1, act='relu')
    # branch2c = conv_bn(branch2b, 1024, 1, 1, 0, None)
    # sum = branch1 + branch2c
    # rescomb = fluid.layers.relu(x=sum)
    assert num_filter % 4 == 0
    branch1 = conv_act_layer(body, name + '_branch1',
                             num_filter, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='', separable=False,
                             filter_in=filter_in)
    # _filter = max(num_filter//4, 16)
    _filter = num_filter // 4
    branch2a = conv_act_layer(body, name + '_branch2a',
                             _filter, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', separable=False,
                             filter_in=filter_in)
    branch2b = conv_act_layer(branch2a, name + '_branch2b',
                             _filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False,
                             filter_in=filter_in)
    branch2c = conv_act_layer(branch2b, name + '_branch2c',
                             num_filter, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='', separable=False,
                             filter_in=filter_in)
    sum = branch1 + branch2c
    ret = mx.symbol.Activation(sum,act_type='relu',name=name + '_concat_relu')
    # conv5x5, conv7x7 = ssh_context_module(body, _filter, filter_in, name + '_context')
    # ret = mx.sym.concat(*[conv3x3, conv5x5, conv7x7], dim=1, name=name + '_concat')
    # ret = mx.symbol.Activation(data=ret, act_type='relu', name=name + '_concat_relu')
    # out_filter = num_filter // 2 + _filter * 2
    # if config.USE_DCN > 0:
    #     ret = conv_deformable(ret, num_filter=num_filter, name=name + '_concat_dcn')
    return ret

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
  if config.use_gn:
      net = GroupNorm(data = net, in_channel= num_filter,name=name + '_gn')
  else:
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
      if config.use_gn:
          conv = GroupNorm(data=conv, in_channel=num_filter, name=name + '_gn')
      else:
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

    isize = config.SCALES[0][0]
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

    m1 = c1
    m2 = c2
    m3 = c3
    if config.head_module_after:
        if config.cpm:
            c1_cpm = cpm_residual(c1, config.cpm, config.cpm, 'cpm_residual_c1')
            c2_cpm = cpm_residual(c2, config.cpm, config.cpm, 'cpm_residual_c2')
            c3_cpm = cpm_residual(c3, config.cpm, config.cpm, 'cpm_residual_c3')
            m1 = head_module(c1_cpm, F2 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c1_det')
            m2 = head_module(c2_cpm, F1 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c2_det')
            m3 = head_module(c3_cpm, F1 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c3_det')
        else:
            m1 = head_module(c1, F2 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c1_det')
            m2 = head_module(c2, F1 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c2_det')
            m3 = head_module(c3, F1 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c3_det')
    if len(config.RPN_ANCHOR_CFG)==3:
      ret = {8: m1, 16:m2, 32: m3}
    elif len(config.RPN_ANCHOR_CFG)==1:
      ret = {16:m2}
    elif len(config.RPN_ANCHOR_CFG)==2:
      ret = {8: m1, 16:m2, 32: m3}
    elif len(config.RPN_ANCHOR_CFG)==4:
      c0_lateral = conv_act_layer(c0, 'rf_c0_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_up = upsampling(c1, F2, 'rf_c1_upsampling')
      if config.USE_CROP:
        c1_up = mx.symbol.Crop(*[c1_up, c0_lateral])
      c0 = c0_lateral+c1_up
      c0 = conv_act_layer(c0, 'rf_c0_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      m0 = c0
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
      m0 = c0
      m4 = c4

      if config.head_module_after:
          if config.cpm:
              c0_cpm = cpm_residual(c0, config.cpm, config.cpm, 'cpm_residual_c0')
              c4_cpm = cpm_residual(c4, config.cpm, config.cpm, 'cpm_residual_c4')
          m0 = head_module(c0_cpm, F2 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c0_det')
          m4 = head_module(c4_cpm, F1 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c4_det')
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
      m0 = c0
      m4 = c4
      m5 = c5
      ret = {4: m0, 8: m1, 16:m2, 32: m3, 64: m4, 128: m5}

    if config.bfp:
        idx = 0
        for key in ret.keys():# choose gather key
            if idx == len(ret)-2:
                gather_key = key
                break
            idx+=1
        gather_size = isize / gather_key
        feats = []
        for key in ret.keys():
            key_size = isize / key
            if key < gather_key:
                filter_size = key_size / gather_size
                filter_stride = filter_size
                gathered = mx.symbol.Pooling(data=ret[key],kernel=(3,3),pool_type='max',stride=(filter_stride,filter_stride),
                                            pad=(1,1),name='gathered%s'%key)
            elif key > gather_key:
                scale = key / gather_key
                gathered = mx.symbol.UpSampling(ret[key], scale=scale, sample_type='bilinear', workspace=512, name='gathered%s'%key, num_filter=F1)
            else:
                gathered = ret[key]
            if config.USE_CROP:
                gathered = mx.symbol.Crop(*[gathered, ret[gather_key]])
            feats.append(gathered)
        # glb = mx.sym.Pooling(data=ret[ret.keys()[len(ret)-1]], global_pool=True, kernel=(1,1), pool_type='avg',
        #                          name='global_pooling')
        # feats.append(mx.symbol.broadcast_like(glb,gathered))
        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        bsf = non_local_block(bsf,num_filter=F1)
        # step 3: scatter refined features to multi-levels by a residual path
        for key in ret.keys():
            if key < gather_key:
                scale = gather_key / key
                temp = mx.symbol.UpSampling(bsf, scale=scale, sample_type='bilinear', workspace=512, name='bsf%s'%key, num_filter=F1)
                if config.USE_CROP:
                    temp = mx.symbol.Crop(*[temp,ret[key]])
                ret[key] = temp + ret[key]
            else:
                ret[key] = mx.symbol.Pooling(data=bsf,kernel=(3,3),pool_type='max',stride=(key/gather_key,key/gather_key),
                                            pad=(1,1),name='bsf%s'%key) + ret[key]
    if config.centernet_branch:
        for stride in config.ct_stride:
            if stride not in ret.keys():
                if stride == 4:
                    c0_lateral = conv_act_layer(c0, 'rf_c0_lateral',
                                                F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu',
                                                bias_wd_mult=_bwm)
                    c1_up = upsampling(c1, F2, 'rf_c1_upsampling')
                    if config.USE_CROP:
                        c1_up = mx.symbol.Crop(*[c1_up, c0_lateral])
                    c0 = c0_lateral + c1_up
                    c0 = conv_act_layer(c0, 'rf_c0_aggr',
                                        F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

                    m0 = c0
                    if config.head_module_after:
                        m0 = head_module(c0, F2 * config.CONTEXT_FILTER_RATIO, F2, 'rf_c0_det')
                    ret = {4: m0, 8: m1, 16: m2, 32: m3}
    #return {8: m1, 16:m2, 32: m3}
    return ret

def get_out(conv_fpn_feat, prefix, stride, min_stride,landmark=False, lr_mult=1.0, shared_vars = None, gt_boxes = None, gt_landmarks = None,mined_labels = None, mined_bbox_weight = None):
    A = config.NUM_ANCHORS
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
      bbox_pred_len = 5
    if config.USE_OCCLUSION:
      landmark_pred_len = 15
    ret_group = []
    deephead = config.deephead
    if prefix != 'face':
        deephead = False
    num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
    cls_label = mx.symbol.Variable(name='%s_label_stride%d'%(prefix,stride))
    bbox_target = mx.symbol.Variable(name='%s_bbox_target_stride%d'%(prefix,stride))
    bbox_weight = mx.symbol.Variable(name='%s_bbox_weight_stride%d'%(prefix,stride))
    if config.centernet_branch and config.ct_mining and stride in config.ct_mining_stride and prefix == 'face':
        cls_label = mined_labels
        bbox_weight = mined_bbox_weight
    if landmark:
      landmark_target = mx.symbol.Variable(name='%s_landmark_target_stride%d'%(prefix,stride))
      landmark_weight = mx.symbol.Variable(name='%s_landmark_weight_stride%d'%(prefix,stride))

    if config.USE_OHEGS:
        rpn_relu = conv_fpn_feat[min_stride]
    else:
        rpn_relu = conv_fpn_feat[stride]

    F1 = config.HEAD_FILTER_NUM
    F2 = F1
    if config.non_local:
        print('use non_local')
        rpn_relu = non_local_block(rpn_relu,F2,ith=stride)

    if not config.head_module_after:
        if stride not in config.STR_FPN_STRIDE:
            rpn_relu = head_module(rpn_relu, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_head_stride%d'%(stride))
        elif not config.STR:
            rpn_relu = head_module(rpn_relu, F1 * config.CONTEXT_FILTER_RATIO, F2, 'rf_head_stride%d' % (stride))

    if config.CBAM:
        print("use CBAM")
        rpn_relu = CBAM(rpn_relu,'rpn_relu',F1,0.0625,'prelu',stride)

    if config.USE_SEBLOCK:

        #c1 = SE_Block(c1, 'c1', F2, 0.0625, 80)
        #c2 = SE_Block(c2, 'c2', F2, 0.0625, 40)
        if stride == 32:
            rpn_relu = SE_Block(rpn_relu, 'rf_head_stride%d_se' % stride, F2, 0.0625, 20)
            print('use SEBlock in 32')

    ####################deep head####################
    if deephead:
        cls_conv1 = mx.sym.Convolution(
            data=rpn_relu, kernel=(3, 3), pad=(1, 1), weight=shared_vars[0][0], bias=shared_vars[0][1],
            num_filter=F1, name="cls_conv1_3x3/s" + str(stride))
        # cls_conv1 = conv_only(from_layer=rpn_relu, kernel=(3, 3), pad=(1, 1), weight=shared_vars[0][0], bias=shared_vars[0][1],
        #     num_filter=F1, name="cls_conv1_3x3/s" + str(stride))
        cls_conv1 = mx.symbol.Activation(data=cls_conv1, act_type='relu')
        #    cls_conv1 = mx.sym.Custom(op_type='Check',  data=cls_conv1)
        cls_conv2 = mx.sym.Convolution(
            data=cls_conv1, kernel=(3, 3), pad=(1, 1), weight=shared_vars[1][0], bias=shared_vars[1][1],
            num_filter=F1, name="cls_conv2_3x3/s" + str(stride))
        cls_conv2 = mx.symbol.Activation(data=cls_conv2, act_type='relu')
        cls_conv3 = mx.sym.Convolution(
            data=cls_conv2, kernel=(3, 3), pad=(1, 1), weight=shared_vars[2][0], bias=shared_vars[2][1],
            num_filter=F1, name="cls_conv3_3x3/s" + str(stride))
        cls_conv3 = mx.symbol.Activation(data=cls_conv3, act_type='relu')
        cls_conv4 = mx.sym.Convolution(
            data=cls_conv3, kernel=(3, 3), pad=(1, 1), weight=shared_vars[3][0], bias=shared_vars[3][1],
            num_filter=F1, name="cls_conv4_3x3/s" + str(stride))
        cls_conv4 = mx.symbol.Activation(data=cls_conv4, act_type='relu')

        box_conv1 = mx.sym.Convolution(
            data=rpn_relu, kernel=(3, 3), pad=(1, 1), weight=shared_vars[5][0], bias=shared_vars[5][1],
            num_filter=F1, name="box_conv1_3x3/s" + str(stride))
        box_conv1 = mx.symbol.Activation(data=box_conv1, act_type='relu')
        box_conv2 = mx.sym.Convolution(
            data=box_conv1, kernel=(3, 3), pad=(1, 1), weight=shared_vars[6][0], bias=shared_vars[6][1], num_filter=F1,
            name="box_conv2_3x3/s" + str(stride))
        box_conv2 = mx.symbol.Activation(data=box_conv2, act_type='relu')
        box_conv3 = mx.sym.Convolution(
            data=box_conv2, kernel=(3, 3), pad=(1, 1), weight=shared_vars[7][0], bias=shared_vars[7][1], num_filter=F1,
            name="box_conv3_3x3/s" + str(stride))
        box_conv3 = mx.symbol.Activation(data=box_conv3, act_type='relu')
        box_conv4 = mx.sym.Convolution(
            data=box_conv3, kernel=(3, 3), pad=(1, 1), weight=shared_vars[8][0], bias=shared_vars[8][1], num_filter=F1,
            name="box_conv4_3x3/s" + str(stride))
        box_conv4 = mx.symbol.Activation(data=box_conv4, act_type='relu')
        if landmark:
            ldk_conv1 = mx.sym.Convolution(
                data=rpn_relu, kernel=(3, 3), pad=(1, 1), weight=shared_vars[10][0], bias=shared_vars[10][1],
                num_filter=F1, name="ldk_conv1_3x3/s" + str(stride))
            ldk_conv1 = mx.symbol.Activation(data=ldk_conv1, act_type='relu')
            ldk_conv2 = mx.sym.Convolution(
                data=ldk_conv1, kernel=(3, 3), pad=(1, 1), weight=shared_vars[11][0], bias=shared_vars[11][1],
                num_filter=F1,
                name="ldk_conv2_3x3/s" + str(stride))
            ldk_conv2 = mx.symbol.Activation(data=ldk_conv2, act_type='relu')
            ldk_conv3 = mx.sym.Convolution(
                data=ldk_conv2, kernel=(3, 3), pad=(1, 1), weight=shared_vars[12][0], bias=shared_vars[12][1],
                num_filter=F1,
                name="ldk_conv3_3x3/s" + str(stride))
            ldk_conv3 = mx.symbol.Activation(data=ldk_conv3, act_type='relu')
            ldk_conv4 = mx.sym.Convolution(
                data=ldk_conv3, kernel=(3, 3), pad=(1, 1), weight=shared_vars[13][0], bias=shared_vars[13][1],
                num_filter=F1,
                name="ldk_conv4_3x3/s" + str(stride))
            ldk_conv4 = mx.symbol.Activation(data=ldk_conv4, act_type='relu')
    maxout_stat = 0
    if config.USE_MAXOUT>=1 and stride==config.RPN_FEAT_STRIDE[-1]:
      maxout_stat = 1
    if config.USE_MAXOUT>=2 and stride!=config.RPN_FEAT_STRIDE[-1]:
      maxout_stat = 2

    if maxout_stat==0:
      rpn_cls_score = conv_only(rpn_relu if not deephead else cls_conv4, '%s_rpn_cls_score_stride%d'%(prefix, stride), 2*num_anchors,
          kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[4][0], shared_bias = shared_vars[4][1])
    elif maxout_stat==1:
      cls_list = []
      for a in range(num_anchors):
        rpn_cls_score_bg = conv_only(rpn_relu if not deephead else cls_conv4, '%s_rpn_cls_score_stride%d_anchor%d_bg'%(prefix,stride,a), 3,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        rpn_cls_score_bg = mx.sym.max(rpn_cls_score_bg, axis=1, keepdims=True)
        cls_list.append(rpn_cls_score_bg)
        rpn_cls_score_fg = conv_only(rpn_relu if not deephead else cls_conv4, '%s_rpn_cls_score_stride%d_anchor%d_fg'%(prefix,stride,a), 1,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        cls_list.append(rpn_cls_score_fg)
      rpn_cls_score = mx.sym.concat(*cls_list, dim=1, name='%s_rpn_cls_score_stride%d'%(prefix,stride))
    else:
      cls_list = []
      for a in range(num_anchors):
        rpn_cls_score_bg = conv_only(rpn_relu if not deephead else cls_conv4, '%s_rpn_cls_score_stride%d_anchor%d_bg'%(prefix,stride,a), 1,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        cls_list.append(rpn_cls_score_bg)
        rpn_cls_score_fg = conv_only(rpn_relu if not deephead else cls_conv4, '%s_rpn_cls_score_stride%d_anchor%d_fg'%(prefix,stride,a), 3,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        rpn_cls_score_fg = mx.sym.max(rpn_cls_score_fg, axis=1, keepdims=True)
        cls_list.append(rpn_cls_score_fg)
      rpn_cls_score = mx.sym.concat(*cls_list, dim=1, name='%s_rpn_cls_score_stride%d'%(prefix,stride))

    rpn_bbox_pred = conv_only(rpn_relu if not deephead else box_conv4, '%s_rpn_bbox_pred_stride%d'%(prefix,stride), bbox_pred_len*num_anchors,
        kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[9][0], shared_bias = shared_vars[9][1])

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                              shape=(0, 2, -1),
                                              name="%s_rpn_cls_score_reshape_stride%s" % (prefix,stride))

    rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                              shape=(0, 0, -1),
                                              name="%s_rpn_bbox_pred_reshape_stride%s" % (prefix,stride))
    if landmark:
      rpn_landmark_pred = conv_only(rpn_relu if not deephead else ldk_conv4, '%s_rpn_landmark_pred_stride%d'%(prefix,stride), landmark_pred_len*num_anchors,
          kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[14][0], shared_bias = shared_vars[14][1])
      rpn_landmark_pred_reshape = mx.symbol.Reshape(data=rpn_landmark_pred,
                                              shape=(0, 0, -1),
                                              name="%s_rpn_landmark_pred_reshape_stride%s" % (prefix,stride))

    if config.USE_KLLOSS:
        alpha_pred = conv_only(rpn_relu, '%s_alpha_pred_stride%d'%(prefix,stride), bbox_pred_len*num_anchors,
                               kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = shared_vars[1][0], shared_bias = shared_vars[1][1])
        alpha_pred_reshape = mx.symbol.Reshape(data=alpha_pred, shape=(0,0,-1), name="%s_alpha_pred_reshape_stride%s"%(prefix, stride))

    if config.landmark_klloss:
        beta_pred = conv_only(rpn_relu, '%s_beta_pred_stride%d' % (prefix, stride), landmark_pred_len * num_anchors,
                               kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=shared_vars[2][0],
                               shared_bias=shared_vars[2][1])
        beta_pred_reshape = mx.symbol.Reshape(data=beta_pred, shape=(0, 0, -1),
                                               name="%s_beta_pred_reshape_stride%s" % (prefix, stride))
    if config.TRAIN.RPN_ENABLE_OHEM>=2:
      label, anchor_weight, valid_count = mx.sym.Custom(op_type='rpn_fpn_ohem3', stride=int(stride), network=config.network, dataset=config.dataset, prefix=prefix, cls_score=rpn_cls_score_reshape, labels = cls_label)

      _bbox_weight = mx.sym.tile(anchor_weight, (1,1,bbox_pred_len))
      _bbox_weight = _bbox_weight.reshape((0, -1, A * bbox_pred_len)).transpose((0,2,1))
      bbox_weight = mx.sym.elemwise_mul(bbox_weight, _bbox_weight, name='%s_bbox_weight_mul_stride%s'%(prefix,stride))

      if landmark:
        _landmark_weight = mx.sym.tile(anchor_weight, (1,1,landmark_pred_len))
        _landmark_weight = _landmark_weight.reshape((0, -1, A * landmark_pred_len)).transpose((0,2,1))
        landmark_weight = mx.sym.elemwise_mul(landmark_weight, _landmark_weight, name='%s_landmark_weight_mul_stride%s'%(prefix,stride))
    else:
        label = cls_label
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
    if config.USE_FOCALLOSS:
        print('use focalloss')
        rpn_cls_prob = mx.symbol.Custom(data=rpn_cls_score_reshape, op_type='FocalLoss', labels = label, name='%s_rpn_cls_prob_stride%d'%(prefix,stride), alpha=0.25, gamma=2)
    #print(rpn_cls_prob.list_arguments())
    ret_group.append(rpn_cls_prob)
    ret_group.append(mx.sym.BlockGrad(label))

    if config.TRAIN.RPN_ENABLE_OHEM >= 2:
        valid_count = mx.symbol.sum(valid_count)
        valid_count = valid_count + 0.001 #avoid zero

    #bbox loss
    bbox_diff = rpn_bbox_pred_reshape-bbox_target
    bbox_diff = bbox_diff * bbox_weight
    if not config.USE_KLLOSS:
        bbox_lr_mode0 = 0.25 * lr_mult * config.TRAIN.BATCH_IMAGES / config.TRAIN.RPN_BATCH_SIZE
        landmark_lr_mode0 = 0.4 * config.LANDMARK_LR_MULT * bbox_lr_mode0
        rpn_bbox_loss_ = mx.symbol.smooth_l1(name='%s_rpn_bbox_loss_stride%d_'%(prefix,stride), scalar=3.0, data=bbox_diff)
        if config.LR_MODE==0:
          rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d'%(prefix,stride), data=rpn_bbox_loss_, grad_scale=1.0*lr_mult / (config.TRAIN.RPN_BATCH_SIZE))
        else:
          if config.TRAIN.RPN_ENABLE_OHEM >= 2:
            rpn_bbox_loss_ = mx.symbol.broadcast_div(rpn_bbox_loss_, valid_count)
          rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d'%(prefix,stride), data=rpn_bbox_loss_, grad_scale=0.25*lr_mult)
        ret_group.append(rpn_bbox_loss)
        ret_group.append(mx.sym.BlockGrad(bbox_weight))

    if config.USE_KLLOSS:
        # klloss_ = mx.symbol.Custom(bbox_diff=bbox_diff, alpha= alpha_pred_reshape,op_type='KLLoss',
        #                                 name='%s_kllosst_stride%d' % (prefix, stride))
        # klloss = mx.symbol.MakeLoss(name='%s_klloss_stride%d' % (prefix, stride),data=klloss_,grad_scale=0.1*config.LANDMARK_LR_MULT*lr_mult)
        # ret_group.append(klloss)
        # ret_group.append(mx.symbol.BlockGrad(bbox_diff))
        bbox_diff_abs = mx.symbol.abs(data=bbox_diff, name='abs_bbox_diff') #bbox_l1bs
        bbox_diff_sq = mx.symbol.pow(bbox_diff, 2) #bbox_sq
        eq10_label = mx.symbol.broadcast_greater(bbox_diff_abs, mx.symbol.ones_like(bbox_diff_abs)).astype('float32') #wl1
        eq9_label = mx.symbol.broadcast_lesser_equal(bbox_diff_abs, mx.symbol.ones_like(bbox_diff_abs)).astype('float32') #wl2

        eq9_result = mx.symbol.elemwise_mul(bbox_diff_sq, eq9_label) #bbox_l2_
        eq9_result = mx.symbol.elemwise_mul(eq9_result,0.5*mx.symbol.ones_like(eq9_result)) #bbox_l2_
        eq10_result = mx.symbol.elemwise_sub(bbox_diff_abs, 0.5*mx.symbol.ones_like(bbox_diff_abs))
        eq10_result = mx.symbol.elemwise_mul(eq10_result, eq10_label) #bbox_l1_
        result_1 = mx.symbol.elemwise_add(eq9_result, eq10_result) #bbox_inws

        alpha_pred_reshape_abs = mx.symbol.abs(alpha_pred_reshape)
        alpha_pred_reshape_log = mx.symbol.elemwise_mul(alpha_pred_reshape_abs, 0.5*mx.symbol.ones_like(alpha_pred_reshape_abs)/4)
        #alpha_pred_reshape_log = mx.symbol.broadcast_sub(alpha_pred_reshape_log,mx.symbol.mean(alpha_pred_reshape_log,axis=[1,2],keepdims=True))
        alpha_pred_reshape_negative = mx.symbol.negative(alpha_pred_reshape_log)
        alpha_pred_reshape_exp = mx.symbol.exp(alpha_pred_reshape_negative)

        result_1 = mx.symbol.elemwise_mul(result_1, alpha_pred_reshape_exp) #bbox_inws_out
        result_1 = mx.symbol.broadcast_div(result_1,4*mx.symbol.ones_like(result_1)) #bbox_inws_out
        #if config.try_end2end:
        result_1 = mx.symbol.mean(result_1,axis=(0))
        # bbox_outside_weights = bbox_outside_weights.reshape((0,num_anchors*bbox_pred_len,-1))
        num_example = mx.symbol.sum(label >= 0, axis=[1],keepdims=True)
        _bbox_weight = mx.symbol.reshape(data=bbox_weight, shape=(0,-1,4))
        num_example = mx.symbol.broadcast_like(num_example,_bbox_weight.reshape((0,-1)))
        num_example = mx.symbol.reshape(data= num_example,shape=(0,-1,4))
        # bbox_outside_weights = mx.symbol.zeros_like(alpha_pred_reshape_log)
        bbox_outside_weights = mx.symbol.ones_like(_bbox_weight)
        out_inds = mx.symbol.broadcast_greater_equal(label,mx.symbol.zeros_like(label))
        out_inds = mx.symbol.broadcast_like(out_inds.expand_dims(axis = 2), bbox_outside_weights)
        bbox_outside_weights = mx.symbol.broadcast_mul(bbox_outside_weights,out_inds)
        _bbox_outside_weights = mx.symbol.broadcast_div(bbox_outside_weights, num_example)
        bbox_outside_weights = mx.symbol.reshape(data=_bbox_outside_weights, shape=(0,num_anchors*bbox_pred_len,-1))
        alpha_pred_reshape_logw = mx.symbol.elemwise_mul(alpha_pred_reshape_log, bbox_outside_weights)
        #if config.try_end2end:
        alpha_pred_reshape_logw = mx.symbol.mean(alpha_pred_reshape_logw,axis=(0))
        # alpha_pred_reshape_log_mean = mx.symbol.mean(alpha_pred_reshape_log,axis=[1,2], keepdims=True)
        # alpha_pred_reshape_log_meanr = mx.symbol.broadcast_sub(alpha_pred_reshape_log, alpha_pred_reshape_log_mean)

        log_loss = mx.symbol.sum(alpha_pred_reshape_logw)

        #not scale result1 line165 in fast_rcnn_heads
        # result_1_mean = mx.symbol.mean(result_1, axis = [1,2], keepdims=True)
        # result_1_meanr = mx.symbol.broadcast_sub(result_1, result_1_mean)
        mul_loss = mx.symbol.sum(result_1)

        KL_loss = (log_loss + mul_loss)
        KL_loss = mx.symbol.MakeLoss(KL_loss, name='%s_KL_loss_stride%d'%(prefix,stride), grad_scale=lr_mult)
        ret_group.append(KL_loss)
        # if config.try_end2end:
        #     ret_group.append(mx.symbol.BlockGrad(mx.symbol.ones_like(KL_loss)))
        # else:
        #     ret_group.append(mx.symbol.BlockGrad(bbox_weight))
    # if config.USE_KLLOSS:
    #     # klloss_ = mx.symbol.Custom(bbox_diff=bbox_diff, alpha= alpha_pred_reshape,op_type='KLLoss',
    #     #                                 name='%s_kllosst_stride%d' % (prefix, stride))
    #     # klloss = mx.symbol.MakeLoss(name='%s_klloss_stride%d' % (prefix, stride),data=klloss_,grad_scale=0.1*config.LANDMARK_LR_MULT*lr_mult)
    #     # ret_group.append(klloss)
    #     # ret_group.append(mx.symbol.BlockGrad(bbox_diff))
    #     bbox_diff_abs = mx.symbol.abs(data=bbox_diff, name='abs_bbox_diff') #bbox_l1bs
    #     bbox_diff_sq = mx.symbol.pow(bbox_diff, 2) #bbox_sq
    #     eq10_label = mx.symbol.broadcast_greater(bbox_diff_abs, mx.symbol.ones_like(bbox_diff_abs))
    #     eq9_label = mx.symbol.broadcast_lesser_equal(bbox_diff_abs, mx.symbol.ones_like(bbox_diff_abs))
    #
    #     eq9_result = mx.symbol.elemwise_mul(bbox_diff_sq, eq9_label)
    #     eq9_result = mx.symbol.elemwise_mul(eq9_result,0.5*mx.symbol.ones_like(eq9_result))
    #     eq10_result = mx.symbol.elemwise_sub(bbox_diff_abs, 0.5*mx.symbol.ones_like(bbox_diff_abs))
    #     eq10_result = mx.symbol.elemwise_mul(eq10_result, eq10_label)
    #     result_1 = mx.symbol.elemwise_add(eq9_result, eq10_result) #bbox_inws
    #
    #     alpha_pred_reshape_abs = mx.symbol.abs(alpha_pred_reshape)
    #     alpha_pred_reshape_log = mx.symbol.elemwise_mul(alpha_pred_reshape_abs, 0.5*mx.symbol.ones_like(alpha_pred_reshape_abs)/4)
    #     #alpha_pred_reshape_log = mx.symbol.broadcast_sub(alpha_pred_reshape_log,mx.symbol.mean(alpha_pred_reshape_log,axis=[1,2],keepdims=True))
    #     alpha_pred_reshape_negative = mx.symbol.negative(alpha_pred_reshape_log)
    #     alpha_pred_reshape_exp = mx.symbol.exp(alpha_pred_reshape_negative)
    #
    #     result_1 = mx.symbol.elemwise_mul(result_1, alpha_pred_reshape_exp)
    #     result_1 = mx.symbol.broadcast_div(result_1,4*mx.symbol.ones_like(result_1)) #bbox_inws_out
    #     if config.try_end2end:
    #         result_1 = mx.symbol.mean(result_1,axis=(1,2))
    #     num_example = mx.symbol.sum(label >= 0, axis=[1],keepdims=True)
    #     _bbox_weight = mx.symbol.reshape(data=bbox_weight, shape=(0,-1))
    #     # bbox_outside_weights = mx.symbol.zeros_like(alpha_pred_reshape_log)
    #     _bbox_outside_weights = mx.symbol.broadcast_div(_bbox_weight, num_example)
    #     bbox_outside_weights = mx.symbol.reshape(data=_bbox_outside_weights, shape=(0,num_anchors*bbox_pred_len,-1))
    #     alpha_pred_reshape_logw = mx.symbol.elemwise_mul(alpha_pred_reshape_log, bbox_outside_weights)
    #     if config.try_end2end:
    #         alpha_pred_reshape_logw = mx.symbol.mean(alpha_pred_reshape_logw,axis=(1,2))
    #     # alpha_pred_reshape_log_mean = mx.symbol.mean(alpha_pred_reshape_log,axis=[1,2], keepdims=True)
    #     # alpha_pred_reshape_log_meanr = mx.symbol.broadcast_sub(alpha_pred_reshape_log, alpha_pred_reshape_log_mean)
    #
    #     log_loss = alpha_pred_reshape_logw
    #
    #     #not scale result1 line165 in fast_rcnn_heads
    #     # result_1_mean = mx.symbol.mean(result_1, axis = [1,2], keepdims=True)
    #     # result_1_meanr = mx.symbol.broadcast_sub(result_1, result_1_mean)
    #     mul_loss = result_1
    #
    #     KL_loss = log_loss + mul_loss
    #     KL_loss = mx.symbol.make_loss(KL_loss, name='%s_KL_loss_stride%d'%(prefix,stride), grad_scale=0.25*lr_mult)
    #     ret_group.append(KL_loss)
    #     if config.try_end2end:
    #         ret_group.append(mx.symbol.BlockGrad(mx.symbol.ones_like(KL_loss)))
    #     else:
    #         ret_group.append(mx.symbol.BlockGrad(bbox_weight))

    #landmark loss
    if landmark:
      landmark_diff = rpn_landmark_pred_reshape-landmark_target
      landmark_diff = landmark_diff * landmark_weight
      rpn_landmark_loss_ = mx.symbol.smooth_l1(name='%s_rpn_landmark_loss_stride%d_'%(prefix,stride), scalar=3.0, data=landmark_diff)
      if config.LR_MODE==0:
        rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d'%(prefix,stride), data=rpn_landmark_loss_, grad_scale=0.4*config.LANDMARK_LR_MULT*lr_mult / (config.TRAIN.RPN_BATCH_SIZE))
      else:
        if config.TRAIN.RPN_ENABLE_OHEM >= 2:
            rpn_landmark_loss_ = mx.symbol.broadcast_div(rpn_landmark_loss_, valid_count)
        rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d'%(prefix,stride), data=rpn_landmark_loss_, grad_scale=0.1*config.LANDMARK_LR_MULT*lr_mult)
      ret_group.append(rpn_landmark_loss)
      ret_group.append(mx.sym.BlockGrad(landmark_weight))

    if config.landmark_klloss:
        landmark_diff_abs = mx.symbol.abs(data=landmark_diff, name='abs_landmark_diff')
        landmark_diff_sq = mx.symbol.pow(landmark_diff, 2)
        eq10_label = mx.symbol.broadcast_greater(landmark_diff_abs, mx.symbol.ones_like(landmark_diff_abs))
        eq9_label = mx.symbol.broadcast_lesser_equal(landmark_diff_abs, mx.symbol.ones_like(landmark_diff_abs))

        eq9_result = mx.symbol.elemwise_mul(landmark_diff_sq, eq9_label)
        eq9_result = mx.symbol.elemwise_mul(eq9_result, 0.5 * mx.symbol.ones_like(eq9_result))
        eq10_result = mx.symbol.elemwise_sub(landmark_diff_abs, 0.5 * mx.symbol.ones_like(landmark_diff_abs))
        eq10_result = mx.symbol.elemwise_mul(eq10_result, eq10_label)
        result_1 = mx.symbol.elemwise_add(eq9_result, eq10_result)  # bbox_inws

        beta_pred_reshape_abs = mx.symbol.abs(beta_pred_reshape)
        beta_pred_reshape_log = mx.symbol.elemwise_mul(beta_pred_reshape_abs,
                                                        0.5 * mx.symbol.ones_like(beta_pred_reshape_abs) / 4)
        # alpha_pred_reshape_log = mx.symbol.broadcast_sub(alpha_pred_reshape_log,mx.symbol.mean(alpha_pred_reshape_log,axis=[1,2],keepdims=True))
        beta_pred_reshape_negative = mx.symbol.negative(beta_pred_reshape_log)
        beta_pred_reshape_exp = mx.symbol.exp(beta_pred_reshape_negative)

        result_1 = mx.symbol.elemwise_mul(result_1, beta_pred_reshape_exp)
        result_1 = mx.symbol.broadcast_div(result_1, 4 * mx.symbol.ones_like(result_1))

        num_example = mx.symbol.sum(label >= 0, axis=[1], keepdims=True)
        _landmark_weight = mx.symbol.reshape(data=landmark_weight, shape=(0, -1))
        # bbox_outside_weights = mx.symbol.zeros_like(alpha_pred_reshape_log)
        _landmark_outside_weights = mx.symbol.broadcast_div(_landmark_weight, num_example)
        landmark_outside_weights = mx.symbol.reshape(data=_landmark_outside_weights, shape=(0, num_anchors * landmark_pred_len, -1))
        beta_pred_reshape_logw = mx.symbol.elemwise_mul(beta_pred_reshape_log, landmark_outside_weights)
        # alpha_pred_reshape_log_mean = mx.symbol.mean(alpha_pred_reshape_log,axis=[1,2], keepdims=True)
        # alpha_pred_reshape_log_meanr = mx.symbol.broadcast_sub(alpha_pred_reshape_log, alpha_pred_reshape_log_mean)

        log_loss = beta_pred_reshape_logw

        # not scale result1 line165 in fast_rcnn_heads
        # result_1_mean = mx.symbol.mean(result_1, axis = [1,2], keepdims=True)
        # result_1_meanr = mx.symbol.broadcast_sub(result_1, result_1_mean)
        mul_loss = result_1

        KL_loss = log_loss + mul_loss
        KL_loss_l = mx.symbol.MakeLoss(KL_loss, name='%s_KL_loss_landmark_stride%d' % (prefix, stride), grad_scale=0.25 * lr_mult)
        ret_group.append(KL_loss_l)
        ret_group.append(mx.symbol.BlockGrad(landmark_weight))
    if config.USE_3D:
      from rcnn.PY_OP import rpn_3d_mesh
      pass

    # if config.STR and stride in config.STR_FPN_STRIDE:
    #     body = head_module(rpn_relu, F1 * config.CONTEXT_FILTER_RATIO, F2, 'rf_head_stride%d_cas' % stride)
    #     cls_score_ft = rpn_cls_score_reshape
    #     cls_label_raw = cls_label
    #     cls_label = label
    #     bbox_pred_ft = rpn_bbox_pred_reshape
    #     bbox_target_ft = bbox_target
    #     landmark_target_ft = landmark_target
    #
    #     cls_pred = conv_only(body, 'STR_%s_rpn_cls_score_stride%d' % (prefix, stride), 2 * num_anchors,
    #                          kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    #     rpn_cls_score_reshape = mx.symbol.Reshape(data=cls_pred,
    #                                               shape=(0, 2, -1),
    #                                               name="STR_%s_rpn_cls_score_reshape_stride%s" % (
    #                                               prefix, stride))
    #     cls_label, bbox_target_st,landmark_target_st, anchor_weight, pos_count = mx.sym.Custom(op_type='STR', stride=int(stride),
    #                                                                     network=config.network,
    #                                                                     dataset=config.dataset, prefix=prefix,
    #                                                                     cls_label_t0=cls_label,
    #                                                                     cls_pred_t0=cls_score_ft,
    #                                                                     cls_pred=rpn_cls_score_reshape,
    #                                                                     bbox_pred_t0=bbox_pred_ft,
    #                                                                     bbox_label_t0=bbox_target_ft,
    #                                                                     cls_label_raw=cls_label_raw,
    #                                                                     cas_gt_boxes=gt_boxes,
    #                                                                     cas_gt_landmarks = gt_landmarks,
    #                                                                     landmark_target_t0 = landmark_target_ft)
    #     if config.STR_cls:
    #         rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape,
    #                                                label=cls_label,
    #                                                multi_output=True,
    #                                                normalization='valid', use_ignore=True, ignore_label=-1,
    #                                                grad_scale=lr_mult,
    #                                                name='STR_%s_rpn_cls_prob_stride%d' % (prefix, stride))
    #         ret_group.append(rpn_cls_prob)
    #         ret_group.append(mx.sym.BlockGrad(cls_label))
    #     bbox_pred = conv_only(body, 'STR_%s_rpn_bbox_pred_stride%d' % (prefix, stride),
    #                           bbox_pred_len * num_anchors,
    #                           kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    #
    #     rpn_bbox_pred_reshape = mx.symbol.Reshape(data=bbox_pred,
    #                                               shape=(0, 0, -1),
    #                                               name="STR_%s_rpn_bbox_pred_reshape_stride%s" % (
    #                                               prefix, stride))
    #     if landmark and config.STR_landmark:
    #         rpn_landmark_pred = conv_only(body, 'STR_%s_rpn_landmark_pred_stride%d' % (prefix, stride),
    #                                       landmark_pred_len * num_anchors,
    #                                       kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=shared_vars[2][0],
    #                                       shared_bias=shared_vars[2][1])
    #         rpn_landmark_pred_reshape = mx.symbol.Reshape(data=rpn_landmark_pred,
    #                                                       shape=(0, 0, -1),
    #                                                       name="STR_%s_rpn_landmark_pred_reshape_stride%s" % (
    #                                                       prefix, stride))
    #     _bbox_weight = mx.sym.tile(anchor_weight, (1, 1, bbox_pred_len))
    #     _bbox_weight = _bbox_weight.reshape((0, -1, A * bbox_pred_len)).transpose((0, 2, 1))
    #     bbox_weight = _bbox_weight
    #     if landmark and config.STR_landmark:
    #         _landmark_weight = mx.sym.tile(anchor_weight, (1, 1, landmark_pred_len))
    #         _landmark_weight = _landmark_weight.reshape((0, -1, A * landmark_pred_len)).transpose((0, 2, 1))
    #         landmark_weight = _landmark_weight
    #     pos_count = mx.symbol.sum(pos_count)
    #     pos_count = pos_count + 0.01  # avoid zero
    #     # bbox_weight = mx.sym.elemwise_mul(bbox_weight, _bbox_weight, name='%s_bbox_weight_mul_stride%s'%(prefix,stride))
    #     # bbox loss
    #     bbox_diff = rpn_bbox_pred_reshape - bbox_target_st
    #     bbox_diff = bbox_diff * bbox_weight
    #     rpn_bbox_loss_ = mx.symbol.smooth_l1(name='STR_%s_rpn_bbox_loss_stride%d' % (prefix, stride),
    #                                          scalar=3.0, data=bbox_diff)
    #     if config.LR_MODE == 0:
    #         rpn_bbox_loss = mx.sym.MakeLoss(name='STR_%s_rpn_bbox_loss_stride%d' % (prefix, stride),
    #                                         data=rpn_bbox_loss_, grad_scale=bbox_lr_mode0)
    #     else:
    #         rpn_bbox_loss_ = mx.symbol.broadcast_div(rpn_bbox_loss_, pos_count)
    #         rpn_bbox_loss = mx.sym.MakeLoss(name='STR_%s_rpn_bbox_loss_stride%d' % (prefix, stride),
    #                                         data=rpn_bbox_loss_, grad_scale=0.5 * lr_mult)
    #     ret_group.append(rpn_bbox_loss)
    #     ret_group.append(mx.sym.BlockGrad(bbox_weight))
    #
    #     if landmark and config.STR_landmark:
    #         landmark_diff = rpn_landmark_pred_reshape - landmark_target_st
    #         landmark_diff = landmark_diff * landmark_weight
    #         rpn_landmark_loss_ = mx.symbol.smooth_l1(name='STR_%s_rpn_landmark_loss_stride%d_' % (prefix, stride),
    #                                                  scalar=3.0, data=landmark_diff)
    #         if config.LR_MODE == 0:
    #             rpn_landmark_loss = mx.sym.MakeLoss(name='STR_%s_rpn_landmark_loss_stride%d' % (prefix, stride),
    #                                                 data=rpn_landmark_loss_, grad_scale=landmark_lr_mode0)
    #         else:
    #             rpn_landmark_loss_ = mx.symbol.broadcast_div(rpn_landmark_loss_, pos_count)
    #             rpn_landmark_loss = mx.sym.MakeLoss(name='STR_%s_rpn_landmark_loss_stride%d' % (prefix, stride),
    #                                                 data=rpn_landmark_loss_,
    #                                                 grad_scale=0.1*config.LANDMARK_LR_MULT*lr_mult)
    #         ret_group.append(rpn_landmark_loss)
    #         ret_group.append(mx.sym.BlockGrad(landmark_weight))


    return ret_group


def get_out_af(conv_fpn_feat, prefix, stride, min_stride,landmark=False, lr_mult=1.0, shared_vars = None, gt_boxes = None, gt_landmarks = None):
    ret_group = []
    hm = mx.symbol.Variable(name='ct_hm_stride_%s'%stride)
    wh = mx.symbol.Variable(name='ct_wh_stride_%s' % stride)
    wh_mask = mx.symbol.Variable(name='ct_wh_mask_stride_%s' % stride)
    if config.ct_offset:
        offset = mx.symbol.Variable(name='ct_offset_stride_%s' % stride)
    if config.ct_landmarks:
        landmark_label = mx.symbol.Variable(name='ct_landmark_stride_%s' % stride)
    ind = mx.symbol.Variable(name='ct_ind_stride_%s' % stride)
    rpn_relu = conv_fpn_feat[stride]
    if config.ct_context_module:
        rpn_relu = head_module(rpn_relu, 64 * config.CONTEXT_FILTER_RATIO, 64, '%s_rf_head_stride%d' % (prefix, stride))
    hm_1 =  conv_only(rpn_relu, 'hm_1_stride%d'%stride, 64,
          kernel=(3,3), pad=(1,1), stride=(1, 1), shared_weight = None, shared_bias = None)
    hm_2 = mx.symbol.Activation(data=hm_1, act_type='relu',
                                name="hm_relu_stride_%s"%stride)
    if config.ct_hmclasses:
        hm_pred = conv_only(hm_2, 'hm_pred_stride%d' % stride, config.ct_hmclasses,
                            kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=None,
                            shared_bias=None)
        hm_pred = mx.symbol.softmax(data=hm_pred,axis=1)
    else:
        hm_pred = conv_only(hm_2, 'hm_pred_stride%d'%stride, 1,
              kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = None, shared_bias = None)
        # bias = mx.symbol.Variable(name='bias',init=mx.init.Constant(-2.19))
        # hm_pred = mx.symbol.Convolution(data=hm_2, kernel=(1,1), pad=(0,0), stride=(1, 1), num_filter=1, name='hm_pred_stride%d'%stride,
        #                              bias=bias)
        hm_pred = mx.symbol.sigmoid(hm_pred)
    hm_pred = mx.symbol.clip(hm_pred,a_min= 1e-14,a_max=1-1e-4)
    def hm_loss(pred, gt):
        pos_inds = mx.symbol.broadcast_equal(gt, mx.symbol.ones_like(gt))
        neg_inds = mx.symbol.broadcast_lesser(gt, mx.symbol.ones_like(gt))
        neg_weights = mx.symbol.pow(1-gt, 4)

        loss = 0

        pos_loss = mx.symbol.log(pred) * mx.symbol.pow(1 - pred, 2) * pos_inds
        neg_loss = mx.symbol.log(1 - pred) * mx.symbol.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.astype('float32').sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # if num_pos.__eq__(mx.symbol.zeros_like(num_pos)).__bool__:
        #     loss = loss - neg_loss
        # else:
        #     loss = loss - (pos_loss + neg_loss) / num_pos
        num_pos  = num_pos*10 + 1
        loss = loss - (pos_loss + neg_loss) / (num_pos)

        return loss
    if config.ct_reproduce:
        hm_loss_1 = mx.symbol.Custom(op_type='hm_focal_loss', data = hm_pred,label = hm)
    else:
        hm_loss_1 = hm_loss(hm_pred,hm)
        # hm_loss_1 = mx.symbol.Custom(op_type='hm_focal_loss', data=hm_pred, label=hm)
    heatmap_loss = mx.symbol.MakeLoss(data = hm_loss_1)
    ret_group.append(heatmap_loss)

    wh_1 = conv_only(rpn_relu, 'wh_1_stride%d'%stride, 64,
          kernel=(3,3), pad=(1,1), stride=(1, 1), shared_weight = None, shared_bias = None)
    wh_2 = mx.symbol.Activation(data=wh_1, act_type='relu',
                                name="wh_relu_stride_%s"%stride)
    wh_pred = conv_only(wh_2, 'wh_pred_stride%d'%stride, 2,
          kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = None, shared_bias = None)
    #wh_pred,wh_mask = mx.sym.Custom(op_type='wh_process', wh_pred=wh_pred,ind = ind, mask = wh_mask )
    if config.ct_wh_inds:
        wh_pred = mx.sym.Custom(op_type='wh_process', wh_pred=wh_pred, ind=ind)
        wh_mask = mx.symbol.broadcast_like(wh_mask.expand_dims(axis = 2), wh_pred)
        wh_loss = wh_pred * wh_mask - wh * wh_mask
        wh_loss = mx.symbol.sum(mx.symbol.abs(wh_loss))
        wh_loss = wh_loss / (mx.symbol.sum(wh_mask) + 1e-4)
        wh_loss = wh_loss / 10
        wh_loss1 = mx.symbol.make_loss(wh_loss)
        ret_group.append(wh_loss1)
        if config.ct_offset:
            offset_1 = conv_only(rpn_relu, 'offset_1_stride%d' % stride, 64,
                             kernel=(3, 3), pad=(1, 1), stride=(1, 1), shared_weight=None,
                             shared_bias=None)
            offset_2 = mx.symbol.Activation(data=offset_1, act_type='relu',
                                        name="offset_relu_stride_%s" % stride)
            offset_pred = conv_only(offset_2, 'offset_pred_stride%d' % stride, 2,
                                kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=None,
                                shared_bias=None)
            offset_pred = mx.sym.Custom(op_type='wh_process', wh_pred=offset_pred, ind=ind)
            offset_loss = offset_pred * wh_mask - offset * wh_mask
            offset_loss = mx.symbol.sum(mx.symbol.abs(offset_loss))
            offset_loss = offset_loss / (mx.symbol.sum(wh_mask) + 1e-4)
            offset_loss = offset_loss
            offset_loss1 = mx.symbol.make_loss(offset_loss)
            ret_group.append(offset_loss1)
    else:
        wh_mask_ori = wh_mask
        wh_mask = mx.symbol.broadcast_like(wh_mask,wh_pred)
        wh_loss = wh_pred * wh_mask - wh * wh_mask
        # if config.ct_wh_log:
        #     wh_loss = mx.symbol.smooth_l1(name='wh_loss_stride%d_'%(stride), scalar=3.0, data=wh_loss)
        # else:
        wh_loss = mx.symbol.sum(mx.symbol.abs(wh_loss))
        wh_loss = wh_loss / (mx.symbol.sum(wh_mask) + 1e-4)
        wh_loss = wh_loss / 10
        wh_loss1 = mx.symbol.make_loss(wh_loss)
        ret_group.append(wh_loss1)

        if config.ct_offset:
            offset_1 = conv_only(rpn_relu, 'offset_1_stride%d' % stride, 64,
                             kernel=(3, 3), pad=(1, 1), stride=(1, 1), shared_weight=None,
                             shared_bias=None)
            offset_2 = mx.symbol.Activation(data=offset_1, act_type='relu',
                                        name="offset_relu_stride_%s" % stride)
            offset_pred = conv_only(offset_2, 'offset_pred_stride%d' % stride, 2,
                                kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=None,
                                shared_bias=None)
            offset_loss = offset_pred * wh_mask - offset * wh_mask
            offset_loss = mx.symbol.sum(mx.symbol.abs(offset_loss))
            offset_loss = offset_loss / (mx.symbol.sum(wh_mask) + 1e-4)
            offset_loss = offset_loss
            offset_loss1 = mx.symbol.make_loss(offset_loss)
            ret_group.append(offset_loss1)
    if config.ct_landmarks:
        landmark_1 = conv_only(rpn_relu, 'landmark_1_stride%d' % stride, 64,
                         kernel=(3, 3), pad=(1, 1), stride=(1, 1), shared_weight=shared_vars[0][0],
                         shared_bias=shared_vars[0][1])
        landmark_2 = mx.symbol.Activation(data=landmark_1, act_type='relu',
                                    name="landmark_relu_stride_%s" % stride)
        landmark_pred = conv_only(landmark_2, 'landmark_pred_stride%d' % stride, 10,
                            kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=shared_vars[0][0],
                            shared_bias=shared_vars[0][1])
        landmark_mask = mx.symbol.broadcast_like(wh_mask_ori,landmark_pred)
        # landmark_pred_reshape = mx.symbol.Reshape(data=landmark_pred,
        #                                               shape=(0, 0, -1),
        #                                               name="landmark_pred_stride_%s" % (stride))
        landmark_loss = landmark_pred * landmark_mask - landmark_label * landmark_mask
        #landmark_loss = mx.symbol.smooth_l1(name='landmark_loss_stride%d_'%(stride), scalar=3.0, data=landmark_loss)
        landmark_loss = mx.symbol.sum(mx.symbol.abs(landmark_loss))
        landmark_loss = landmark_loss / (mx.symbol.sum(landmark_mask)*10 + 1e-4)
        landmark_loss1 = mx.symbol.MakeLoss(data=landmark_loss)
        ret_group.append(landmark_loss1)
        # ret_group.append(mx.symbol.BlockGrad(landmark_mask))

    # def _gather_feat(feat, ind, mask=None):
    #     # K cannot be 1 for this implementation
    #     batch_size = 8
    #
    #     flatten_ind = ind.flatten()
    #     for i in range(batch_size):
    #         if i == 0:
    #             output = feat[i, ind[i]].expand_dims(2)
    #         else:
    #             output = nd.concat(output, feat[i, ind[i]].expand_dims(2), dim=2)
    #
    #     output = output.swapaxes(dim1=1, dim2=2)
    #     return output
    #
    # def _tranpose_and_gather_feat(feat, ind):
    #     feat = mx.symbol.transpose(feat, axes=(0, 2, 3, 1))
    #     feat = mx.symbol.reshape(feat, shape=(feat.shape[0], -1, feat.shape[3]))
    #     feat = _gather_feat(feat, ind)
    #     return feat

    # wh_pred = _tranpose_and_gather_feat(wh_pred,ind)
    if config.ct_mining and stride in config.ct_mining_stride:
        cls_label = mx.symbol.Variable(name='%s_label_stride%d' % (prefix, stride))
        bbox_weight = mx.symbol.Variable(name='%s_bbox_weight_stride%d' % (prefix, stride))
        # overlaps = mx.symbol.Variable(name='overlaps')
        mined_labels, mined_bbox_weight = mx.symbol.Custom(op_type='af_mining', hm_pred=hm_pred, wh_pred=wh_pred,
                                                           offset_pred=offset_pred,
                                                           gt_boxes=gt_boxes, labels=cls_label,
                                                           bbox_weight=bbox_weight)
        return ret_group, mined_labels, mined_bbox_weight

    else:
        return ret_group

def get_out_af_head(conv_fpn_feat, prefix, stride, min_stride,landmark=False, lr_mult=1.0, shared_vars = None, gt_boxes = None, gt_landmarks = None):
    ret_group = []
    hm = mx.symbol.Variable(name='ct_head_hm_stride_%s'%stride)
    wh = mx.symbol.Variable(name='ct_head_wh_stride_%s' % stride)
    wh_mask = mx.symbol.Variable(name='ct_head_wh_mask_stride_%s' % stride)
    if config.ct_offset:
        offset = mx.symbol.Variable(name='ct_head_offset_stride_%s' % stride)
    if config.ct_landmarks:
        landmark_label = mx.symbol.Variable(name='ct_head_landmark_stride_%s' % stride)
    ind = mx.symbol.Variable(name='ct_head_ind_stride_%s' % stride)
    rpn_relu = conv_fpn_feat[stride]
    if config.ct_context_module:
        rpn_relu = head_module(rpn_relu, 64 * config.CONTEXT_FILTER_RATIO, 64, '%s_rf_head_stride%d' % (prefix, stride))
    hm_1 =  conv_only(rpn_relu, 'head_hm_1_stride%d'%stride, 64,
          kernel=(3,3), pad=(1,1), stride=(1, 1), shared_weight = None, shared_bias = None)
    hm_2 = mx.symbol.Activation(data=hm_1, act_type='relu',
                                name="head_hm_relu_stride_%s"%stride)
    if config.ct_hmclasses:
        hm_pred = conv_only(hm_2, 'head_hm_pred_stride%d' % stride, config.ct_hmclasses,
                            kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=None,
                            shared_bias=None)
        hm_pred = mx.symbol.softmax(data=hm_pred,axis=1)
    else:
        hm_pred = conv_only(hm_2, 'head_hm_pred_stride%d'%stride, 1,
              kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = None, shared_bias = None)
        # bias = mx.symbol.Variable(name='bias',init=mx.init.Constant(-2.19))
        # hm_pred = mx.symbol.Convolution(data=hm_2, kernel=(1,1), pad=(0,0), stride=(1, 1), num_filter=1, name='hm_pred_stride%d'%stride,
        #                              bias=bias)
        hm_pred = mx.symbol.sigmoid(hm_pred)
    hm_pred = mx.symbol.clip(hm_pred,a_min= 1e-14,a_max=1-1e-4)
    def hm_loss(pred, gt):
        pos_inds = mx.symbol.broadcast_equal(gt, mx.symbol.ones_like(gt))
        neg_inds = mx.symbol.broadcast_lesser(gt, mx.symbol.ones_like(gt))
        neg_weights = mx.symbol.pow(1-gt, 4)

        loss = 0

        pos_loss = mx.symbol.log(pred) * mx.symbol.pow(1 - pred, 2) * pos_inds
        neg_loss = mx.symbol.log(1 - pred) * mx.symbol.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.astype('float32').sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # if num_pos.__eq__(mx.symbol.zeros_like(num_pos)).__bool__:
        #     loss = loss - neg_loss
        # else:
        #     loss = loss - (pos_loss + neg_loss) / num_pos
        num_pos  = num_pos*10 + 1
        loss = loss - (pos_loss + neg_loss) / (num_pos)

        return loss
    if config.ct_reproduce:
        hm_loss_1 = mx.symbol.Custom(op_type='hm_focal_loss', data = hm_pred,label = hm)
    else:
        hm_loss_1 = hm_loss(hm_pred,hm)
        # hm_loss_1 = mx.symbol.Custom(op_type='hm_focal_loss', data=hm_pred, label=hm)
    heatmap_loss = mx.symbol.MakeLoss(data = hm_loss_1)
    ret_group.append(heatmap_loss)

    wh_1 = conv_only(rpn_relu, 'head_wh_1_stride%d'%stride, 64,
          kernel=(3,3), pad=(1,1), stride=(1, 1), shared_weight = None, shared_bias = None)
    wh_2 = mx.symbol.Activation(data=wh_1, act_type='relu',
                                name="head_wh_relu_stride_%s"%stride)
    wh_pred = conv_only(wh_2, 'head_wh_pred_stride%d'%stride, 2,
          kernel=(1,1), pad=(0,0), stride=(1, 1), shared_weight = None, shared_bias = None)
    #wh_pred,wh_mask = mx.sym.Custom(op_type='wh_process', wh_pred=wh_pred,ind = ind, mask = wh_mask )
    if config.ct_wh_inds:
        wh_pred = mx.sym.Custom(op_type='wh_process', wh_pred=wh_pred, ind=ind)
        wh_mask = mx.symbol.broadcast_like(wh_mask.expand_dims(axis = 2), wh_pred)
        wh_loss = wh_pred * wh_mask - wh * wh_mask
        wh_loss = mx.symbol.sum(mx.symbol.abs(wh_loss))
        wh_loss = wh_loss / (mx.symbol.sum(wh_mask) + 1e-4)
        wh_loss = wh_loss / 10
        wh_loss1 = mx.symbol.make_loss(wh_loss)
        ret_group.append(wh_loss1)
        if config.ct_offset:
            offset_1 = conv_only(rpn_relu, 'offset_1_stride%d' % stride, 64,
                             kernel=(3, 3), pad=(1, 1), stride=(1, 1), shared_weight=None,
                             shared_bias=None)
            offset_2 = mx.symbol.Activation(data=offset_1, act_type='relu',
                                        name="offset_relu_stride_%s" % stride)
            offset_pred = conv_only(offset_2, 'offset_pred_stride%d' % stride, 2,
                                kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=None,
                                shared_bias=None)
            offset_pred = mx.sym.Custom(op_type='wh_process', wh_pred=offset_pred, ind=ind)
            offset_loss = offset_pred * wh_mask - offset * wh_mask
            offset_loss = mx.symbol.sum(mx.symbol.abs(offset_loss))
            offset_loss = offset_loss / (mx.symbol.sum(wh_mask) + 1e-4)
            offset_loss = offset_loss
            offset_loss1 = mx.symbol.make_loss(offset_loss)
            ret_group.append(offset_loss1)
    else:
        wh_mask_ori = wh_mask
        wh_mask = mx.symbol.broadcast_like(wh_mask,wh_pred)
        wh_loss = wh_pred * wh_mask - wh * wh_mask
        # if config.ct_wh_log:
        #     wh_loss = mx.symbol.smooth_l1(name='wh_loss_stride%d_'%(stride), scalar=3.0, data=wh_loss)
        # else:
        wh_loss = mx.symbol.sum(mx.symbol.abs(wh_loss))
        wh_loss = wh_loss / (mx.symbol.sum(wh_mask) + 1e-4)
        wh_loss = wh_loss / 10
        wh_loss1 = mx.symbol.make_loss(wh_loss)
        ret_group.append(wh_loss1)

        if config.ct_offset:
            offset_1 = conv_only(rpn_relu, 'head_offset_1_stride%d' % stride, 64,
                             kernel=(3, 3), pad=(1, 1), stride=(1, 1), shared_weight=None,
                             shared_bias=None)
            offset_2 = mx.symbol.Activation(data=offset_1, act_type='relu',
                                        name="head_offset_relu_stride_%s" % stride)
            offset_pred = conv_only(offset_2, 'head_offset_pred_stride%d' % stride, 2,
                                kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=None,
                                shared_bias=None)
            offset_loss = offset_pred * wh_mask - offset * wh_mask
            offset_loss = mx.symbol.sum(mx.symbol.abs(offset_loss))
            offset_loss = offset_loss / (mx.symbol.sum(wh_mask) + 1e-4)
            offset_loss = offset_loss
            offset_loss1 = mx.symbol.make_loss(offset_loss)
            ret_group.append(offset_loss1)
    if config.ct_landmarks:
        landmark_1 = conv_only(rpn_relu, 'head_landmark_1_stride%d' % stride, 64,
                         kernel=(3, 3), pad=(1, 1), stride=(1, 1), shared_weight=shared_vars[0][0],
                         shared_bias=shared_vars[0][1])
        landmark_2 = mx.symbol.Activation(data=landmark_1, act_type='relu',
                                    name="head_landmark_relu_stride_%s" % stride)
        landmark_pred = conv_only(landmark_2, 'head_landmark_pred_stride%d' % stride, 10,
                            kernel=(1, 1), pad=(0, 0), stride=(1, 1), shared_weight=shared_vars[0][0],
                            shared_bias=shared_vars[0][1])
        landmark_mask = mx.symbol.broadcast_like(wh_mask_ori,landmark_pred)
        # landmark_pred_reshape = mx.symbol.Reshape(data=landmark_pred,
        #                                               shape=(0, 0, -1),
        #                                               name="landmark_pred_stride_%s" % (stride))
        landmark_loss = landmark_pred * landmark_mask - landmark_label * landmark_mask
        #landmark_loss = mx.symbol.smooth_l1(name='landmark_loss_stride%d_'%(stride), scalar=3.0, data=landmark_loss)
        landmark_loss = mx.symbol.sum(mx.symbol.abs(landmark_loss))
        landmark_loss = landmark_loss / (mx.symbol.sum(landmark_mask)*10 + 1e-4)
        landmark_loss1 = mx.symbol.MakeLoss(data=landmark_loss)
        ret_group.append(landmark_loss1)
        # ret_group.append(mx.symbol.BlockGrad(landmark_mask))


    return ret_group


def get_sym_train(sym, sel_stride, flag):
    data = mx.symbol.Variable(name="data")

    # shared convolutional layers
    conv_fpn_feat = get_sym_conv(data, sym)
    ret_group = []
    shared_vars = []
    gt_boxes = None
    gt_landmarks = None
    # if config.SHARE_WEIGHT_BBOX:
    #   assert config.USE_MAXOUT==0
    #   _name = 'face_rpn_cls_score_share'
    #   shared_weight = mx.symbol.Variable(name="{}_weight".format(_name),
    #       init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    #   shared_bias = mx.symbol.Variable(name="{}_bias".format(_name),
    #       init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(0.0)})
    #   shared_vars.append( [shared_weight, shared_bias] )
    #   _name = 'face_rpn_bbox_pred_share'
    #   shared_weight = mx.symbol.Variable(name="{}_weight".format(_name),
    #       init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    #   shared_bias = mx.symbol.Variable(name="{}_bias".format(_name),
    #       init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(0.0)})
    #   shared_vars.append( [shared_weight, shared_bias] )
    # else:
    #   shared_vars.append( [None, None] )
    #   shared_vars.append( [None, None] )
    # if config.SHARE_WEIGHT_LANDMARK:
    #   _name = 'face_rpn_landmark_pred_share'
    #   shared_weight = mx.symbol.Variable(name="{}_weight".format(_name),
    #       init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    #   shared_bias = mx.symbol.Variable(name="{}_bias".format(_name),
    #       init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(0.0)})
    #   shared_vars.append( [shared_weight, shared_bias] )
    # else:
    #   shared_vars.append( [None, None] )
    if config.deephead:
        for tmp_type in ['cls', 'box', 'ldk']:
            for tmp_i in range(1,5):
                shared_pairs = []
                for hz in ['_weight','_bias']:
                    tmp_name = tmp_type + '_conv' + str(tmp_i) + '_3x3'+hz
                    shared_item = mx.symbol.Variable(name="{}".format(tmp_name),
                          init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
                    shared_pairs.append(shared_item)
                shared_vars.append(shared_pairs)
            shared_pairs = []
            for hz in ['_weight', '_bias']: # don't share weight and bias for the last head
                tmp_name = tmp_type + '_score' + hz
                shared_item = mx.symbol.Variable(name="{}".format(tmp_name),
                                                 init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
                shared_pairs.append(shared_item)
            shared_vars.append(shared_pairs)
    else:
        for i in range(15):
            shared_vars.append([None, None])
    if config.STR:
        gt_boxes = mx.symbol.Variable('gt_boxes')
        if config.FACE_LANDMARK:
            gt_landmarks = mx.symbol.Variable('gt_landmarks')
    if config.ct_mining and config.centernet_branch:
        gt_boxes = mx.symbol.Variable('gt_boxes')
    min_stride = 10000
    for stride in config.RPN_FEAT_STRIDE:
        if stride < min_stride:
            min_stride = stride
    if config.selective_branch and config.PROGRESSIVE:
        if flag == True:
            for stride in config.RPN_FEAT_STRIDE:
                ret = get_out(conv_fpn_feat, 'face', stride, min_stride, config.FACE_LANDMARK, lr_mult=1.0,
                              shared_vars=shared_vars)
                ret_group += ret
                if config.HEAD_BOX:
                    assert not config.SHARE_WEIGHT_BBOX and not config.SHARE_WEIGHT_LANDMARK
                    shared_vars = [[None, None], [None, None], [None, None]]
                    ret = get_out(conv_fpn_feat, 'head', stride, False, lr_mult=0.5, shared_vars=shared_vars)
                    ret_group += ret
        else:
            for stride in sel_stride:
                ret = get_out(conv_fpn_feat, 'face', stride, min_stride, config.FACE_LANDMARK, lr_mult=1.0,
                              shared_vars=shared_vars)
                ret_group += ret
                if config.HEAD_BOX:
                    assert not config.SHARE_WEIGHT_BBOX and not config.SHARE_WEIGHT_LANDMARK
                    shared_vars = [[None, None], [None, None], [None, None]]
                    ret = get_out(conv_fpn_feat, 'head', stride, False, lr_mult=0.5, shared_vars=shared_vars)
                    ret_group += ret
    else:

        for stride in config.ct_stride:
            if config.centernet_branch:
                if config.ct_mining and stride in config.ct_mining_stride:
                    ret_af, mined_labels, mined_bbox_weight = get_out_af(conv_fpn_feat, 'face', stride, min_stride,
                                                                         config.FACE_LANDMARK, lr_mult=1.0,
                                                                         shared_vars=shared_vars, gt_boxes=gt_boxes,
                                                                         gt_landmarks=gt_landmarks)

                else:
                    ret_af = get_out_af(conv_fpn_feat, 'face', stride, min_stride, config.FACE_LANDMARK,
                                        lr_mult=1.0,
                                        shared_vars=shared_vars, gt_boxes=gt_boxes,
                                        gt_landmarks=gt_landmarks)

                if config.ct_head:
                    shared_vars_tmp = []
                    for i in range(15):
                        shared_vars_tmp.append([None, None])
                    # ret = get_out(conv_fpn_feat, 'head', stride, False, lr_mult=0.5, shared_vars = shared_vars)
                    ret = get_out_af_head(conv_fpn_feat, 'head', stride, min_stride, False,lr_mult=1,
                                  shared_vars=shared_vars_tmp)  # jiashuz changed in 3.17
                    ret_group += ret

                ret_group += ret_af
        for stride in config.RPN_FEAT_STRIDE:
          if not config.ct_reproduce:
              if not config.ct_mining:
                ret = get_out(conv_fpn_feat, 'face', stride, min_stride, config.FACE_LANDMARK, lr_mult=1.0, shared_vars = shared_vars, gt_boxes=gt_boxes, gt_landmarks=gt_landmarks)
              else:
                  ret = get_out(conv_fpn_feat, 'face', stride, min_stride, config.FACE_LANDMARK, lr_mult=1.0,
                                shared_vars=shared_vars, gt_boxes=gt_boxes, gt_landmarks=gt_landmarks, mined_labels=mined_labels, mined_bbox_weight= mined_bbox_weight)
              ret_group += ret
              if config.HEAD_BOX:
                # assert not config.SHARE_WEIGHT_BBOX and not config.SHARE_WEIGHT_LANDMARK
                # shared_vars_tmp = [ [None, None], [None, None], [None, None] ]
                shared_vars_tmp = []
                for i in range(15):
                    shared_vars_tmp.append([None,None])
                # ret = get_out(conv_fpn_feat, 'head', stride, False, lr_mult=0.5, shared_vars = shared_vars)
                ret = get_out(conv_fpn_feat, 'head', stride, False, lr_mult=1, shared_vars=shared_vars_tmp) # jiashuz changed in 3.17
                ret_group += ret


    return mx.sym.Group(ret_group)


