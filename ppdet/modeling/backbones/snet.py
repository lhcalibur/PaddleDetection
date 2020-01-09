# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from enum import unique, Enum

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.core.config.yaml_helpers import serializable
from ppdet.core.workspace import register

__all__ = ['SNet49']


@unique
class SNetArch(Enum):
    SNet49 = 'SNet49'
    SNet146 = 'SNet146'
    SNet535 = 'SNet535'


class SNet(object):
    def __init__(self,
                 backbone_arch,
                 norm_decay=0.,
                 conv_learning_rate=1.0,
                 ):
        if backbone_arch == SNetArch.SNet49:
            stages_repeats, stages_out_channels = [3, 7, 3], [24, 60, 120, 240, 512]
        elif backbone_arch == SNetArch.SNet146:
            stages_repeats, stages_out_channels = [3, 7, 3], [24, 132, 264, 528]
        elif backbone_arch == SNetArch.SNet535:
            stages_repeats, stages_out_channels = [3, 7, 3], [48, 248, 496, 992]
        else:
            raise RuntimeError(f"BackBone arch: {backbone_arch} not valid")
        self.backbone_arch = backbone_arch
        self.stages_repeats = stages_repeats
        self.stages_out_channels = stages_out_channels
        self.conv_learning_rate = conv_learning_rate
        self.norm_decay = norm_decay

    def _conv_norm(self,
                   input,
                   num_filters,
                   filter_size,
                   stride,
                   padding,
                   num_groups=1,
                   act='relu',
                   use_cudnn=True):
        parameter_attr = ParamAttr(
            learning_rate=self.conv_learning_rate,
            initializer=fluid.initializer.MSRA())
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)

        norm_decay = self.norm_decay
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(norm_decay))
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(norm_decay))
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr)

    def _shuffle_unit(self, inputs, out_channels, down_sampling):
        bottleneck_channels = out_channels // 2
        if not down_sampling:
            strides = (1, 1)

            c_hat, c = fluid.layers.split(inputs, num_or_sections=2, dim=1, name='split')
            inputs = c
        else:
            strides = (2, 2)

        x = self._conv_norm(inputs, bottleneck_channels, 1, 1, 'SAME')
        x = self._conv_norm(x, bottleneck_channels, 5, strides, 'SAME', num_groups=bottleneck_channels, act=None)
        x = self._conv_norm(x, bottleneck_channels, 1, 1, 'SAME')

        if not down_sampling:
            x = fluid.layers.concat([x, c_hat], axis=1)
        else:
            s2 = self._conv_norm(inputs, inputs.shape[1], 5, strides, 'SAME', num_groups=inputs.shape[1], act=None)
            s2 = self._conv_norm(s2, bottleneck_channels, 1, 1, 'SAME')
            x = fluid.layers.concat([x, s2], axis=1)
        x = fluid.layers.shuffle_channel(x, out_channels)
        return x

    @staticmethod
    def _cem(c_4, c_5):
        with fluid.name_scope("context_enhancement_module"):
            c_glb = fluid.layers.pool2d(c_5, pool_type='avg', global_pooling=True)
            c_glb = fluid.layers.reshape(c_glb, [-1, 1, 1, c_glb.shape[1]])
            c_4 = fluid.layers.conv2d(c_4, 245, 1, 1, padding='SAME')
            c_5 = fluid.layers.conv2d(c_5, 245, 1, 1, padding='SAME')
            c_glb = fluid.layers.conv2d(c_glb, 245, 1, 1, padding='SAME')

            c_5 = fluid.layers.resize_nearest(c_5, scale=2.0)
            x = c_4 + c_5 + c_glb
            return x

    @staticmethod
    def _rpn_backbone(x):
        out_channels = 256
        with fluid.name_scope("rpn_backbone"):
            x = fluid.layers.conv2d(x, x.shape[1], 5, 1, padding='SAME', groups=x.shape[1], act='relu')
            x = fluid.layers.conv2d(x, out_channels, 1, 1, padding='SAME', act='relu')
            return x

    def _sam(self, cem_features, rpn_features):
        cem_channels = cem_features.shape[1]
        with fluid.name_scope("spatial_attention_module"):
            attn = self._conv_norm(rpn_features, cem_channels, 1, 1, padding='SAME', act='sigmoid')
            sam_features = fluid.layers.elementwise_mul(cem_features, attn)
            return sam_features

    def __call__(self, input):
        output_channels = self.stages_out_channels[0]
        with fluid.name_scope('conv1'):
            x = self._conv_norm(input, output_channels, 3, 2, 'SAME')
        x = fluid.layers.pool2d(x, 3, 'max', 2, pool_padding='SAME', name='maxpool2d')
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, self.stages_repeats, self.stages_out_channels[1:]):
            with fluid.name_scope(name):
                x = self._shuffle_unit(x, output_channels, True)
                for i in range(repeats - 1):
                    x = self._shuffle_unit(x, output_channels, False)
                if name == 'stage3':
                    c_4 = x
                elif name == 'stage4':
                    c_5 = x
        if self.backbone_arch == SNetArch.SNet49:
            with fluid.name_scope("Conv5"):
                output_channels = self.stages_out_channels[-1]
                x = self._conv_norm(x, output_channels, 1, 1, 'SAME')
                c_5 = x
        cem_out = self._cem(c_4, c_5)
        rpn_feats = self._rpn_backbone(cem_out)
        sam_out = self._sam(cem_out, rpn_feats)
        return OrderedDict(rpn_feats=rpn_feats, sam=sam_out)


@register
@serializable
class SNet49(SNet):
    def __init__(self,
                 norm_decay=0.,
                 conv_learning_rate=1.0):
        super().__init__(
            SNetArch.SNet49,
            norm_decay,
            conv_learning_rate)
