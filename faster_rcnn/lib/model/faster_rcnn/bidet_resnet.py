from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.utils.config import cfg
from lib.model.faster_rcnn.faster_rcnn import _fasterRCNN_BiDet
import lib.model.faster_rcnn.binary_utils as b_utils

import torch
import torch.nn as nn
import math
import pdb


def binary_conv1x1(in_planes, out_planes, stride=1, **kwargs):
    """3x3 convolution with padding"""
    return b_utils.BinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                  padding=0, bias=False,
                                  **kwargs)


def binary_conv3x3(in_planes, out_planes, stride=1, **kwargs):
    """3x3 convolution with padding"""
    return b_utils.BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False,
                                  **kwargs)


def binary_block3x3(in_planes, out_planes, stride=1, **kwargs):
    """3x3 convolution with padding"""
    return b_utils.BinBlock(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=1, bias=False,
                            **kwargs)


def binary_block5x5(in_planes, out_planes, stride=1, **kwargs):
    """3x3 convolution with padding"""
    return b_utils.BinBlock(in_planes, out_planes, kernel_size=5, stride=stride,
                            padding=2, bias=False,
                            **kwargs)


def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1, **kwargs):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, **kwargs)


class BinBasicBlock(nn.Module):
    """
    Shortcut between every two adjacent convs
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BinBasicBlock, self).__init__()
        if downsample is not None:
            res_func1 = downsample
        else:
            res_func1 = b_utils.myid
        res_func2 = b_utils.myid

        self.conv1 = binary_block3x3(inplanes, planes, stride, res_func=res_func1, **kwargs)
        self.conv2 = binary_block3x3(planes, planes, res_func=res_func2, **kwargs)

        self.stride = stride

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)

        return out


class BiDetResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, channels=(64, 128, 256, 512), **kwargs):
        super(BiDetResNet, self).__init__()
        self.inplanes = channels[0]
        first_inplanes = self.inplanes
        self.conv1 = nn.Conv2d(3, first_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(first_inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, first_inplanes, layers[0], **kwargs)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, **kwargs)
        if len(channels) == 4:
            self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, **kwargs)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(channels[-1] * block.expansion, num_classes, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, b_utils.BinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, b_utils.BinarizeLinear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv = nn.Conv2d
            ds_out_planes = planes * block.expansion
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride=stride, ceil_mode=True),
                conv(self.inplanes, ds_out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(ds_out_planes)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if hasattr(self, 'layer4'):
            x = self.layer4(x)

        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)

        return self.log_softmax(x)


def bidetnet18(**kwargs):
    model = BiDetResNet(BinBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def bidetnet34(**kwargs):
    model = BiDetResNet(BinBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


class bidet_resnet(_fasterRCNN_BiDet):

    def __init__(self, classes, num_layers=18, class_agnostic=False, model_path=None,
                 fix_real_conv=True, fix_base_bn=True, fix_top_bn=True, nms_threshold=0.01, sample_sigma=0.001,
                 rpn_prior_weight=0.2, rpn_reg_weight=0.1, head_prior_weight=0.2, head_reg_weight=0.1):
        # assume that base net can only be bireal18 or bireal34
        self.depth = num_layers
        self.model_path = model_path
        self.dout_base_model = 256
        self.pooled_feat_size = 512
        self.class_agnostic = class_agnostic
        self.fix_real_conv = fix_real_conv
        self.fix_base_bn = fix_base_bn
        self.fix_top_bn = fix_top_bn

        _fasterRCNN_BiDet.__init__(self, classes, class_agnostic, sample_sigma=sample_sigma,
                                   nms_threshold=nms_threshold,
                                   rpn_prior_weight=rpn_prior_weight, rpn_reg_weight=rpn_reg_weight,
                                   head_prior_weight=head_prior_weight, head_reg_weight=head_reg_weight)

    def _init_modules(self):

        if self.depth == 18:
            resnet = bidetnet18()
        elif self.depth == 34:
            resnet = bidetnet34()
        else:
            exit(-1)

        if self.model_path is not None:
            print("Loading pretrained weights from %s" % self.model_path)
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict(state_dict, strict=True)

        # Build resnet
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool,
                                       resnet.layer1, resnet.layer2, resnet.layer3)

        self.RCNN_top = nn.Sequential(resnet.layer4)

        self.RCNN_cls_score = nn.Linear(self.pooled_feat_size, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(self.pooled_feat_size, 8)
        else:
            self.RCNN_bbox_pred = nn.Linear(self.pooled_feat_size, 8 * self.n_classes)

        # Fix blocks
        if self.fix_real_conv:
            print("fix base net conv1 and bn1")
            for p in self.RCNN_base[0].parameters(): p.requires_grad = False
            for p in self.RCNN_base[1].parameters(): p.requires_grad = False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            print("fix base net layer3")
            for p in self.RCNN_base[5].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            print("fix base net layer2")
            for p in self.RCNN_base[4].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            print("fix base net layer1")
            for p in self.RCNN_base[3].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        if self.fix_base_bn:
            print("fix rcnn base bn")
            self.RCNN_base.apply(set_bn_fix)
        if self.fix_top_bn:
            print("fix rcnn top bn")
            self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            # base[0] and base[1] are in eval mode
            self.RCNN_base.eval()
            assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
            if cfg.RESNET.FIXED_BLOCKS == 3:
                # fix base[0], [1], [3], [4], [5]
                pass
            elif cfg.RESNET.FIXED_BLOCKS == 2:
                # fix base[0], [1], [3], [4]
                self.RCNN_base[5].train()
            elif cfg.RESNET.FIXED_BLOCKS == 1:
                # fix base[0], [1], [3]
                self.RCNN_base[5].train()
                self.RCNN_base[4].train()
            elif cfg.RESNET.FIXED_BLOCKS == 0:
                # fix base[0], [1]
                self.RCNN_base[5].train()
                self.RCNN_base[4].train()
                self.RCNN_base[3].train()

            if not self.fix_real_conv:
                self.RCNN_base[0].train()
                self.RCNN_base[1].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            if self.fix_base_bn:
                self.RCNN_base.apply(set_bn_eval)
            if self.fix_top_bn:
                self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7
