"""
This file contains code for implementing bi-real net architectures.
Credit to Mr.Daquexian.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import typing


class SignTwoOrders(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input_wrt_output2 = torch.zeros_like(grad_output)
        ge0_lt1_mask = input.ge(0) & input.lt(1)
        grad_input_wrt_output2 = torch.where(ge0_lt1_mask, (2 - 2 * input), grad_input_wrt_output2)
        gen1_lt0_mask = input.ge(-1) & input.lt(0)
        grad_input_wrt_output2 = torch.where(gen1_lt0_mask, (2 + 2 * input), grad_input_wrt_output2)
        grad_input = grad_input_wrt_output2 * grad_output

        return grad_input


class SignSTE(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask = input.ge(-1) & input.le(1)
        grad_input = torch.where(mask, grad_output, torch.zeros_like(grad_output))
        return grad_input


class SignSTEWeight(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(grad_output.size())
        grad_input.copy_(grad_output)
        return grad_input


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_magnitude_aware=True, activation_value_aware=True,
                 **kwargs):
        super(BinarizeConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.weight_magnitude_aware = weight_magnitude_aware
        self.activation_value_aware = activation_value_aware

    def forward(self, input):
        if self.activation_value_aware:
            input = SignTwoOrders.apply(input)
        else:
            input = SignSTE.apply(input)

        subed_weight = self.weight
        if self.weight_magnitude_aware:
            self.weight_bin_tensor = subed_weight.abs(). \
                                         mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) \
                                     * SignSTEWeight.apply(subed_weight)
        else:
            self.weight_bin_tensor = SignSTEWeight.apply(subed_weight)
        self.weight_bin_tensor.requires_grad_()

        input = F.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]),
                      mode='constant', value=-1)
        out = F.conv2d(input, self.weight_bin_tensor, self.bias, self.stride, 0, self.dilation, self.groups)
        return out


class BinarizeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BinarizeLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        input = SignTwoOrders.apply(input)

        self.weight_bin_tensor = SignSTEWeight.apply(self.weight)
        self.weight_bin_tensor.requires_grad_()

        out = F.linear(input, self.weight_bin_tensor, self.bias)

        return out


def myid(x):
    return x


class BinBlock(nn.Module):
    def __init__(self, inplanes, planes, res_func=myid, **kwargs):
        super(BinBlock, self).__init__()
        self.conv = BinarizeConv2d(inplanes, planes, **kwargs)
        self.bn = nn.BatchNorm2d(planes)
        self.res_func = res_func

    def forward(self, input):
        if self.res_func is not None:
            residual = self.res_func(input)
        out = self.conv(input)
        out = self.bn(out)
        if self.res_func is not None:
            out += residual
        return out
