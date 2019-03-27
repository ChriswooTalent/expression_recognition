#!/usr/bin/env python3
# coding: utf-8

from __future__ import division

""" 
Creates a MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017

Modified By cleardusk
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['mobileDensenetv2CL_2', 'mobileDensenetv2CL_1', 'mobileDensenetv2CL_075', 'mobileDensenetv2CL_05', 'mobileDensenetv2CL_025']

class PoolingBlock(nn.Module):
    def __init__(self, ksize=3, stride=1, prelu=False):
        super(PoolingBlock, self).__init__()
        self.pooling = nn.MaxPool2d(ksize, stride=stride, padding=1)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pooling(x)
        out = self.relu(out)

        return out

class MergeBlock(nn.Module):
    def __init__(self, inplanes, temp_planes, planes, ksize1=3, ksize2=1, stride=1, prelu=False):
        super(MergeBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, int(temp_planes), kernel_size=ksize1, padding=1, stride=stride, groups=int(temp_planes),
                                 bias=False)
        self.bn_dw = nn.BatchNorm2d(int(temp_planes))
        self.conv_sep = nn.Conv2d(int(temp_planes), planes, kernel_size=ksize2, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out

class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
                                 bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)

        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class MobileDenseNetV2_CL(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileDenseNetV2_CL, self).__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.pooling1 = PoolingBlock(ksize=3, stride=4, prelu=prelu)
        self.merge1 = MergeBlock(256 * widen_factor, 128 * widen_factor, 128 * widen_factor, ksize1=3, ksize2=1,
                                  stride=4, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.merge2 = MergeBlock(512 * widen_factor, 256 * widen_factor, 256 * widen_factor, ksize1=3, ksize2=1,
                                  stride=2, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.merge3 = MergeBlock(512 * widen_factor, 256 * widen_factor, 256 * widen_factor, ksize1=3, ksize2=1,
                                  stride=2, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.merge4 = MergeBlock(512 * widen_factor, 256 * widen_factor, 256 * widen_factor, ksize1=3, ksize2=1,
                                  stride=2, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.reluip1 = nn.ReLU()

        self.Densefc = nn.Linear(int(1920 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        res_input1 = self.merge1(x)
        x = self.dw4_2(x)

        x = self.dw5_1(x)
        res_input2 = self.merge2(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        res_input3 = self.merge3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        res_input4 = self.merge4(x)
        x = self.dw5_6(x)

        res_output = torch.cat((res_input1, res_input2), 1)
        res_output = torch.cat((res_output, res_input3), 1)
        res_output = torch.cat((res_output, res_input4), 1)

        x = self.dw6(x)
        cat_output = torch.cat((res_output, x), 1)

        x = self.avgpool(cat_output)
        x = x.view(x.size(0), -1)
        ip1 = self.reluip1(x)
        ip2 = self.Densefc(ip1)

        return ip1, F.log_softmax(ip2, dim=1)


def mobileDensenetv2CL(widen_factor=1.0, num_classes=1000):
    """
    Construct MobileNet.
    widen_factor=1.0  for mobilenet_1
    widen_factor=0.75 for mobilenet_075
    widen_factor=0.5  for mobilenet_05
    widen_factor=0.25 for mobilenet_025
    """
    model = MobileDenseNetV2_CL(widen_factor=widen_factor, num_classes=num_classes)
    return model


def mobileDensenetv2CL_2(num_classes=62, input_channel=3):
    model = MobileDenseNet_CL(widen_factor=2.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobileDensenetv2CL_1(num_classes=62, input_channel=3):
    model = MobileDenseNetV2_CL(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobileDensenetv2CL_075(num_classes=62, input_channel=3):
    model = MobileDenseNetV2_CL(widen_factor=0.75, num_classes=num_classes, input_channel=input_channel)
    return model


def mobileDensenetv2CL_05(num_classes=62, input_channel=3):
    model = MobileDenseNetV2_CL(widen_factor=0.5, num_classes=num_classes, input_channel=input_channel)
    return model


def mobileDensenetv2CL_025(num_classes=62, input_channel=3):
    model = MobileDenseNetV2_CL(widen_factor=0.25, num_classes=num_classes, input_channel=input_channel)
    return model