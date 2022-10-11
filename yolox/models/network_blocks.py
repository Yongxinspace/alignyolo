#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from functools import reduce
import numpy as np
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class BaseConv_t(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels, in_channels, ksize=ksize,
            stride=stride, groups=in_channels, act=act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1,
            stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class DWConv_t(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv_t(
            in_channels, in_channels, ksize=ksize,
            stride=stride, groups=in_channels, act=act
        )
        self.pconv = BaseConv_t(
            in_channels, out_channels, ksize=1,
            stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, in_channels, out_channels, shortcut=True,
        expansion=0.5, depthwise=False, act="silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class SPPBottleneck_t(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv_t(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv_t(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self, in_channels, out_channels, n=1,
        shortcut=True, expansion=0.5, depthwise=False, act="silu"
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class CSPLayer_t(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self, in_channels, out_channels, n=1,
        shortcut=True, expansion=0.5, depthwise=False, act="silu"
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        )
        return self.conv(x)

class Focus_t(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv_t(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        )
        return self.conv(x)

class SKNet(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKNet,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加
        return V


class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d
        self.M=M
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
        self.alpha = nn.Conv2d(in_channels, 1, 1, 1, 0)
        self.bata = nn.Conv2d(in_channels, 1, 1, 1, 0)
        self.dynamic_filter = nn.Conv2d(in_channels, 3 * 3 * in_channels, 3, 1, 1)
        self.dynamic_filter_t = nn.Conv2d(in_channels, 3 * 3 * in_channels, 3, 1, 1)
        self.fus3 = convblock(out_channels, in_channels, 1, 1, 0)

    def forward(self, input, input_t):
        output = []
        # affine_gt = self.alpha(input) * input_t + self.bata(input)
        # filter1 = self.dynamic_filter(input)
        # filter2 = self.dynamic_filter_t(affine_gt)
        # dm_t = kernel2d_conv(affine_gt, filter1, 3)
        # dm_r = kernel2d_conv_t(input, filter2, 3)
        # U = self.fus3(dm_t + input)
        batch_size = input.size(0)
        output.append(input)
        output.append(input_t)
        U = reduce(lambda x, y: x + y, output)
        # U = torch.add(input, input_t)# 逐元素相加生成 混合特征U
        s = self.global_pool(U)
        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b = list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b = list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V = list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加
        return V


def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
def convblock_t(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.RReLU(inplace=True)
    )


def kernel2d_conv_t(feat_in, kernel, ksize):
    """
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out
class MySTN(nn.Module):
        def __init__(self, in_ch, mode='Curve'):
            super(MySTN, self).__init__()

            self.mode = mode
            self.down_block_1 = nn.Sequential(
                convblock(in_ch, 128, 3, 2, 1),
                convblock(128, 128, 1, 1, 0)
            )
            self.down_block_2 = nn.Sequential(
                convblock(128, 128, 3, 2, 1),
                convblock(128, 128, 1, 1, 0)
            )
            if mode == 'Curve':
                self.up_blcok_1 = convblock(128, 128, 1, 1, 0)
                self.up_blcok_2 = convblock(128, 64, 1, 1, 0)
                self.wrap_filed = nn.Conv2d(64, 2, 3, 1, 1)
                self.wrap_filed.weight.data.normal_(mean=0.0, std=5e-4)
                self.wrap_filed.bias.data.zero_()
                self.wrap_grid = None
            elif mode == 'Affine':
                self.down_block_3 = nn.Sequential(
                    convblock(128, 128, 3, 2, 1),
                    convblock(128, 128, 1, 1, 0),
                )
                self.deta = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(128, 6, 1, 1, 0)
                )
                # Start with identity transformation
                self.deta[-1].weight.data.normal_(mean=0.0, std=5e-4)
                self.deta[-1].bias.data.zero_()
                self.affine_matrix = None
                self.wrap_grid = None

        def forward(self, in_):
            size = in_.shape[2:]
            n1 = self.down_block_1(in_)
            n2 = self.down_block_2(n1)

            if self.mode == "Curve":
                n2 = self.up_blcok_1(F.interpolate(n2, size=n1.shape[2:], mode='bilinear', align_corners=True))
                n2 = self.up_blcok_2(F.interpolate(n2, size=in_.shape[2:], mode='bilinear', align_corners=True))

                xx = torch.linspace(-1, 1, size[1]).view(1, -1).repeat(size[0], 1)
                yy = torch.linspace(-1, 1, size[0]).view(-1, 1).repeat(1, size[1])
                xx = xx.view(1, size[0], size[1])
                yy = yy.view(1, size[0], size[1])
                grid = torch.cat((xx, yy), 0).float16().unsqueeze(0).repeat(in_.shape[0], 1, 1, 1)
                grid = grid.clone().detach().requires_grad_(False)
                if in_.is_cuda:
                    grid = grid.cuda()

                filed_residal = self.wrap_filed(n2)
                self.wrap_grid = grid + filed_residal

            elif self.mode == "Affine":
                identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float16).requires_grad_(False)
                if in_.is_cuda:
                    identity_theta = identity_theta.cuda()

                n3 = self.down_block_3(n2)
                deta = self.deta(n3)
                bsize = deta.shape[0]
                self.affine_matrix = deta.view(bsize, -1) + identity_theta.unsqueeze(0).repeat(bsize, 1)
                # print(self.affine_matrix.view(-1, 2, 3))
                theta = self.affine_matrix.view(-1, 2, 3)
                # theta = 0.1 * F.softsign(theta)
                # theta[:, 0, 0] = 1
                # theta[:, 0, 1] = 0
                # theta[:, 1, 1] = 1
                # theta[:, 1, 0] = 0
                # print(theta)
                self.wrap_grid = F.affine_grid(theta, in_.size(),
                                               align_corners=True).permute(0, 3, 1, 2)

        def wrap(self, x):
            if not x.shape[-1] == self.wrap_grid.shape[-1]:
                sampled_grid = F.interpolate(self.wrap_grid, size=x.shape[2:], mode='bilinear', align_corners=True)
                wrap_x = F.grid_sample(x, sampled_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',
                                       align_corners=True)
            else:
                wrap_x = F.grid_sample(x, self.wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',
                                       align_corners=True)
            return wrap_x

        def wrap_inverse(self, x):
            t1, t2 = self.affine_matrix.view(-1, 2, 3)[:, :, :2], self.affine_matrix.view(-1, 2, 3)[:, :, 2].unsqueeze(
                2)
            matrix_inverse = torch.cat((t1.inverse(), -t2), dim=2)
            sampled_grid = F.affine_grid(matrix_inverse, x.size(), align_corners=True)
            wrap_x = F.grid_sample(x, sampled_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            return wrap_x

class MAM(nn.Module):
        def __init__(self, inch, in_ch):
            super(MAM, self).__init__()
            self.stn = MySTN(inch, "Affine")
            self.alpha = nn.Conv2d(in_ch, 1, 1, 1, 0)
            self.bata = nn.Conv2d(in_ch, 1, 1, 1, 0)

        def forward(self, gr, gt):
            self.stn(torch.cat([gr, gt], dim=1))
            gt = self.stn.wrap(gt)
            affine_gt = self.alpha(gr) * gt + self.bata(gr)
            return gr, affine_gt

class mam(nn.Module):
    def __init__(self, in_ch, ou_ch):

        super(mam, self).__init__()
        # self.stn = MySTN(256, "Affine")

        self.fus1 = convblock(256, 64, 1, 1, 0)
        self.alpha = nn.Conv2d(in_ch, 1, 1, 1, 0)
        self.bata = nn.Conv2d(in_ch, 1, 1, 1, 0)
        self.fus2 = convblock(in_ch, ou_ch, 3, 1, 1)
        #
        self.dynamic_filter = nn.Conv2d(in_ch,3*3*in_ch,3,1,1)
        self.fus3 = convblock(in_ch, ou_ch, 1, 1, 0)
        self.combine = convblock(in_ch,in_ch,3,1,1)

    def forward(self, gr, gt):

        # self.stn(torch.cat([gr,gt],dim=1))
        # in1 = self.fus1(torch.cat([gr, self.stn.wrap(gt)],dim=1))

        affine_gt = self.alpha(gr)*gt + self.bata(gr)
        in2 = self.fus2(gr+affine_gt)
        filter = self.dynamic_filter(gr)
        in3 =  self.fus3(kernel2d_conv(gt,filter,3)+gr)
        return self.combine(torch.cat([in2,in3],dim=1))
def kernel2d_conv(feat_in, kernel, ksize):
    """
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out
class Bottleneckt(nn.Module):
    # Standard bottleneck
    def __init__(
        self, in_channels, out_channels, shortcut=True,
        expansion=0.5, depthwise=False, act="silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x, x_input):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x_input
        return y
class fdfm(nn.Module):
    def __init__(self, in_ch):

        super(fdfm, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Tanh()
        self.combine = Bottleneckt(in_ch, in_ch, shortcut=True,expansion=1.0, depthwise=False, act="silu")
        self.combine_t = Bottleneckt(in_ch, in_ch, shortcut=True,expansion=1.0, depthwise=False, act="silu")
        # self.alpha = nn.Conv2d(in_ch, 1, 1, 1, 0)
        # self.bata = nn.Conv2d(in_ch, 1, 1, 1, 0)
        # self.alpha_t = nn.Conv2d(in_ch, 1, 1, 1, 0)
        # self.bata_t = nn.Conv2d(in_ch, 1, 1, 1, 0)
        # self.fus2 = convblock(in_ch, ou_ch, 3, 1, 1)
        # self.dynamic_filter = nn.Conv2d(in_ch, 3 * 3 * in_ch, 3, 1, 1)
        # self.dynamic_filter_t = nn.Conv2d(in_ch, 3 * 3 * in_ch, 3, 1, 1)

    def forward(self, x, x_lwir):
        subtracted = torch.sub(x, x_lwir)
        subtracted_weight = self.global_pool(subtracted)
        excitation_weight = self.act(subtracted_weight)

        subtracted2 = torch.sub(x_lwir, x)
        subtracted_weight2 = self.global_pool(subtracted2)
        excitation_weight2 = self.act(subtracted_weight2)

        x_weight = torch.multiply(x, excitation_weight)
        x_lwir_weight = torch.multiply(x_lwir, excitation_weight2)

        x_mix = torch.add(x_lwir_weight, x)
        x_lwir_mix = torch.add(x_lwir, x_weight)
        f_x = self.combine(x_mix, x)
        f_t = self.combine_t(x_lwir_mix, x_lwir)
        # affine_gt = self.alpha(f_x) * f_t + self.bata(f_x)
        # affine_gr = self.alpha_t(f_t) * f_x + self.bata_t(f_t)
        # filter1 = self.dynamic_filter(affine_gr)
        # filter2 = self.dynamic_filter_t(affine_gt)
        # dm_t = kernel2d_conv(affine_gt, filter1, 3)
        # dm_r = kernel2d_conv_t(affine_gr, filter2, 3)
        return  f_x, f_t

class Dynamic(nn.Module):
    def __init__(self, in_ch):

        super(Dynamic, self).__init__()

        self.dynamic_filter = nn.Conv2d(in_ch, 3 * 3 * in_ch, 3, 1, 1)
        # self.dynamic_filter_t = nn.Conv2d(in_ch, 3 * 3 * in_ch, 3, 1, 1)
        self.fus3 = convblock(in_ch, in_ch, 1, 1, 0)
        # self.combine = convblock(in_ch,in_ch,3,1,1)

    def forward(self, gr, gt):
        filter1 = self.dynamic_filter(gr)
        # filter2 = self.dynamic_filter_t(gt)
        dm_t = kernel2d_conv(gt, filter1, 3)
        # dm_r = kernel2d_conv_t(gr, filter2, 3)
        fn = self.fus3(dm_t+gr)
        return fn
class MDNet(nn.Module):
    def __init__(self, kin, kout, qout):
        super(MDNet, self).__init__()
        # self.K = K
        self.channel_attention = ChannelAttention(kin)
        self.spatial_attention = SpatialAttention()
        self.conv1x1_Tk = nn.Conv2d(kin, kout, 1, 1)
        self.conv1x1_Tq = nn.Conv2d(kin, qout, 1, 1)
        self.conv1x1_Vk = nn.Conv2d(kin, kout, 1, 1)
        self.conv1x1_Vq = nn.Conv2d(kin, qout, 1, 1)
        self.conv1x1_Tk = self.conv1x1_Tk.cuda()
        self.conv1x1_Tq = self.conv1x1_Tq.cuda()
        self.conv1x1_Vk = self.conv1x1_Vk.cuda()
        self.conv1x1_Vq = self.conv1x1_Vq.cuda()
        self.sigmoid = nn.Sigmoid()
        self.combine = convblock(kin, kout, 1, 1, 0)

    def forward(self, x_v, x_i):
        rgbt_feats = torch.cat((x_v, x_i), dim=1)  ## torch.Size([1, 192, 62, 91])

        # pdb.set_trace()

        rgbt_feats = self.channel_attention(rgbt_feats) * rgbt_feats
        rgbt_feats = self.spatial_attention(rgbt_feats) * rgbt_feats

        Tk_feats = self.conv1x1_Tk(rgbt_feats)  ## torch.Size([1, 96, 117, 71])
        Tq_feats = self.conv1x1_Tq(rgbt_feats)  ## torch.Size([1, 9, 117, 71])
        Vk_feats = self.conv1x1_Vk(rgbt_feats)
        Vq_feats = self.conv1x1_Vq(rgbt_feats)

        # pdb.set_trace()

        Tk_feats = torch.squeeze(Tk_feats, dim=0)
        # aaa = torch.squeeze(Tk_feats, dim=1)
        Tk_feats = Tk_feats.view(-1, Tk_feats.shape[1] * Tk_feats.shape[2])  ## torch.Size([96, 4150])

        Tq_feats = torch.squeeze(Tq_feats, dim=0)
        Tq_feats = Tq_feats.view(-1, Tq_feats.shape[1] * Tq_feats.shape[2])

        Vk_feats = torch.squeeze(Vk_feats, dim=0)
        Vk_feats = Vk_feats.view(-1, Vk_feats.shape[1] * Vk_feats.shape[2])

        Vq_feats = torch.squeeze(Vq_feats, dim=0)
        Vq_feats = Vq_feats.view(-1, Vq_feats.shape[1] * Vq_feats.shape[2])
        # bbb = torch.transpose(Tq_feats, 1, 0)

        #### T_output.shape: torch.Size([96, 9])
        T_output = torch.matmul(Tk_feats, torch.transpose(Tq_feats, 1, 0))
        V_output = torch.matmul(Vk_feats, torch.transpose(Vq_feats, 1, 0))

        # pdb.set_trace()
        T_filters = torch.reshape(T_output, (1, T_output.shape[0], 3, 3))  ## (96, 3, 3)
        V_filters = torch.reshape(V_output, (1, V_output.shape[0], 3, 3))  ## (96, 3, 3)

        # pdb.set_trace()

        adaptive_conv_T = AdaptiveConv2d(x_i.size(1), x_i.size(1), 3, padding=1, groups=x_i.size(1), bias=False)
        adaptive_conv_V = AdaptiveConv2d(x_v.size(1), x_v.size(1), 3, padding=1, groups=x_v.size(1), bias=False)

        dynamic_T_feats = adaptive_conv_T(x_v, T_filters)
        dynamic_V_feats = adaptive_conv_V(x_i, V_filters)

        dynamic_T_feats = self.sigmoid(dynamic_T_feats)
        dynamic_V_feats = self.sigmoid(dynamic_V_feats)

        x_v = x_v + dynamic_V_feats
        x_i = x_i + dynamic_T_feats

        # fuse_x_v_i = torch.cat((x_v, x_i), dim=1)
        # fuse_x_v_i = self.combine(fuse_x_v_i)
        # fuse_x_v_i = torch.add(x_v, x_i)

        return x_v, x_i

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class AdaptiveConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(AdaptiveConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)

    def forward(self, input, dynamic_weight):
        # Get batch num
        batch_num = input.size(0)

        # Reshape input tensor from size (N, C, H, W) to (1, N*C, H, W)
        input = input.view(1, -1, input.size(2), input.size(3))

        # Reshape dynamic_weight tensor from size (N, C, H, W) to (1, N*C, H, W)
        dynamic_weight = dynamic_weight.view(-1, 1, dynamic_weight.size(2), dynamic_weight.size(3))

        # Do convolution
        conv_rlt = F.conv2d(input, dynamic_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Reshape conv_rlt tensor from (1, N*C, H, W) to (N, C, H, W)
        conv_rlt = conv_rlt.view(batch_num, -1, conv_rlt.size(2), conv_rlt.size(3))

        return conv_rlt




# class cfcm(nn.Module):
#     def __init__(self, in_ch, ou_ch):
#
#         super(cfcm, self).__init__()
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.sig = nn.Sigmoid()
#         self.combine = convblock(in_ch,ou_ch,3,1,1)
#         self.combine_t = convblock_t(in_ch, ou_ch, 3, 1, 1)
#
#     def forward(self, x, x_t):
#         subtracted = torch.sub(x, x_t)
#         subtracted_weight = self.global_pool(subtracted)
#         excitation_weight = self.sig(subtracted_weight)
#
#         subtracted2 = torch.sub(x_t, x)
#         subtracted_weight2 = self.global_pool(subtracted2)
#         excitation_weight2 = self.sig(subtracted_weight2)
#
#         x_weight = torch.multiply(x, excitation_weight)
#         x_lwir_weight = torch.multiply(x_t, excitation_weight2)
#
#
#         x_mix = torch.cat([x_lwir_weight, x], dim=1)
#         x_lwir_mix = torch.cat([x_weight, x_t], dim=1)
#         x_mix = self.combine(x_mix)
#         x_lwir_mix = self.combine_t(x_lwir_mix)
#         x_r = self.sig(self.global_pool(x))
#         x_lwir = self.sig(self.global_pool(x_t))
#         final_x = torch.add(x_mix, x_r)
#         final_t = torch.add(x_lwir_mix, x_lwir)
#         return final_x, final_t

