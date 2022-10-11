#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import torchvision
from functools import reduce
import numpy as np
from .network_blocks import BaseConv, BaseConv_t, CSPLayer, CSPLayer_t, DWConv, DWConv_t, Focus, Focus_t, ResLayer, SPPBottleneck, SPPBottleneck_t, MAM, mam, SKNet, SKConv, Dynamic, fdfm


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self, depth, in_channels=3, stem_out_channels=32, out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[2], stride=2))
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)]
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu"
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m


    def forward(self, x):
        outputs = {}

        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):

    def __init__(
        self, dep_mul, wid_mul,
        out_features_t=("dark3t", "dark4t", "dark5t"),
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False, act="silu",
    ):
        super().__init__()
        assert out_features,  "please provide output features of Darknet"
        assert out_features_t, "please provide output_t features of Darknet"
        self.out_features = out_features
        self.out_features_t = out_features_t
        # self.stn_c0 = MAM(inch=64)
        # self.stn_c1 = MAM(inch=128)
        # self.stn_c2 = MAM(inch=256)
        # self.stn_c3 = MAM(inch=512)
        # self.mam1 = mam(in_ch=128, ou_ch=64)
        # self.mam2 = mam(in_ch=256, ou_ch=128)
        # self.mam3 = mam(in_ch=512, ou_ch=256)
        # self.sk1 = SKConv(in_channels=128, out_channels=128, stride=1, M=2, r=16, L=32)
        # self.sk2 = SKConv(in_channels=256, out_channels=256, stride=1, M=2, r=16, L=32)
        # self.sk3 = SKConv(in_channels=512, out_channels=512, stride=1, M=2, r=16, L=32)
        self.cf0 = fdfm(in_ch=32)
        self.cf1 = fdfm(in_ch=64)
        self.cf2 = fdfm(in_ch=128)
        self.cf3 = fdfm(in_ch=256)
        # self.dy1 = Dynamic(in_ch=128)
        # self.dy2 = Dynamic(in_ch=256)
        # self.dy3 = Dynamic(in_ch=512)

        Conv_t = DWConv_t if depthwise else BaseConv_t
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem_t = Focus_t(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2_t = nn.Sequential(
            Conv_t(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer_t(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, act=act
            ),
        )

        # dark3
        self.dark3_t = nn.Sequential(
            Conv_t(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer_t(
                base_channels * 4, base_channels * 4,
                n=base_depth * 2, depthwise=depthwise, act=act,
            ),
        )

        # dark4
        self.dark4_t = nn.Sequential(
            Conv_t(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer_t(
                base_channels * 8, base_channels * 8,
                n=base_depth * 2, depthwise=depthwise, act=act,
            ),
        )

        # dark5

        self.dark5_t = nn.Sequential(
            Conv_t(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck_t(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer_t(
                base_channels * 16, base_channels * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, act=act,
            ),
        )
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, act=act
            ),
        )
        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 2, depthwise=depthwise, act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 2, depthwise=depthwise, act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, act=act,
            ),
        )
        self.localization = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=9),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=4),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)


        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 7 * 7, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))

    def DM_aware_fusion(self, x, x_lwir):
        # self.subtract_feature = reduce(lambda x: torch.sub(x[0], x[1]))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        subtracted = torch.sub(x, x_lwir)
        subtracted_weight = self.global_pool(subtracted)
        excitation_weight = torch.tanh(subtracted_weight)

        subtracted2 = torch.sub(x_lwir, x)
        subtracted_weight2 = self.global_pool(subtracted2)
        excitation_weight2 = torch.tanh(subtracted_weight2)

        x_weight = torch.multiply(x, excitation_weight)
        x_lwir_weight = torch.multiply(x_lwir, excitation_weight2)

        x_mix = torch.add(x_lwir_weight, x)
        x_lwir_mix = torch.add(x_lwir, x_weight)
        return x_mix, x_lwir_mix
    def stn(self,x):
        #提取输入图像中的特征
        # print('output shape0:\t', x.shape)
        xs = self.localization(x)
        # print('output shape1:\t', xs.shape)
        xs = xs.view(-1,10*7*7)
        # print('output shape2:\t', xs.shape)
        #回归theta参数
        theta = self.fc_loc(xs)
        # theta = torch.clamp(theta, -1, 1, out=None)
        theta = F.softsign(theta)
        theta = 0.2 * theta
        # print('output shape3:\t', theta.shape)
        theta = theta.view(-1,2,2)

        tx = (theta[:, 0, 0] + theta[:, 1, 0]) / 2
        ty = (theta[:, 0, 1] + theta[:, 1, 1]) / 2

        pad_func = nn.ConstantPad1d((0, 1), 0)
        theta = pad_func(theta)
        theta[:, 0, 0] = 1
        theta[:, 0, 1] = 0
        theta[:, 1, 1] = 1
        theta[:, 1, 0] = 0
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        print(theta)
        # torch.save(theta, "././theta2.pt")
        # print('output shape4:\t', theta.shape)
        #x = x.view(16,3)
        #利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta,x.size())
        #根据输入图片计算变换后图片位置填充的像素值
        x = F.grid_sample(x,grid)

        return x

    def tensor_to_np(self,tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        return img

    def convert_image_np(self,inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp


    def forward(self, x, x_t):
        outputs = {}
        outputs_t = {}
        # aa = x.cuda().data.cpu()
        # grid = aa[0]
        # grid = self.convert_image_np(torchvision.utils.make_grid(grid))
        # plt.figure()
        # plt.imshow(grid)
        # plt.savefig('/home/cnu228/Documents/cyx/YOLOX-train-your-data/result_cache/stn_out/result8.jpg')
        # x_t = self.stn(x_t)
        # x, x_t = self.stn_c(x, x_t)
        # grid = self.convert_image_np(torchvision.utils.make_grid(x_t.cpu()))
        # aa1 = x_t.cuda().data.cpu()
        # grid1 = aa1[0]
        # grid1 = self.convert_image_np(torchvision.utils.make_grid(grid1))
        # plt.figure()
        # plt.imshow(grid1)
        # plt.savefig('/home/cnu228/Documents/cyx/YOLOX-train-your-data/result_cache/stn_out/result8_s.jpg')
        x = self.stem(x)
        x_t = self.stem_t(x_t)
        # x, x_t = self.stn_c0(x, x_t)
        x, x_t = self.cf0(x, x_t)
        # x, x_t = self.DM_aware_fusion(x, x_t)
        outputs["stem"] = x
        outputs_t["stemt"] = x_t
        x = self.dark2(x)
        x_t = self.dark2_t(x_t)
        # x, x_t = self.stn_c1(x, x_t)
        x, x_t = self.cf1(x, x_t)
        outputs["dark2"] = x
        outputs_t["dark2t"] = x_t
        x = self.dark3(x)
        x_t = self.dark3_t(x_t)
        # x, x_t = self.stn_c2(x, x_t)
        x, x_t = self.cf2(x, x_t)
        # y1 = self.mam1(x, x_t)
        # fu1_1, fu1_2 = self.sk1(x, y1)
        # y1 = self.dy1(x, x_t)
        # fu1 = self.sk1(x, x_t)
        outputs["dark3"] = x
        outputs_t["dark3t"] = x_t
        x = self.dark4(x)
        x_t = self.dark4_t(x_t)
        # x, x_t = self.stn_c3(x, x_t)
        x, x_t = self.cf3(x, x_t)
        # y2 = self.mam2(x, x_t)
        # fu2_1, fu2_2 = self.sk2(x, y2)
        # y2 = self.dy2(x, x_t)
        # fu2 = self.sk2(x, x_t)
        outputs["dark4"] = x
        outputs_t["dark4t"] = x_t
        x = self.dark5(x)
        x_t = self.dark5_t(x_t)
        # y3 = self.mam3(x, x_t)
        # fu3_1, fu3_2 = self.sk3(x, y3)
        # y3 = self.dy3(x, x_t)
        # fu3 = self.sk3(x, x_t)
        outputs["dark5"] = x
        outputs_t["dark5t"] = x_t
        return {k: v for k, v in outputs.items() if k in self.out_features}, {i: j for i, j in outputs_t.items() if i in self.out_features_t}

'''
class CSPDarknet(nn.Module):

    def __init__(
        self, dep_mul, wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False, act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, act=act
            ),
        )
        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, act=act,
            ),
        )

    def convert_image_np(self,inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp

    def forward(self, x):
        outputs = {}
        # grid = self.convert_image_np(torchvision.utils.make_grid(x[0].cpu()))
        # aa = x.cuda().data.cpu()
        # grid = aa[0]
        # grid = self.convert_image_np(torchvision.utils.make_grid(grid))
        # plt.figure()
        # plt.imshow(grid)
        # plt.savefig('/home/cnu228/Documents/cyx/YOLOX-train-your-data/result_cache/stn_out/result4.jpg')
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
'''