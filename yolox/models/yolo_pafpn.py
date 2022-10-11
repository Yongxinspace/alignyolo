#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, BaseConv_t, CSPLayer, CSPLayer_t, DWConv, DWConv_t, SKConv, mam, SKNet, MAM, MDNet


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_features_t=("dark3t", "dark4t", "dark5t"),
        in_channels=[256, 512, 1024],  depthwise=False, act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        # self.backbone_t = CSPDarknet_t(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_features_t = in_features_t
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        Conv_t = DWConv_t if depthwise else BaseConv_t

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample_t = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.lateral_conv0_t = BaseConv_t(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.C3_p4_t = CSPLayer_t(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.reduce_conv1_t = BaseConv_t(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.C3_p3_t = CSPLayer_t(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.bu_conv2_t = Conv_t(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.C3_n3_t = CSPLayer_t(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.bu_conv1_t = Conv_t(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.C3_n4_t = CSPLayer_t(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # self.mam1 = mam(in_ch=128, ou_ch=64)
        # self.mam2 = mam(in_ch=256, ou_ch=128)
        # self.mam3 = mam(in_ch=512, ou_ch=256)
        self.M_2 = SKConv(in_channels=128, out_channels=128, stride=1, M=2, r=16, L=32)
        self.M_1 = SKConv(in_channels=256, out_channels=256, stride=1, M=2, r=16, L=32)
        self.M_0 = SKConv(in_channels=512, out_channels=512, stride=1, M=2, r=16, L=32)
        self.stn_c0 = MAM(inch=256, in_ch=128)
        self.stn_c1 = MAM(inch=512, in_ch=256)
        self.stn_c2 = MAM(inch=1024, in_ch=512)
        self.md1 = MDNet(kin=256,kout=128,qout=9)
        self.md2 = MDNet(kin=512, kout=256, qout=9)
        self.md3 = MDNet(kin=1024, kout=512, qout=9)
    def forward(self, input, input_t):
        """
        Args:
            inputs: input images.

        Returns:c
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        #print(input.shape)
        out_features, out_features_t = self.backbone(input, input_t)
        features = [out_features[f] for f in self.in_features]
        features_t = [out_features_t[f] for f in self.in_features_t]
        [x2, x1, x0] = features
        [x2_t, x1_t, x0_t] = features_t

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        fpn_out0_t = self.lateral_conv0_t(x0_t)  # 1024->512/32
        f_out0_t = self.upsample_t(fpn_out0_t)  # 512/16
        f_out0_t = torch.cat([f_out0_t, x1_t], 1)  # 512->1024/16
        f_out0_t = self.C3_p4_t(f_out0_t)  # 1024->512/16

        fpn_out1_t = self.reduce_conv1_t(f_out0_t)  # 512->256/16
        f_out1_t = self.upsample_t(fpn_out1_t)  # 256/8
        f_out1_t = torch.cat([f_out1_t, x2_t], 1)  # 256->512/8
        pan_out2_t = self.C3_p3_t(f_out1_t)  # 512->256/8

        # fu_out2 = self.mam1(pan_out2, pan_out2_t)
        pan_out2, pan_out2_t = self.stn_c0(pan_out2, pan_out2_t)
        pan_out2, pan_out2_t = self.md1(pan_out2, pan_out2_t)
        sk_2 = self.M_2(pan_out2, pan_out2_t)

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out1_t = self.bu_conv2_t(pan_out2_t)  # 256->256/16
        p_out1_t = torch.cat([p_out1_t, fpn_out1_t], 1)  # 256->512/16
        pan_out1_t = self.C3_n3_t(p_out1_t)  # 512->512/16

        # fu_out1 = self.mam2(pan_out1, pan_out1_t)
        pan_out1, pan_out1_t = self.stn_c1(pan_out1, pan_out1_t)
        pan_out1, pan_out1_t = self.md2(pan_out1, pan_out1_t)
        sk_1 = self.M_1(pan_out1, pan_out1_t)

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        p_out0_t = self.bu_conv1_t(pan_out1_t)  # 512->512/32
        p_out0_t = torch.cat([p_out0_t, fpn_out0_t], 1)  # 512->1024/32
        pan_out0_t = self.C3_n4_t(p_out0_t)  # 1024->1024/32

        # fu_out0 = self.mam3(pan_out0, pan_out0_t)
        pan_out0, pan_out0_t = self.stn_c2(pan_out0, pan_out0_t)
        pan_out0, pan_out0_t = self.md3(pan_out0, pan_out0_t)
        sk_0 = self.M_0(pan_out0, pan_out0_t)

        outputs = (sk_2, sk_1, sk_0)
        return outputs
