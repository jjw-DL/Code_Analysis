# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet.models import NECKS


@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(self, 
                 in_channels, # 384+768
                 out_channels, # 512
                 scale_factor=4, # 2
                 input_feature_index = (0,2), # (0, 1)
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2, # None
                 lateral=None):
        super().__init__()
        self.input_feature_index = input_feature_index # (0, 1)
        self.extra_upsample = extra_upsample is not None # None
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True) # 上采样2倍
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False), # 1152-->512
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor,
                      kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample , mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral=  lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral,
                          kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        # (48, 384, 16, 44)和(48, 768, 8, 22)
        # img_bev_encoder_neck: (8, 128, 64, 64)和(8, 512, 16, 16)
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1) # 将底层特征图上采样2倍数 (48, 768, 16, 44) img_bev_encoder_neck:上采样4倍
        x1 = torch.cat([x2, x1], dim=1) # 将两个特征图拼接 (48, 1152, 16, 44)
        x = self.conv(x1) # 压缩到512 (48, 512, 16, 44) img_bev_encoder_neck:(8, 512, 64, 64)
        if self.extra_upsample:
            x = self.up2(x) # (8, 256, 128, 128)
        return x # (48, 512, 16, 44)  img_bev_encoder_neck:(8, 256, 128, 128)



