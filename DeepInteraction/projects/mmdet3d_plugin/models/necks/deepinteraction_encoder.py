import torch
from mmcv.cnn import build_conv_layer
from torch import nn
from mmdet3d.models.builder import NECKS
from projects.mmdet3d_plugin.models.utils.encoder_utils import MMRI_I2P, LocalContextAttentionBlock, ConvBNReLU, MMRI_P2I
import pdb

class DeepInteractionEncoderLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(DeepInteractionEncoderLayer, self ).__init__()
        # lidar2image
        self.I2P_block = MMRI_I2P(hidden_channel, hidden_channel, 0.1) # 128, 128, 0.1
        self.P_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9) # 128, 128, 9
        self.P_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None) # 1维度卷积
        self.P_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        # image2lidar
        self.P2I_block = MMRI_P2I(hidden_channel, hidden_channel, 9)
        self.I_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.I_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.I_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        
    def forward(self, img_feat, lidar_feat, img_metas, pts_metas):
        batch_size = lidar_feat.shape[0] # 1
        BN, I_C, I_H, I_W = img_feat.shape # 6, 128, 112, 200

        I2P_feat = self.I2P_block(lidar_feat, img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas, pts_metas) # (1, 128, 180, 180)
        P2P_feat = self.P_IML(lidar_feat, lidar_feat) # (1, 128, 180, 180)
        P_Aug_feat = self.P_out_proj(torch.cat((I2P_feat, P2P_feat),dim=1)) # (1, 128, 180, 180)
        new_lidar_feat = self.P_integration(torch.cat((P_Aug_feat, lidar_feat),dim=1)) # (1, 128, 180, 180)

        P2I_feat = self.P2I_block(lidar_feat, img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas, pts_metas) # (1, 6, 128, 112, 200)
        I2I_feat = self.I_IML(img_feat, img_feat) # (6, 128, 112, 200)
        I_Aug_feat = self.I_out_proj(torch.cat((P2I_feat.view(BN, -1, I_H, I_W), I2I_feat),dim=1)) # (6, 128, 112, 200)
        new_img_feat = self.I_integration(torch.cat((I_Aug_feat, img_feat),dim=1)) # (6, 128, 112, 200)

        return new_img_feat, new_lidar_feat # (6, 128, 112, 200)和(1, 128, 180, 180)

@NECKS.register_module()
class DeepInteractionEncoder(nn.Module):
    def __init__(self,
                num_layers=2,
                in_channels_img=64,
                in_channels_pts=128 * 3,
                hidden_channel=128,
                bn_momentum=0.1,
                bias='auto',
                ):
        super(DeepInteractionEncoder, self).__init__()

        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts, # 512
            hidden_channel, # 128
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img, # 256
            hidden_channel, # 128
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.num_layers = num_layers # 2
        self.fusion_blocks = nn.ModuleList()
        # 构造融合层
        for i in range(self.num_layers):
            self.fusion_blocks.append(DeepInteractionEncoderLayer(hidden_channel))

        self.bn_momentum = bn_momentum # 0.1
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, img_feats, pts_feats, img_metas, pts_metas):
        # img_feats: (6, 256, 112, 200)
        # pts_feats: (1, 512, 180, 180)
        # img_metas: 元信息
        # pts_metas: pillar信息
        new_img_feat = self.shared_conv_img(img_feats) # (6, 128, 112，200)
        new_pts_feat = self.shared_conv_pts(pts_feats) # (1, 128, 180，180)
        pts_feat_conv = new_pts_feat.clone() # (1, 128, 180，180)
        # 交互融合
        for i in range(self.num_layers):
            new_img_feat, new_pts_feat = self.fusion_blocks[i](new_img_feat, new_pts_feat, img_metas, pts_metas) # (6, 128, 112, 200)和(1, 128, 180, 180)   
        return new_img_feat, [pts_feat_conv, new_pts_feat] # (6, 128, 112，200)和[(1, 128, 180，180), (1, 128, 180，180)]