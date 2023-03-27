# Copyright (c) Gorilla-Lab. All rights reserved.
import logging

import numpy as np
import torch
from torch import nn

from ..registry import DETECTORS
from .voxelnet import VoxelNet
from .. import builder

@DETECTORS.register_module
class OHS_Multiview(VoxelNet):
    def __init__(
        self,
        reader, # "VoxelFeatureExtractorV3"
        backbone, # "MultiViewBackbone"
        neck, # "RPNT"
        bbox_head, # "DeepMultiGroupOHSHeadClear_Decouple"
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(OHS_Multiview, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # front view neck配置
        cfg_rv_neck = dict(
            type=neck['type'], # RPNT
            layer_nums=[i for i in neck['layer_nums']], # [5, 5]
            ds_layer_strides=[i for i in neck['ds_layer_strides']], # [1, 2]
            ds_num_filters=[i for i in neck['ds_num_filters']], # [128, 256]
            us_layer_strides=[i for i in neck['us_layer_strides']], # [1, 2]
            us_num_filters=[i for i in neck['us_num_filters']], # [256, 128]
            num_input_features=neck["num_input_features"], # 640
            norm_cfg=neck['norm_cfg'],
            logger=logging.getLogger("RVRPN")
        )
        # 跨视角解耦注意力机制配置
        cfg_ca = dict(
            type='Cross_Attention_Decouple',
            bev_input_channel=384,
            rv_input_channel=384,
            embed_dim=384,
            num_heads=1,
            bev_size=(160, 160),
            bev_block_res=(40, 40),
            rv_size=(160, 12),
            rv_block_res=(40, 12),
            hidden_channels=512,
        )
        self.rv_neck = builder.build_neck(cfg_rv_neck)
        self.ca_neck = builder.build_neck(cfg_ca)
        if train_cfg:
            self.bbox_head.set_train_cfg(train_cfg['assigner'])
        else:
            self.bbox_head.set_train_cfg(test_cfg['assigner'])

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"]) # # (70563, 5) voxel内取平均值
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        (x_bev, x_rv) = x # (4, 640, 160, 160)和(4, 640, 160, 12)
        x_bev = self.neck(x_bev) # (4, 384, 160, 160) 这里的neck是single stage的neck
        x_rv = self.rv_neck(x_rv) # (4, 384, 160, 12)

        # cross atten 的关键部分
        return self.ca_neck((x_bev, x_rv))  # List[Tensor] 2个LIST (4, 384, 160, 160)和(4, 1600, 480)

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"] # (70563, 6, 5)
        coordinates = example["coordinates"] # (70563,4)
        num_points_in_voxel = example["num_points"] # (70563,)
        num_voxels = example["num_voxels"] # (4,)

        batch_size = len(num_voxels) # 4

        data = dict(
            features=voxels, # (70563, 6, 5)
            num_voxels=num_points_in_voxel, # (70563,)
            coors=coordinates, # (70563, 4)
            batch_size=batch_size, # 4
            input_shape=example["shape"][0], # (1280, 1280, 100)
        )
        xs, atten_maps = self.extract_feat(data) # List[Tensor] (4, 384, 160, 160)和(4, 1600, 480)

        preds = self.bbox_head(xs, return_loss)
        del data['features']
        del example["voxels"]
        # torch.cuda.empty_cache()

        if return_loss:
            return self.bbox_head.loss(example, preds, atten_map=atten_maps)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)