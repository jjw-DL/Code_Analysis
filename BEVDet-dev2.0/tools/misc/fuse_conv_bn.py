# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmcv.runner import save_checkpoint
from torch import nn as nn

from mmdet3d.apis import init_model


def fuse_conv_bn(conv, bn):
    """During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures."""
    conv_w = conv.weight # 提取卷积的weight和bias eg:(160, 80, 3, 3)
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean) # eg:(160,)
    # 融合BN
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps) # 计算BN的系数 eg:(160,)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1])) # eg:(160, 80, 3, 3)
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias) # eg:(160,)
    return conv # 返回融合后的BN


def fuse_module(m):
    last_conv = None
    last_conv_name = None
    # 逐个module处理
    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv 只处理卷积后的BN层
                continue
            fused_conv = fuse_conv_bn(last_conv, child)
            m._modules[last_conv_name] = fused_conv # eg:last_conv_name:conv1 覆盖原始卷积
            # To reduce changes, set BN as Identity instead of deleting it.
            m._modules[name] = nn.Identity() # 将BN层设置为Identity，onnx会自动优化
            last_conv = None
        elif isinstance(child, nn.Conv2d): # 针对卷积直接赋值
            last_conv = child
            last_conv_name = name
        else:
            fuse_module(child) # 递归处理
    return m


def parse_args():
    parser = argparse.ArgumentParser(
        description='fuse Conv and BN layers in a model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('out', help='output path of the converted model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint)
    # fuse conv and bn layers of the model
    fused_model = fuse_module(model)
    save_checkpoint(fused_model, args.out)


if __name__ == '__main__':
    main()
