_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/nuimages.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=10, norm_cfg=dict(type='BN', requires_grad=True)),
    auxiliary_head=dict(num_classes=10, norm_cfg=dict(type='BN', requires_grad=True)))
# use cityscapes pre-trained models
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101b-d8_512x1024_80k_cityscapes/deeplabv3plus_r101b-d8_512x1024_80k_cityscapes_20201226_190843-9c3c93a4.pth'
evaluation = dict(interval=80000, metric='mIoU')