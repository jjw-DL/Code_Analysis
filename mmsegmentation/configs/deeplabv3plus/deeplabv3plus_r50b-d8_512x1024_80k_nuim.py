_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/nuimages.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(_delete_=True, type='BN', requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(num_classes=10, norm_cfg=dict(type='BN', requires_grad=True)),
    auxiliary_head=dict(num_classes=10, norm_cfg=dict(type='BN', requires_grad=True)))
# use cityscapes pre-trained models
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth'
evaluation = dict(interval=80000, metric='mIoU')