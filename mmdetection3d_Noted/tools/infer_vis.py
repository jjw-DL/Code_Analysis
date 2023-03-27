import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

# --------------------------------------------------
# 1.解析命令行参数
# --------------------------------------------------
parser = argparse.ArgumentParser(
        description='MMDet test (and visual) a model')
parser.add_argument('config', help='test config file path') # 指定配置文件路径
parser.add_argument('checkpoint', help='checkpoint file') # 指定权重路径
args = parser.parse_args() # 解析命令行参数

# --------------------------------------------------
# 2.构造Config对象, 设置batch size
# --------------------------------------------------
cfg = Config.fromfile(args.config) # 根据配置文件路径构造Config对象
samples_per_gpu = 1 # 设置batch size为1

# --------------------------------------------------
# 3.build the dataloader
# --------------------------------------------------
dataset = build_dataset(cfg.data.test) # 构建test数据集
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False) # 构建test data_loader

# --------------------------------------------------
# 4.build the model and load checkpoint
# --------------------------------------------------
cfg.model.train_cfg = None # 将train cfg设置为None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')) # 根据配置文件构建模型
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu') # 加载权重文件

# --------------------------------------------------
# 4.模型 infer
# --------------------------------------------------
model = MMDataParallel(model, device_ids=[0])
model.eval() # 将模型设置为eval模式
results = [] # 初始化results
for i, data in enumerate(data_loader):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data) # 通过模型获得预测结果
        results.extend(result) # 将预测结果加入results

# --------------------------------------------------
# 5.结果可视化 根据data和result可视化
# --------------------------------------------------