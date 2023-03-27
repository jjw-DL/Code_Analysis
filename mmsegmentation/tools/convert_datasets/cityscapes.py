# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
from cityscapesscripts.preparation.json2labelImg import json2labelImg

# 进程函数接受的是单个文件，内部逻辑也是对单个文件的处理
def convert_json_to_label(json_file):
    """
    将json文件转化为图片标签
    """
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png') # 拼接输出标签路径
    # 编码方式 "trainIds" : classes are encoded using the training IDs
    json2labelImg(json_file, label_file, 'trainIds')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds') # 功能描述
    parser.add_argument('cityscapes_path', help='cityscapes data path') # 数据集文件夹data/cityscapes
    parser.add_argument('--gt-dir', default='gtFine', type=str) # ground truth文件夹
    parser.add_argument('-o', '--out-dir', help='output path') # 输出文件夹
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process') # 进程数
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path # 如果没有定义输出文件夹则采用数据集文件夹
    mmcv.mkdir_or_exist(out_dir) # 创建输出文件夹

    gt_dir = osp.join(cityscapes_path, args.gt_dir) # 拼接gt文件夹

    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json', recursive=True): # 扫描gtFine下的.json文件
        poly_file = osp.join(gt_dir, poly) # 拼接json路径
        poly_files.append(poly_file) # 加入poly list
    if args.nproc > 1:
        mmcv.track_parallel_progress(convert_json_to_label, poly_files,
                                     args.nproc) # 转化json文件到lable
    else:
        mmcv.track_progress(convert_json_to_label, poly_files)

    # 对split分开处理
    split_names = ['train', 'val', 'test']

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(
                osp.join(gt_dir, split), '_polygons.json', recursive=True):
            filenames.append(poly.replace('_gtFine_polygons.json', '')) # 将json替换为空
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames) # 将文件写入磁盘


if __name__ == '__main__':
    main()
