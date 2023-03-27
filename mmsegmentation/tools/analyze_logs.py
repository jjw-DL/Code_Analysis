# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend) # 设置后端
    sns.set_style(args.style) # 设置绘制style
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend # 设置legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys # 设置评价指标

    # 逐个文件绘制
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        # 逐个评价指标绘制
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}') # 打印绘制第几个文件的第几个评价指标
            plot_epochs = []
            plot_iters = []
            plot_values = []
            # In some log files, iters number is not correct, `pre_iter` is
            # used to prevent generate wrong lines.
            pre_iter = -1
            # 逐个epoch添加值
            for epoch in epochs:
                epoch_logs = log_dict[epoch] # 获取该epoch的dict
                if metric not in epoch_logs.keys():
                    continue
                if metric in ['mIoU', 'mAcc', 'aAcc']:
                    plot_epochs.append(epoch) # 如评价指标存在，则添加epoch数 eg:1
                    plot_values.append(epoch_logs[metric][0]) # 同时添加该评价指标
                else:
                    for idx in range(len(epoch_logs[metric])): # 遍历该评价指标
                        if pre_iter > epoch_logs['iter'][idx]:
                            continue
                        pre_iter = epoch_logs['iter'][idx]
                        plot_iters.append(epoch_logs['iter'][idx]) # 在其中添加该iter数
                        plot_values.append(epoch_logs[metric][idx]) # 在其中添加该指标
            ax = plt.gca()
            label = legend[i * num_metrics + j]
            if metric in ['mIoU', 'mAcc', 'aAcc']: # 如果是这几个评价指标则一epoch为基准绘制
                ax.set_xticks(plot_epochs)
                plt.xlabel('epoch')
                plt.plot(plot_epochs, plot_values, label=label, marker='o')
            else:
                plt.xlabel('iter')
                plt.plot(plot_iters, plot_values, label=label, linewidth=0.5) # 绘制曲线
        plt.legend() # 添加legend
        if args.title is not None:
            plt.title(args.title) # 添加标题
    if args.out is None:
        plt.show() # 如果不保存图片则展示图片
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out) # 保存图片
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format') # 可以同时添加多个json文件
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='the metric that you want to plot') # 可以同时添加多个指标
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot') # 可以同时添加多个legend
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs] # 初始化n个dict的list
    for json_log, log_dict in zip(json_logs, log_dicts): # 逐个便利json文件
        with open(json_log, 'r') as log_file: # 打开json文件
            for line in log_file:
                log = json.loads(line.strip()) # 逐行处理
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch') # 弹出'epoch'字段，返回该epoch的值
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list) # 初始化该epoch字段为dict，默认值为list
                for k, v in log.items():
                    log_dict[epoch][k].append(v) # 将该行的所有字段加入在对应list中
    return log_dicts # 返回n个dict


def main():
    args = parse_args() # 解析命令行参数
    json_logs = args.json_logs # 赋值json文件位置
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs) # 加载json文件
    plot_curve(log_dicts, args) # 根据dict和args绘制曲线


if __name__ == '__main__':
    main()
