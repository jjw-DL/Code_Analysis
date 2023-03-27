# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file') # 点云
    parser.add_argument('config', help='Config file') # 配置文件
    parser.add_argument('checkpoint', help='Checkpoint file') # 权重
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference') # cuda device
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold') # 分数阈值
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results') # 输出文件夹路径
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results') # 是否可视化预测结果
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results') # 是否保存截图
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device) # 初始化模型
    # test a single image
    result, data = inference_detector(model, args.pcd) # infer
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='det') # 显示结果


if __name__ == '__main__':
    main()
