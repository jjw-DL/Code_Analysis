# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info

from custom_eval.prd_evaluator import PRDEvaluator
from custom_eval.internal_evaluator import InternalEvaluator
from custom_eval.internal_visualizer import InternalVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--sample_rate', help='sample_rate')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    rank, _ = get_dist_info()
    if rank == 0:
        if not os.path.exists(os.path.split(args.out)[0]):
            os.makedirs(os.path.split(args.out)[0], exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    rank, _ = get_dist_info()
    if rank == 0:
        # evaluate_internal_data
        prd_evaluator = PRDEvaluator(cfg, args.out)
        prd_evaluator.evaluate_internal_data()

        # evaluate_internal_data
        # internal_evaluator = InternalEvaluator(cfg, args.out)
        # internal_evaluator.evaluate_internal_data()

        # visualize_internal_data
        internal_visualizer = InternalVisualizer(args.config,
                                                 args.out,
                                                 cfg_options=args.cfg_options)
        internal_visualizer.visualize(sample_rate=args.sample_rate)


if __name__ == '__main__':
    main()
