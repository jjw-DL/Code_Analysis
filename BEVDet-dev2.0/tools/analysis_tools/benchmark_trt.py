import time
from typing import Dict, Optional, Sequence, Union

import tensorrt as trt
import torch
import torch.onnx
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import argparse

from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('engine', help='checkpoint file')
    parser.add_argument('--samples', default=500, help='samples to benchmark')
    parser.add_argument('--postprocessing', action='store_true')
    args = parser.parse_args()
    return args


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')


class TRTWrapper(torch.nn.Module):

    def __init__(self,
                 engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine # engine文件名 eg:bevdet-r50-trtbevdet_int8_fuse.engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime: # 创建logger和runtime
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read() # 读取engine
                self.engine = runtime.deserialize_cuda_engine(engine_bytes) # 反序列化eigen-->ECudaEngine
        self.context = self.engine.create_execution_context() # 通过engine创建执行上下-->IExecutionContext
        names = [_ for _ in self.engine] # 获取输入和输出的名字
        input_names = list(filter(self.engine.binding_is_input, names)) # 获取输入名称
        self._input_names = input_names # 赋值输入名称
        self._output_names = output_names # 赋值输出名称

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        bindings = [None] * (len(self._input_names) + len(self._output_names)) # 初始化绑定值[]
        # input 逐个输入处理
        for input_name, input_tensor in inputs.items():
            idx = self.engine.get_binding_index(input_name) # 根据名称获取绑定id eg:0
            self.context.set_binding_shape(idx, tuple(input_tensor.shape)) # 将id和shape进行绑定 eg:(6, 3, 256, 704)
            bindings[idx] = input_tensor.contiguous().data_ptr() # 将数据指针和id进行绑定

        # create output tensors 逐个输出处理
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name) # 根据名称获取绑定id eg:6
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx)) # 获取绑定的数据类型 eg:torch.float32
            shape = tuple(self.context.get_binding_shape(idx)) # 获取绑定的shape eg:(1, 2, 128, 128)

            device = torch.device('cuda') # 设置cuda device
            output = torch.zeros(size=shape, dtype=dtype, device=device) # 初始化输出
            outputs[output_name] = output # 记录输出
            bindings[idx] = output.data_ptr() # 将数据指针和id进行绑定
        
        # 执行推理
        # bindings内记录的是输入和输出的数据指针
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def main():

    load_tensorrt_plugin() # 加载tensorrt库

    args = parse_args() # 解析参数

    cfg = Config.fromfile(args.config) # 读取配置文件
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT' # 设置模型类型 bevdetTRT
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0] # 设置GPU

    # build dataloader
    assert cfg.data.test.test_mode
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test) # 构建数据集
    data_loader = build_dataloader(dataset, **test_loader_cfg) # 构建dataloader

    # build the model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')) # 构建模型

    # build tensorrt model
    trt_model = TRTWrapper(args.engine, [f'output_{i}' for i in range(36)]) # 构建TRT模型

    num_warmup = 50 # 设置热身数量
    pure_inf_time = 0 # 记录纯推理时间

    init_ = True
    metas = dict()
    # benchmark with several samples and take the average 
    # 以多个样本为基准并取平均值，逐帧处理
    for i, data in enumerate(data_loader):
        if init_:
            inputs = [t.cuda() for t in data['img_inputs'][0]] # 提取输入input-->[0]表示batch中的第一帧
            metas_ = model.get_bev_pool_input(inputs) # 根据各个变换矩阵计算bev pool的输入
            metas = dict(
                ranks_bev=metas_[0].int().contiguous(),
                ranks_depth=metas_[1].int().contiguous(),
                ranks_feat=metas_[2].int().contiguous(),
                interval_starts=metas_[3].int().contiguous(),
                interval_lengths=metas_[4].int().contiguous()) # 重新组织bev pool的输入
            init_ = False # 因为测试过程中没有数据增强，所以只需要计算一次bev pool的输入
        img = data['img_inputs'][0][0].cuda().squeeze(0).contiguous()
        torch.cuda.synchronize() # 同步cuda
        start_time = time.perf_counter() # 记录开始时间
        trt_output = trt_model.forward(dict(img=img, **metas)) # 执行infer，调用TRTBEVPoolv2

        # postprocessing
        if args.postprocessing: # 对输出进行后处理
            trt_output = [trt_output[f'output_{i}'] for i in range(36)]
            pred = model.result_deserialize(trt_output)
            img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
            bbox_list = model.pts_bbox_head.get_bboxes(
                pred, img_metas, rescale=True)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
        torch.cuda.synchronize() # 同步cuda
        elapsed = time.perf_counter() - start_time # 计算infer时间

        if i >= num_warmup: # 当infer数量大于热身数量时
            pure_inf_time += elapsed # 累加infer时间
            if (i + 1) % 50 == 0: # 每50帧计算一次
                fps = (i + 1 - num_warmup) / pure_inf_time # 计算fps
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s') # 打印fps

        if (i + 1) == args.samples: # 当infer数量等于设定数量时
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall \nfps: {fps:.1f} img / s '
                  f'\ninference time: {1000/fps:.1f} ms')
            return fps # 结束评估


if __name__ == '__main__':
    fps = main()
