import argparse

import torch.onnx
from mmcv import Config
from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import os
from typing import Dict, Optional, Sequence, Union

import h5py
import mmcv
import numpy as np
import onnx
import pycuda.driver as cuda
import tensorrt as trt
import torch
import tqdm
from mmcv.runner import load_checkpoint
from mmdeploy.apis.core import no_mp
from mmdeploy.backend.tensorrt.calib_utils import HDF5Calibrator
from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin
from mmdeploy.utils import load_config
from packaging import version
from torch.utils.data import DataLoader

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module


class HDF5CalibratorBEVDet(HDF5Calibrator):

    def get_batch(self, names: Sequence[str], **kwargs) -> list:
        """Get batch data."""
        # name:['img', 'ranks_depth', 'ranks_feat', 'ranks_bev', 'interval_starts', 'interval_lengths']
        if self.count < self.dataset_length:
            # 每处理100个数据，打印输出一次
            if self.count % 100 == 0:
                print('%d/%d' % (self.count, self.dataset_length))
            ret = []
            # 逐个字段处理
            for name in names:
                input_group = self.calib_data[name] # 取出该组数据
                if name == 'img':
                    data_np = input_group[str(self.count)][...].astype(
                        np.float32) # 取出图片数据，并转换为numpy格式的float类型
                else:
                    data_np = input_group[str(self.count)][...].astype(
                        np.int32) # 针对其他数据转换为numpy的int32类型

                # tile the tensor so we can keep the same distribute
                opt_shape = self.input_shapes[name]['opt_shape'] # eg:(6, 3 256, 704)
                data_shape = data_np.shape # 

                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ] # eg:(1, 1, 1, 1)

                data_np = np.tile(data_np, reps) # (6, 3 256, 704)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list] # 切片数据

                data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes) # 创建numpy和cuda的数据指针
                cuda.memcpy_htod(data_np_cuda_ptr,
                                 np.ascontiguousarray(data_np)) # 从host拷贝到device
                self.buffers[name] = data_np_cuda_ptr # 记录数据

                ret.append(self.buffers[name])
            self.count += 1
            return ret
        else:
            return None


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='work dir to save file')
    parser.add_argument(
        '--prefix', default='bevdet', help='prefix of the save file name')
    parser.add_argument(
        '--fp16', action='store_true', help='Whether to use tensorrt fp16')
    parser.add_argument(
        '--int8', action='store_true', help='Whether to use tensorrt int8')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def create_calib_input_data_impl(calib_file: str,
                                 dataloader: DataLoader,
                                 model_partition: bool = False,
                                 metas: list = []) -> None:
    # 以写的方式打开标定文件
    with h5py.File(calib_file, mode='w') as file:
        # 创建HDF5的data group
        calib_data_group = file.create_group('calib_data') # <HDF5 group "/calib_data">
        assert not model_partition
        # create end2end group
        input_data_group = calib_data_group.create_group('end2end') # <HDF5 group "/calib_data/end2end">
        input_group_img = input_data_group.create_group('img') # <HDF5 group "/calib_data/end2end/img">
        input_keys = [
            'ranks_bev', 'ranks_depth', 'ranks_feat', 'interval_starts',
            'interval_lengths'
        ]
        input_groups = []
        for input_key in input_keys:
            input_groups.append(input_data_group.create_group(input_key)) # 逐个创建输入group eg: <HDF5 group "/calib_data/end2end/ranks_bev">
        # bev pool输入信息：
        # ranks_bev: (179535,)
        # ranks_depth: (179535,)
        # ranks_feat: (179535,)
        # interval_starts:（11404,）
        # interval_lengths:（11404,）
        metas = [
            metas[i].int().detach().cpu().numpy() for i in range(len(metas))
        ]
        # 逐个处理
        for data_id, input_data in enumerate(tqdm.tqdm(dataloader)):
            # save end2end data
            input_tensor = input_data['img_inputs'][0][0] # img:(1, 6, 3, 256, 704)
            input_ndarray = input_tensor.squeeze(0).detach().cpu().numpy() # (6, 3, 256, 704)
            # print(input_ndarray.shape, input_ndarray.dtype)
            input_group_img.create_dataset(
                str(data_id), # eg:0
                shape=input_ndarray.shape, # eg:(6, 3, 256, 704)
                compression='gzip',
                compression_opts=4,
                data=input_ndarray) # 创建一个HDF5数据集 <HDF5 group "/calib_data/end2end/img">
            for kid, input_key in enumerate(input_keys):
                input_groups[kid].create_dataset(
                    str(data_id),
                    shape=metas[kid].shape,
                    compression='gzip',
                    compression_opts=4,
                    data=metas[kid]) # 逐个input group的数据集构建 eg:<HDF5 group "/calib_data/end2end/ranks_bev">
            file.flush() # 刷新文件流


def create_calib_input_data(calib_file: str,
                            deploy_cfg: Union[str, mmcv.Config],
                            model_cfg: Union[str, mmcv.Config],
                            model_checkpoint: Optional[str] = None,
                            dataset_cfg: Optional[Union[str,
                                                        mmcv.Config]] = None,
                            dataset_type: str = 'val',
                            device: str = 'cpu',
                            metas: list = [None]) -> None:
    """Create dataset for post-training quantization.

    Args:
        calib_file (str): The output calibration data file.
        deploy_cfg (str | mmcv.Config): Deployment config file or
            Config object.
        model_cfg (str | mmcv.Config): Model config file or Config object.
        model_checkpoint (str): A checkpoint path of PyTorch model,
            defaults to `None`.
        dataset_cfg (Optional[Union[str, mmcv.Config]], optional): Model
            config to provide calibration dataset. If none, use `model_cfg`
            as the dataset config. Defaults to None.
        dataset_type (str, optional): The dataset type. Defaults to 'val'.
        device (str, optional): Device to create dataset. Defaults to 'cpu'.
    """
    with no_mp():
        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load cfg if necessary
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg) # 加载部署配置和模型配置

        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load dataset_cfg if necessary
        dataset_cfg = load_config(dataset_cfg)[0]

        from mmdeploy.apis.utils import build_task_processor
        task_processor = build_task_processor(model_cfg, deploy_cfg, device) # VoxelDetection类

        dataset = task_processor.build_dataset(dataset_cfg, dataset_type) # 构建数据集

        dataloader = task_processor.build_dataloader(
            dataset, 1, 1, dist=False, shuffle=False) # 构建dataloader

        create_calib_input_data_impl(
            calib_file, dataloader, model_partition=False, metas=metas) # 具体实现数据标定


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              int8_mode: bool = False,
              int8_param: Optional[dict] = None,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              **kwargs) -> trt.ICudaEngine:
    """Create a tensorrt engine from ONNX.

    Modified from mmdeploy.backend.tensorrt.utils.from_onnx
    """

    import os
    old_cuda_device = os.environ.get('CUDA_DEVICE', None) # None
    os.environ['CUDA_DEVICE'] = str(device_id) # 0
    import pycuda.autoinit  # noqa:F401
    if old_cuda_device is not None:
        os.environ['CUDA_DEVICE'] = old_cuda_device
    else:
        os.environ.pop('CUDA_DEVICE')

    load_tensorrt_plugin() # 加载tensorrt的plugin库
    # 1.create builder and network
    logger = trt.Logger(log_level) # 创建logger
    builder = trt.Builder(logger) # 创建builder
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # 确定batch size
    network = builder.create_network(EXPLICIT_BATCH) # 创建网络

    # 2.parse onnx
    parser = trt.OnnxParser(network, logger) # 创建onnx解析器

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model) # 加载onnx模型

    if not parser.parse(onnx_model.SerializeToString()): # 如果没能解析则报错
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # 3.config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size
    # 3.1 create config
    config = builder.create_builder_config() # 从builder中创建config
    config.max_workspace_size = max_workspace_size # 设置最大工作空间

    cuda_version = search_cuda_version() # 获取cuda版本 eg:11.2
    if cuda_version is not None:
        version_major = int(cuda_version.split('.')[0])
        if version_major < 11:
            # cu11 support cublasLt, so cudnn heuristic tactic should disable CUBLAS_LT # noqa E501
            tactic_source = config.get_tactic_sources() - (
                1 << int(trt.TacticSource.CUBLAS_LT))
            config.set_tactic_sources(tactic_source)
    # 3.2 create profile
    profile = builder.create_optimization_profile() # 从builder中创建profile

    # 在profile中设置输入名称, 最小尺寸, 可选尺寸和最大尺寸
    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode: # 设置fp16
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode: # 设置int8 量化
        config.set_flag(trt.BuilderFlag.INT8) # 设置INT8 Flag
        assert int8_param is not None
        config.int8_calibrator = HDF5CalibratorBEVDet(
            int8_param['calib_file'], # input calibration file
            input_shapes, # the min/opt/max shape of each input.
            model_type=int8_param['model_type'], # 'end2end'
            device_id=device_id, # 0
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2)) # 标定算法类型
        if version.parse(trt.__version__) < version.parse('8'):
            builder.int8_mode = int8_mode
            builder.int8_calibrator = config.int8_calibrator

    # 4.create engine
    engine = builder.build_engine(network, config) # 根据network和config使用builder创建engine

    assert engine is not None, 'Failed to create TensorRT engine'

    save(engine, output_file_prefix + '.engine') # 保存engine
    return engine


def main():
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    load_tensorrt_plugin() # 加载自定义tensorrt插件
    assert 'bev_pool_v2' in get_plugin_names(), \
        'bev_pool_v2 is not in the plugin list of tensorrt, ' \
        'please install mmdeploy from ' \
        'https://github.com/HuangJunJie2017/mmdeploy.git' # 判断bev_pool_v2是否在tensorrt插件中

    if args.int8:
        assert args.fp16
    model_prefix = args.prefix # bevdet
    if args.int8:
        model_prefix = model_prefix + '_int8' # bevdet_int8
    elif args.fp16:
        model_prefix = model_prefix + '_fp16'
    cfg = Config.fromfile(args.config) # bevdet-r50.py
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT' # BevDetTRT

    cfg = compat_cfg(cfg) # 兼容配置文件
    cfg.gpu_ids = [0] # [0]

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False) # test data配置

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {}) # {'workers_per_gpu:4}
    }
    dataset = build_dataset(cfg.data.test) # 构建dataset
    data_loader = build_dataloader(dataset, **test_loader_cfg) # 构建dataloader

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')) # 构建model
    assert model.img_view_transformer.grid_size[0] == 128
    assert model.img_view_transformer.grid_size[1] == 128
    assert model.img_view_transformer.grid_size[2] == 1
    load_checkpoint(model, args.checkpoint, map_location='cpu') # 加载checkpoint
    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse' # bevdet_int8_fuse
        model = fuse_module(model)
    model.cuda() # 将model放到CUDA上
    model.eval() # 将model设置为eval模式

    # --------------------------------------
    # 导出onnx模型，让数据虚拟流动
    # --------------------------------------
    for i, data in enumerate(data_loader):
        # 取第一帧的输入(原始数据按照batch组织) List[img, rots, trans, cam2imgs, post_rots, post_trans, bda]
        # img: (1, 6, 3, 256, 704)
        # rots: (1, 6, 3, 3) cam2lidar旋转矩阵
        # trans: (1, 6, 3) cam2lidar平移向量
        # cam2imgs:(1, 6, 3, 3) cam内参
        # post_rots: (1, 6, 3, 3) 图像数据增强后的旋转矩阵
        # post_trans: (1, 6, 3) 图像数据增强后的平移向量
        # bda:(1, 3, 3) bev特征增强后的旋转矩阵
        inputs = [t.cuda() for t in data['ie(mg_inputs'][0]]
        # ranks_bev: (179535,)
        # ranks_depth: (179535,)
        # ranks_feat: (179535,)
        # interval_starts:（11404,）
        # interval_lengths:（11404,）
        metas = model.get_bev_pool_input(inputs)
        img = inputs[0].squeeze(0) # （6, 3, 256, 704)
        with torch.no_grad():
            torch.onnx.export(
                model, # 模型
                (img.float().contiguous(), metas[1].int().contiguous(),
                 metas[2].int().contiguous(), metas[0].int().contiguous(),
                 metas[3].int().contiguous(), metas[4].int().contiguous()), # 虚拟输入
                args.work_dir + model_prefix + '.onnx', # 保存文件名称
                opset_version=11, # 版本
                input_names=[
                    'img', 'ranks_depth', 'ranks_feat', 'ranks_bev',
                    'interval_starts', 'interval_lengths'
                ], # 输入名称
                output_names=[f'output_{j}' for j in range(36)] # 输出名称
                ) # 导出onnx模型
        break
    # check onnx model 加载onnx模型
    onnx_model = onnx.load(args.work_dir + model_prefix + '.onnx')
    try:
        onnx.checker.check_model(onnx_model) # 检查onnx模型
    except Exception:
        print('ONNX Model Incorrect')
    else:
        print('ONNX Model Correct')

    # --------------------------------------
    # 将onnx模型转换为tensorrt
    # --------------------------------------
    num_points = metas[0].shape[0] # 179535
    num_intervals = metas[3].shape[0] # 11404
    img_shape = img.shape # 6, 3, 256, 704
    input_shapes = dict(
        img=dict(
            min_shape=img_shape, opt_shape=img_shape, max_shape=img_shape),
        ranks_depth=dict(
            min_shape=[num_points], # 179535
            opt_shape=[num_points],
            max_shape=[num_points]),
        ranks_feat=dict(
            min_shape=[num_points],
            opt_shape=[num_points],
            max_shape=[num_points]),
        ranks_bev=dict(
            min_shape=[num_points],
            opt_shape=[num_points],
            max_shape=[num_points]),
        interval_starts=dict(
            min_shape=[num_intervals], # 11404
            opt_shape=[num_intervals],
            max_shape=[num_intervals]),
        interval_lengths=dict(
            min_shape=[num_intervals],
            opt_shape=[num_intervals],
            max_shape=[num_intervals]))
    deploy_cfg = dict(
        backend_config=dict(
            type='tensorrt',  # backend
            common_config=dict(
                fp16_mode=args.fp16,
                max_workspace_size=1073741824,
                int8_mode=args.int8),
            model_inputs=[dict(input_shapes=input_shapes)]),
        codebase_config=dict(
            type='mmdet3d', task='VoxelDetection', model_type='end2end')) # onnx_config和calib_config两步分开实现，不写在配置中

    if args.int8:
        calib_filename = 'calib_data.h5'
        calib_path = os.path.join(args.work_dir, calib_filename) # eg:works_dirs/bevdettrt-r50-trt/calib_data.h5
        create_calib_input_data(
            calib_path,
            deploy_cfg,
            args.config,
            args.checkpoint,
            dataset_cfg=None,
            dataset_type='val',
            device='cuda:0',
            metas=metas) # 生成标定数据

    from_onnx(
        args.work_dir + model_prefix + '.onnx',
        args.work_dir + model_prefix,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        int8_param=dict(
            calib_file=os.path.join(args.work_dir, 'calib_data.h5'),
            model_type='end2end'),
        max_workspace_size=1 << 30,
        input_shapes=input_shapes)

    if args.int8:
        os.remove(calib_path)


if __name__ == '__main__':

    main()
