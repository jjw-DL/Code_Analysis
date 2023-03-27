# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import re
import sys
from typing import Any, Dict, Optional, Sequence, Union

import onnx
import tensorrt as trt
from packaging import version

from mmdeploy.utils import get_root_logger
from .init_plugins import load_tensorrt_plugin


def save(engine: trt.ICudaEngine, path: str) -> None:
    """Serialize TensorRT engine to disk.

    Args:
        engine (tensorrt.ICudaEngine): TensorRT engine to be serialized.
        path (str): The absolute disk path to write the engine.
    """
    with open(path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))


def load(path: str, allocator: Optional[Any] = None) -> trt.ICudaEngine:
    """Deserialize TensorRT engine from disk.

    Args:
        path (str): The disk path to read the engine.
        allocator (Any): gpu allocator

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine loaded from disk.
    """
    load_tensorrt_plugin()
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        if allocator is not None:
            runtime.gpu_allocator = allocator
        with open(path, mode='rb') as f:
            engine_bytes = f.read()
        trt.init_libnvinfer_plugins(logger, namespace='')
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine


def search_cuda_version() -> str:
    """try cmd to get cuda version, then try `torch.cuda`

    Returns:
        str: cuda version, for example 10.2
    """

    version = None

    pattern = re.compile(r'[0-9]+\.[0-9]+')
    platform = sys.platform.lower()

    def cmd_result(txt: str):
        cmd = os.popen(txt)
        return cmd.read().rstrip().lstrip()

    if platform == 'linux' or platform == 'darwin' or platform == 'freebsd':  # noqa E501
        version = cmd_result(
            " nvcc --version | grep  release | awk '{print $5}' | awk -F , '{print $1}' "  # noqa E501
        )
        if version is None or pattern.match(version) is None:
            version = cmd_result(
                " nvidia-smi  | grep CUDA | awk '{print $9}' ")

    elif platform == 'win32' or platform == 'cygwin':
        # nvcc_release = "Cuda compilation tools, release 10.2, V10.2.89"
        nvcc_release = cmd_result(' nvcc --version | find "release" ')
        if nvcc_release is not None:
            result = pattern.findall(nvcc_release)
            if len(result) > 0:
                version = result[0]

        if version is None or pattern.match(version) is None:
            # nvidia_smi = "| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |" # noqa E501
            nvidia_smi = cmd_result(' nvidia-smi | find "CUDA Version" ')
            result = pattern.findall(nvidia_smi)
            if len(result) > 2:
                version = result[2]

    if version is None or pattern.match(version) is None:
        try:
            import torch
            version = torch.version.cuda
        except Exception:
            pass

    return version


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

    Args:
        onnx_model (str or onnx.ModelProto): Input onnx model to convert from.
        output_file_prefix (str): The path to save the output ncnn file.
        input_shapes (Dict[str, Sequence[int]]): The min/opt/max shape of
            each input.
        max_workspace_size (int): To set max workspace size of TensorRT engine.
            some tactics and layers need large workspace. Defaults to `0`.
        fp16_mode (bool): Specifying whether to enable fp16 mode.
            Defaults to `False`.
        int8_mode (bool): Specifying whether to enable int8 mode.
            Defaults to `False`.
        int8_param (dict): A dict of parameter  int8 mode. Defaults to `None`.
        device_id (int): Choice the device to create engine. Defaults to `0`.
        log_level (trt.Logger.Severity): The log level of TensorRT. Defaults to
            `trt.Logger.ERROR`.

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine created from onnx_model.

    Example:
        >>> from mmdeploy.apis.tensorrt import from_onnx
        >>> engine = from_onnx(
        >>>             "onnx_model.onnx",
        >>>             {'input': {"min_shape" : [1, 3, 160, 160],
        >>>                        "opt_shape" : [1, 3, 320, 320],
        >>>                        "max_shape" : [1, 3, 640, 640]}},
        >>>             log_level=trt.Logger.WARNING,
        >>>             fp16_mode=True,
        >>>             max_workspace_size=1 << 30,
        >>>             device_id=0)
        >>>             })
    """

    if device_id != 0:
        import os
        old_cuda_device = os.environ.get('CUDA_DEVICE', None) # None
        os.environ['CUDA_DEVICE'] = str(device_id) # 0
        import pycuda.autoinit  # noqa:F401
        if old_cuda_device is not None:
            os.environ['CUDA_DEVICE'] = old_cuda_device
        else:
            os.environ.pop('CUDA_DEVICE')

    load_tensorrt_plugin() # 加载tensorrt的plugin库
    # create builder and network
    logger = trt.Logger(log_level) # 创建logger
    builder = trt.Builder(logger) # 创建builder

    # TODO: use TorchAllocator as builder.gpu_allocator

    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # 确定batch size
    network = builder.create_network(EXPLICIT_BATCH) # 创建网络

    # parse onnx
    parser = trt.OnnxParser(network, logger) # 创建onnx解析器

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model) # 加载onnx模型

    if not parser.parse(onnx_model.SerializeToString()): # 如果没能解析则报错
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config() # 从builder中创建config

    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     max_workspace_size)
    else:
        config.max_workspace_size = max_workspace_size # 设置最大工作空间

    cuda_version = search_cuda_version() # 获取cuda版本 eg:11.2
    if cuda_version is not None:
        version_major = int(cuda_version.split('.')[0])
        if version_major < 11:
            # cu11 support cublasLt, so cudnn heuristic tactic should disable CUBLAS_LT # noqa E501
            tactic_source = config.get_tactic_sources() - (
                1 << int(trt.TacticSource.CUBLAS_LT))
            config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile() # 从builder中创建profile

    # 在profile中设置输入名称, 最小尺寸, 可选尺寸和最大尺寸
    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    if config.add_optimization_profile(profile) < 0:
        logger.warning(f'Invalid optimization profile {profile}.')

    if fp16_mode: # 设置fp16
        if not getattr(builder, 'platform_has_fast_fp16', True):
            logger.warning('Platform does not has fast native fp16.')
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode: # 设置int8 量化
        if not getattr(builder, 'platform_has_fast_int8', True):
            logger.warning('Platform does not has fast native int8.')
        from .calib_utils import HDF5Calibrator
        config.set_flag(trt.BuilderFlag.INT8) # 设置INT8 Flag
        assert int8_param is not None
        config.int8_calibrator = HDF5Calibrator(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2)) # 在config中设置INT8标定器
        if version.parse(trt.__version__) < version.parse('8'):
            builder.int8_mode = int8_mode
            builder.int8_calibrator = config.int8_calibrator

    # create engine
    engine = builder.build_engine(network, config) # 根据network和config使用builder创建engine

    assert engine is not None, 'Failed to create TensorRT engine'

    save(engine, output_file_prefix + '.engine') # 保存engine
    return engine


def get_trt_log_level() -> trt.Logger.Severity:
    """Get tensorrt log level from root logger.

    Returns:
        level (tensorrt.Logger.Severity):
        Logging level of tensorrt.Logger.
    """
    logger = get_root_logger()
    level = logger.level
    trt_log_level = trt.Logger.INFO
    if level == logging.ERROR:
        trt_log_level = trt.Logger.ERROR
    elif level == logging.WARNING:
        trt_log_level = trt.Logger.WARNING
    elif level == logging.DEBUG:
        trt_log_level = trt.Logger.VERBOSE
    return trt_log_level