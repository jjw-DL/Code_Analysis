# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        # 如果图片的  宽 > 高, 记为 为 1
        #            宽 < 高, 记为 为 0
        # flag 是一个记录了数据集中所有图片的 ndarray
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        # np.bincount 计算每个索引出现的次数
        # 在这里就相当于计算了有多少个宽 > 高的图片, 和有多少个宽 < 高的图片
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            # 保证每组的 sample 数都能被 samples_per_gpu 的数量整除
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
             # 如果数据集中的所有的图片的宽都 < 高, 那么进行下一次循环.
            if size == 0:
                continue
            # 找到 宽 < 高(i = 0) 或 宽 > 高(i = 1) 的所有的图片索引
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            # 随机打乱索引
            np.random.shuffle(indice)
            # 因为图片个数不一定会被 samples_per_gpu 整除, 所以添加额外的数据.
            # num_extra 即为添加额外数据的数量.
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            # np.concatenate(需要concat的list, axis=0)
            # np.random.choice(list, 选的size)
            # 生成所有的 index
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        # 整合所有的 index
        indices = np.concatenate(indices)
        # 如下操作可以保证每个 samples_per_gpu 的 flag 都相同
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        # 获取 rank 和 world_size (num_replicas)
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        # 统计了有多少个宽 > 高的图片, 和有多少个宽 < 高的图片
        self.group_sizes = np.bincount(self.flag)

        # 每个进程需要采样的样本数
        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            # self.group_sizes[i] / self.samples_per_gpu：能分成多少组
            # 下面的代表计算了每个机器分的个数.
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        # 所有进程要采样的样本总数
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        # 把当前的 epoch 作为随机数种子,
        # 这样能保证在相同的 epoch 的实验有可重复性,
        # 且在不同的 epoch 之间有随机性.
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            # 如果有样本
            if size > 0:
                # 找出所有属于此类的索引
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                # 随机打乱索引
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                # 总共需要额外添加的样本数
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                # 填充 indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                # 取随机后的前 extra 个作为 extra 样本.
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        # 打乱 sample_per_gpu 之间的顺序,
        # 因为上面已经打乱了每个 group 之内的元素,
        # 所以这里只用打乱组之间的顺序即可.
        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        # 采样 num_samples 个.不同进程之间按照打乱的数据集顺序采样.
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
