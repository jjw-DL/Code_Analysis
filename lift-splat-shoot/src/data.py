"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W'] # 900, 1600
        fH, fW = self.data_aug_conf['final_dim'] # 128, 352
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim']) # (0.193-0.225)-->随机均匀分布eg:0.2172
            resize_dims = (int(W*resize), int(H*resize)) # eg:(347, 192)
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH # (0, 0.22)随机均匀分布 --> 45
            crop_w = int(np.random.uniform(0, max(0, newW - fW))) # 0 (resize后新的宽小于特征图宽)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH) # (0, 45, 352, 173)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True # 0.5的概率决定是否翻转
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim']) # (-5.4, 5.4) 随机均匀分布 eg:4.7279
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        # 逐个相机处理
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam]) # 获取该帧相机的record
            imgname = os.path.join(self.nusc.dataroot, samp['filename']) # 拼接该该帧相机的图片路径
            img = Image.open(imgname) # 读取图片
            post_rot = torch.eye(2) # 初始化旋转
            post_tran = torch.zeros(2) # 初始化平移

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token']) # 获取该相机的record的token
            intrin = torch.Tensor(sens['camera_intrinsic']) # 获取该相机的内参
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix) # 获取该相机的旋转矩阵(相机到自车)
            tran = torch.Tensor(sens['translation']) # 获取该相机的平移向量

            # augmentation (resize, crop, horizontal flip, rotate) 数据增强的参数
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # 进行数据增强 --> 数据增强后的图片，旋转矩阵和平移向量
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     ) 
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3) # 初始化平移向量(3D)
            post_rot = torch.eye(3) # 初始化旋转矩阵(3D)
            post_tran[:2] = post_tran2 # 2D-->3D的赋值
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img)) # 正则化图像
            intrins.append(intrin) # 相机内参
            rots.append(rot) # 相机到自车的旋转矩阵
            trans.append(tran) # 相机到自车的平移向量
            post_rots.append(post_rot) # 数据增强后的旋转矩阵
            post_trans.append(post_tran) # 数据增强后的平移向量

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2) # 拼接nsweeps帧点云到自车坐标系下
        return torch.Tensor(pts)[:3]  # x,y,z 

    def get_binimg(self, rec):
        # 获取自车位姿的record
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation']) # 获取平移向量（自车在全局坐标系下的平移，将全局系下的点变换到自车系则取负数）
        rot = Quaternion(egopose['rotation']).inverse # 获取旋转四元数的逆(同样旋转也取逆)
        img = np.zeros((self.nx[0], self.nx[1])) # 初始化图像(200, 200)
        # 逐个标注处理
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok) # 获取该ann的record
            # add category for lyft 只分割车
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation'])) # 初始化Nuscenes的box类(全局坐标系)
            box.translate(trans) # 平移box到自车坐标系
            box.rotate(rot) # 旋转box到自车坐标系

            pts = box.bottom_corners()[:2].T # 获取box的底部角点 (3, 4)-->(2, 4)-->(4, 2) xyz-->xy
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32) # 计算特征图坐标(原点从中心点移到左下角，同时除分辨率) --> (4, 2)
            pts[:, [1, 0]] = pts[:, [0, 1]] # x和y交换，自车坐标系的车长在图像坐标系中为y
            cv2.fillPoly(img, [pts], 1.0) # 填充图像

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']): # 5 < 6
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False) # 随机drop相机, 6个相机随机选择5个
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index] # 获取该sample的record

        cams = self.choose_cams() # 随机选择5个相机，drop掉一个
        # imgs:(5, 3 128, 352)
        # rots:(5, 3, 3)
        # trans:(5, 3)
        # intrins:(5, 3, 3)
        # post_rots:(5, 3, 3)
        # post_trans:(5, 3)
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec) # 根据标注计算标签图像(1, 200, 200)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
