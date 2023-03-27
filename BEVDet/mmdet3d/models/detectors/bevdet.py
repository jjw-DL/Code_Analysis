# Copyright (c) Phigent Robotics. All rights reserved.

import os
import torch
import torch.nn.functional as F

from mmdet.models import DETECTORS
from .centerpoint import CenterPoint
from .. import builder
from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class BEVDet(CenterPoint):
    def __init__(self, img_view_transformer, img_bev_encoder_backbone, img_bev_encoder_neck, **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

    def image_encoder(self,img):
        imgs = img
        B, N, C, imH, imW = imgs.shape # 8, 6, 3, 256, 704
        imgs = imgs.view(B * N, C, imH, imW) # 48, 3, 256, 704
        x = self.img_backbone(imgs) # (48, 384, 16, 44)和(48, 768, 8, 22)
        if self.with_img_neck:
            x = self.img_neck(x) # (48, 512, 16, 44)
        _, output_dim, ouput_H, output_W = x.shape # 48, 512, 16, 44
        x = x.view(B, N, output_dim, ouput_H, output_W) # (8, 6, 512, 16, 44)
        return x # (8, 6, 512, 16, 44)

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x) # [(8, 128, 64, 64), (8, 256, 32, 32), (8, 512, 16, 16)]
        x = self.img_bev_encoder_neck(x) # (8, 256, 128, 128)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0]) # (8, 6, 512, 16, 44)
        x = self.img_view_transformer([x] + img[1:]) # (8, 64, 128, 128)
        x = self.bev_encoder(x) # (8, 256, 128, 128)
        return [x]

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas) # [(8, 256, 128, 128)]
        pts_feats = None
        return (img_feats, pts_feats)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        """
        points: None
        img_inputs: 
            imgs: (8, 6, 3, 256, 704)
            rots: (8, 6, 3, 3)
            trans: (8, 6, 3)
            intrins: (8, 6, 3, 3)
            post_rots: (8, 6, 3, 3)
            post_trans: (8, 6, 3)
        img_metas:
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas) # [(8, 256, 128, 128)]和None
        assert self.with_pts_bbox
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0],list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        combine_type = self.test_cfg.get('combine_type','output')
        if combine_type=='output':
            return self.aug_test_combine_output(points, img_metas, img, rescale)
        elif combine_type=='feature':
            return self.aug_test_combine_feature(points, img_metas, img, rescale)
        else:
            assert False

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        img_metas=[dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


@DETECTORS.register_module()
class BEVDetSequential(BEVDet):
    def __init__(self, aligned=False, distill=None, pre_process=None,
                 pre_process_neck=None, detach=True, test_adj_ids=None, **kwargs):
        super(BEVDetSequential, self).__init__(**kwargs)
        self.aligned = aligned
        self.distill = distill is not None
        if self.distill:
            self.distill_net = builder.build_neck(distill)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process) # ResNetForBEVDet
        self.pre_process_neck = pre_process_neck is not None
        if self.pre_process_neck:
            self.pre_process_neck_net = builder.build_neck(pre_process_neck)
        self.detach = detach
        self.test_adj_ids = test_adj_ids
    
    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)
        imgs = torch.split(imgs,1,2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for img, rot, tran, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.img_view_transformer.depthnet(x)
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D])
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H,
                                 W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            if self.pre_process:
                bev_feat = self.pre_process_net(bev_feat)
                if self.pre_process_neck:
                    bev_feat = self.pre_process_neck_net(bev_feat)
                else:
                    bev_feat = bev_feat[0]
            bev_feat_list.append(bev_feat)
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x]


@DETECTORS.register_module()
class BEVDetSequentialES(BEVDetSequential):
    def __init__(self, before=False, interpolation_mode='bilinear',**kwargs):
        super(BEVDetSequentialES, self).__init__(**kwargs)
        self.before=before # True
        self.interpolation_mode=interpolation_mode # 'bilinear'

    def shift_feature(self, input, trans, rots):
        n, c, h, w = input.shape # (8, 64, 128, 128)
        _,v,_ =trans[0].shape # (8, 6, 3)

        # generate grid
        # (128, ) --> (1, 128) --> (128, 128)
        # 0~127产生128个点
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)
        # (128, 128, 3) --> (1, 128, 128, 3) --> (8, 128, 128, 3) --> (8, 128, 128, 3, 1)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n,h,w,3,1)
        grid = grid

        # get transformation from current frame to adjacent frame
        l02c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid) # (8, 6, 4, 4) 当前帧camera2lidar
        l02c[:,:,:3,:3] = rots[0] # (8, 6, 3, 3)
        l02c[:,:,:3,3] = trans[0] # (8, 6, 3)
        l02c[:,:,3,3] =1

        l12c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid) # (8, 6, 4, 4) 前一帧camera2lidar
        l12c[:,:,:3,:3] = rots[1] # (8, 6, 3, 3)
        l12c[:,:,:3,3] = trans[1] # (8, 6, 3)
        l12c[:,:,3,3] =1
        # l0tol1 = l12c.matmul(torch.inverse(l02c))[:,0,:,:].view(n,1,1,4,4)
        # (8, 6, 4, 4) --> (8, 4, 4) --> (8, 1, 1, 4, 4)
        l0tol1 = l02c.matmul(torch.inverse(l12c))[:,0,:,:].view(n,1,1,4,4)
        # (8, 1, 1, 3, 4) --> (8, 1, 1, 3, 3)
        l0tol1 = l0tol1[:,:,:,[True,True,False,True],:][:,:,:,:,[True,True,False,True]]

        feat2bev = torch.zeros((3,3),dtype=grid.dtype).to(grid) # (3, 3)
        feat2bev[0, 0] = self.img_view_transformer.dx[0] # 0.8
        feat2bev[1, 1] = self.img_view_transformer.dx[1] # 0.8 
        feat2bev[0, 2] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2. # -51.2
        feat2bev[1, 2] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2. # -51.2
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1,3,3)
        tf = torch.inverse(feat2bev).matmul(l0tol1).matmul(feat2bev) # (8, 1, 1, 3, 3)

        # transform and normalize
        # 先将网格点转换为实际坐标，然后从前一帧转到当前帧，最后在转换为网格点
        grid = tf.matmul(grid) # (8, 1, 1, 3, 3) * (8, 128, 128, 3, 1) --> (8, 128, 128, 3, 1) 
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype, device=input.device) # [127, 127]
        # (8, 128, 128, 2) / (1, 1, 1, 2) --> (8, 128, 128, 2)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0 # grid的合理值在[-1, 1]之间，超出部分被截断
        # (8, 64, 128, 128)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True, mode=self.interpolation_mode)
        return output


    def extract_img_feat(self, img, img_metas):
        inputs = img # List[5]
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape # 8, 12, 3, 256, 704
        N = N//2 # 6
        imgs = inputs[0].view(B,N,2,3,H,W) # (8, 6, 2, 3, 256, 704)
        imgs = torch.split(imgs,1,2) # dim=2, List[(8, 6, 1, 3, 256, 704)]
        imgs = [t.squeeze(2) for t in imgs] # List[(8, 6, 3, 256, 704)]
        rots, trans, intrins, post_rots, post_trans = inputs[1:] # eg:rot:(8, 12, 3, 3)
        extra = [rots.view(B,2,N,3,3), # (8, 2, 6, 3, 3)
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra] # List[List[(8, 1, 6, 3, 3)]]
        extra = [[p.squeeze(1) for p in t] for t in extra] # List[List[(8, 6, 3, 3)]]
        rots, trans, intrins, post_rots, post_trans = extra # eg:rots:List[(8, 6, 3, 3)]
        bev_feat_list = []
        # 逐帧处理
        for img, _ , _, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            tran = trans[0] # camera2lidar的平移 (8, 6, 3)
            rot = rots[0] # camera2lidar的旋转 (8, 6, 3, 3)
            x = self.image_encoder(img) # (8, 6, 512, 16, 44)
            B, N, C, H, W = x.shape # 8, 6, 512, 16, 44
            x = x.view(B * N, C, H, W) # (48, 512, 16, 44)
            x = self.img_view_transformer.depthnet(x) # (48, 123, 16, 44)
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran) # (8, 6, 59, 16, 44, 3) lidar系下的伪点云坐标
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D]) #（48, 59, 16, 44）
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)] #（48, 64, 16, 44）

            # Lift
            # (48, 1, 59, 16, 44) *  (48, 64, 1, 16, 44) --> (48, 64, 59, 16, 44)
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2) # (48, 64, 59, 16, 44)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H, W) # (8, 6, 64, 59, 16, 44)
            volume = volume.permute(0, 1, 3, 4, 5, 2) # (8, 6, 59, 16, 44, 64)

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume) # (8, 64, 128, 128)

            bev_feat_list.append(bev_feat) # 将BEV Feature加入list
        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list] # List[(8, 64, 128, 128)]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans, rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach() # 将前一帧的特征图detach不参与反向传播
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1) # 将两帧特征图拼接 (8, 128, 128, 128)

        x = self.bev_encoder(bev_feat) # (8, 256, 128, 128)
        return [x]
