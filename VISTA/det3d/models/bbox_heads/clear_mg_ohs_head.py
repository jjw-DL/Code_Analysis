# Copyright (c) Gorilla-Lab. All rights reserved.
import logging
from functools import partial
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.ohs_loss_clear import OHSLossClear
from ..losses.attention_constrain_loss import AttentionConstrainedLoss
from ..registry import HEADS
from ..builder import build_loss
from ...core.bbox import box_torch_ops
from ...core.bbox.geometry import points_in_convex_polygon_torch
from ...core.bbox.box_coders import BoxCoder, GroundBox3dCoderAF
from ipdb import set_trace

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def _get_pos_neg_loss(cls_loss, labels, label_weights):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0] # 4
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1) # (4, 25600)
        cls_neg_loss = ((labels == 0) & (label_weights > 0)).type_as(
            cls_loss) * cls_loss.view(batch_size, -1) # (4, 25600)
        cls_pos_loss = cls_pos_loss.sum() / batch_size # eg:11.3964
        cls_neg_loss = cls_neg_loss.sum() / batch_size # eg:5913.0508
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


@HEADS.register_module
class OHSHeadClear(nn.Module):
    def __init__(self,
                 box_coder: GroundBox3dCoderAF,
                 num_input: int,
                 num_pred: int,
                 num_cls: int,
                 header: bool = True,
                 name: str = "",
                 **kwargs,):
        super().__init__()

        self.box_coder = box_coder
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)
        self.mode = kwargs.get("mode", "bev")
        if self.box_coder.center == "direct":
            self.conv_xy = nn.Conv2d(num_input, 2, 1)
        elif self.box_coder.center == "soft_argmin":
            self.conv_xy = nn.Conv2d(num_input, 2 * self.box_coder.kwargs["xy_bin_num"], 1)
            self.loc_bins_x = torch.linspace(self.box_coder.kwargs["x_range"][0], self.box_coder.kwargs["x_range"][1],
                                             self.box_coder.kwargs["xy_bin_num"]).reshape(1, 1, -1, 1, 1)
            self.loc_bins_y = torch.linspace(self.box_coder.kwargs["y_range"][0], self.box_coder.kwargs["y_range"][1],
                                             self.box_coder.kwargs["xy_bin_num"]).reshape(1, 1, -1, 1, 1)
            self.loc_bins = torch.cat([self.loc_bins_x, self.loc_bins_y], 1)
        else:
            raise NotImplementedError

        if "direct" in self.box_coder.height:
            self.conv_z = nn.Conv2d(num_input, 1, 1)
        elif "soft_argmin" in self.box_coder.height:
            self.conv_z = nn.Conv2d(num_input, self.box_coder.kwargs["z_bin_num"], 1)
            self.z_loc_bins = torch.linspace(self.box_coder.kwargs["z_range"][0], self.box_coder.kwargs["z_range"][1],
                                             self.box_coder.kwargs["z_bin_num"]).reshape(1, self.box_coder.kwargs["z_bin_num"], 1, 1)
        else:
            raise NotImplementedError
        if "soft_argmin" in self.box_coder.dim:
            self.conv_dim = nn.Conv2d(num_input, 3 * self.box_coder.kwargs["dim_bin_num"], 1)
            self.dim_loc_bins = torch.linspace(self.box_coder.kwargs["dim_range"][0], self.box_coder.kwargs["dim_range"][1],
                                               self.box_coder.kwargs["dim_bin_num"]).reshape(1, self.box_coder.kwargs[
                                                   "dim_bin_num"], 1, 1)
            self.dim_bins = torch.cat([self.dim_loc_bins, self.dim_loc_bins, self.dim_loc_bins], 1)
        else:
            self.conv_dim = nn.Conv2d(num_input, 3, 1)
        if self.box_coder.velocity:
            self.conv_velo = nn.Conv2d(num_input, 2, 1)
        if self.box_coder.rotation == "vector":
            self.conv_r = nn.Conv2d(num_input, 2, 1)
        elif self.box_coder.rotation == "soft_argmin":
            self.conv_r = nn.Conv2d(num_input, self.box_coder.kwargs["r_bin_num"], 1)
            self.r_loc_bins = torch.linspace(-np.pi, np.pi, self.box_coder.kwargs["r_bin_num"]).reshape(
                1, self.box_coder.kwargs["r_bin_num"], 1, 1)
        else:
            self.conv_r = nn.Conv2d(num_input, 1, 1)

    def forward(self, x, return_loss):
        x_bev = x
        ret_dict = {}
        cls_preds = self.conv_cls(x_bev).permute(0, 2, 3, 1).contiguous()

        # predict bounding box
        xy = self.conv_xy(x_bev)
        z = self.conv_z(x_bev)
        dim = self.conv_dim(x_bev)
        # encode as bounding box
        if self.box_coder.center == "soft_argmin":
            xy = xy.view(
                (xy.shape[0], 2, self.box_coder.kwargs["xy_bin_num"], xy.shape[2], xy.shape[3]))
            xy = F.softmax(xy, dim=2)
            xy = xy * self.loc_bins.to(xy.device)
            xy = torch.sum(xy, dim=2, keepdim=False)
        if "soft_argmin" in self.box_coder.height:
            z = F.softmax(z, dim=1)
            z = z * self.z_loc_bins.to(z.device)
            z = torch.sum(z, dim=1, keepdim=True)
        if "soft_argmin" in self.box_coder.dim:
            dim = dim.view(
                (dim.shape[0], 3, self.box_coder.kwargs["dim_bin_num"], dim.shape[2], dim.shape[3]))
            dim = F.softmax(dim, dim=2)
            dim = dim * self.dim_loc_bins.to(dim.device)
            dim = torch.sum(dim, dim=2, keepdim=False)
        xy = xy.permute(0, 2, 3, 1).contiguous()
        z = z.permute(0, 2, 3, 1).contiguous()
        dim = dim.permute(0, 2, 3, 1).contiguous()

        if self.box_coder.dim == "direct":
            dim = F.relu(dim)

        if self.box_coder.velocity:
            velo = self.conv_velo(x_bev).permute(0, 2, 3, 1).contiguous()

        r_preds = self.conv_r(x_bev)
        if self.box_coder.rotation == "vector":
            #import pdb; pdb.set_trace()
            r_preds = F.normalize(r_preds, p=2, dim=1)
        elif self.box_coder.rotation == "soft_argmin":
            r_preds = F.softmax(r_preds, dim=1)
            r_preds = r_preds * self.r_loc_bins.to(r_preds.device)
            r_preds = torch.sum(r_preds, dim=1, keepdim=True)
        r_preds = r_preds.permute(0, 2, 3, 1).contiguous()
        if self.box_coder.velocity:
            box_preds = torch.cat([xy, z, dim, velo, r_preds], -1)
        else:
            box_preds = torch.cat([xy, z, dim, r_preds], -1)

        ret_dict.update({"box_preds": box_preds, "cls_preds": cls_preds})

        return ret_dict


@HEADS.register_module
class MultiGroupOHSHeadClear(nn.Module):
    def __init__(self,
                 mode: str = "3d",
                 in_channels: List[int] = [128, ],
                 norm_cfg=None,
                 tasks: List[Dict] = [],
                 weights=[],
                 box_coder: BoxCoder = None,
                 with_cls: bool = True,
                 with_reg: bool = True,
                 encode_background_as_zeros: bool = True,
                 use_sigmoid_score: bool = True,
                 loss_norm: Dict = dict(type="NormByNumPositives",
                                        pos_class_weight=1.0,
                                        neg_class_weight=1.0,),
                 loss_cls: Dict = dict(type="CrossEntropyLoss",
                                       use_sigmoid=False,
                                       loss_weight=1.0,),
                 loss_bbox: Dict = dict(type="SmoothL1Loss",
                                        beta=1.0,
                                        loss_weight=1.0,),
                 atten_res: Sequence[int] = None,
                 assign_cfg: Optional[dict] = dict(),
                 name="rpn",):
        super().__init__()

        assert with_cls or with_reg

        # read tasks and analysis the classes for tasks
        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.num_anchor_per_locs = [1] * len(num_classes)
        self.targets = tasks

        # define the essential paramters
        self.box_coder = box_coder # "ground_box3d_coder_anchor_free"
        self.with_cls = with_cls # True
        self.with_reg = with_reg # True
        self.in_channels = in_channels # 384
        self.num_classes = num_classes # [1, 2, 2, 1, 2, 2]
        self.encode_background_as_zeros = encode_background_as_zeros # True
        self.use_sigmoid_score = use_sigmoid_score # True
        self.box_n_dim = self.box_coder.n_dim # 10
        self.mode = mode # ‘bev’
        self.assign_cfg = assign_cfg
        self.pc_range = np.asarray(self.box_coder.pc_range)  # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.dims = self.pc_range[3:] - self.pc_range[:3]  # [3]

        # initialize loss
        self.loss_norm = loss_norm # "NormByNumPositives"
        self.loss_cls = build_loss(loss_cls) # SigmoidFocalLoss
        self.loss_reg = build_loss(loss_bbox) # "WeightedL1Loss"
        self.atten_res = atten_res # (40, 40)

        # initialize logger
        logger = logging.getLogger("MultiGroupHead")
        self.logger = logger

        # check box_coder
        assert isinstance(
            box_coder, GroundBox3dCoderAF), "OHSLoss must comes with an anchor-free box coder"
        assert box_coder.code_size == len(
            loss_bbox.code_weights), "code weights does not match code size"

        # set multi-tasks heads
        # split each head
        num_clss = []
        num_preds = []
        box_code_sizes = [self.box_coder.n_dim] * len(self.num_classes)
        for num_c, num_a, box_cs in zip(
                self.num_classes, self.num_anchor_per_locs, box_code_sizes
        ): # [1,2,2,1,2,2]和[1,1,1,1,1,1]和[10,10,10,10,10,10]
            if self.encode_background_as_zeros:
                num_cls = num_a * num_c # [1,2,2,1,2,2]
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)

            num_pred = num_a * box_cs
            num_preds.append(num_pred) # [10,10,10,10,10,10]

        self.logger.info(
            f"num_classes: {self.num_classes}, num_preds: {num_preds}"
        )

        # construct each task head
        self.tasks = nn.ModuleList()
        for task_id, (num_pred, num_cls) in enumerate(zip(num_preds, num_clss)):
            self.tasks.append(
                OHSHeadClear(
                    self.box_coder, # "ground_box3d_coder_anchor_free"
                    self.in_channels, # 64
                    num_pred, # [10,10,10,10,10,10]
                    num_cls, # [1,2,2,1,2,2]
                    header=False, # 64
                    mode=self.mode, # 'bev'
                )
            )

    def set_train_cfg(self, cfg):
        self.ohs_loss = []
        self.atten_loss = []
        for task_id, target in enumerate(self.targets):
            self.ohs_loss.append(
                OHSLossClear(self.box_coder,
                             target.num_class,
                             self.loss_cls,
                             self.loss_reg,
                             self.encode_background_as_zeros,
                             cfg,
                             self.loss_norm,
                             task_id,
                             self.mode))
            self.atten_loss.append(
                AttentionConstrainedLoss(
                    self.box_coder, target.num_class, task_id, self.atten_res)
            )
        self.logger.info("Finish Attention Constrained Loss Initialization")
        self.logger.info("Finish MultiGroupOHSHeadClear Initialization")

    def forward(self, x, return_loss=False):
        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x, return_loss)) # 返回6个task{(4, 160, 160, 10)和(4, 160, 160, 1)} 

        return ret_dicts

    def loss(self, example, preds_dicts, **kwargs):
        annos = example["annos"] # {gt_boxes, gt_names, gt_classes}
        batch_size_device = example["num_voxels"].shape[0] # 4
        batch_labels = [anno["gt_classes"] for anno in annos]
        batch_boxes = [anno["gt_boxes"] for anno in annos]
        batch_atten_map = kwargs.get('atten_map', None) # List[Tensor](4, 1600, 480)

        rets = []
        # 逐个任务计算loss
        for task_id, preds_dict in enumerate(preds_dicts):
            box_preds = preds_dict["box_preds"] # (4, 160, 160, 10)
            cls_preds = preds_dict["cls_preds"] # (4, 160, 160, 1)
            bs_per_gpu = len(cls_preds) # 4

            batch_task_boxes = [batch_box[task_id] for batch_box in batch_boxes] # 获取该任务对应的gt bbox --> List[4]
            batch_task_labels = [batch_label[task_id] for batch_label in batch_labels] # 获取该任务对应的gt label --> List[4]

            attention_loss = defaultdict(list) # 初始化attention loss
            # 逐个处理sem和geo的attenation loss
            for index, bam in enumerate(batch_atten_map):
                temp_attention_loss = self.atten_loss[task_id](
                    bam, batch_task_boxes, batch_task_labels)
                for ke, va in temp_attention_loss.items():
                    attention_loss[ke].append(va)
            
            # 为该任务assign target
            """
                6个list，每个list中包含4个item，表示一帧的信息
                labels: (160, 160)
                label_weights: (160, 160)
                bbox_targets: (324, 3)
                bbox_locs: (324, 10)
                num_pos: eg:324
                num_neg: eg:25276
            """
            targets = self.assign_hotspots(cls_preds, # (4, 160, 160, 1)
                                            batch_task_boxes, # 4个list
                                            batch_task_labels)
            labels, label_weights, bbox_targets, bbox_locs, num_total_pos, num_total_neg = targets

            # process assign targets
            labels = torch.stack(labels, 0).view(bs_per_gpu, -1)  # [B, H*W] --> (4, 160*160)
            label_weights = torch.stack(label_weights, 0).view(bs_per_gpu, -1)  # [B, H*W] --> (4, 25600)

            kwargs = {}

            # calculate ohs loss for each task
            loc_loss, cls_loss = self.ohs_loss[task_id](
                box_preds, # (4, 160, 160, 10)
                cls_preds, # (4, 160, 160, 1)
                labels, # (4, 25600)
                label_weights, # (4, 25600)
                bbox_targets, # List[Tensor]
                bbox_locs, # List[Tensor]
                **kwargs
            ) # (1, 651, 10)和(4, 25600, 1)

            if self.loss_norm["type"] == "NormByNumExamples":
                normalizer = num_total_pos + num_total_neg
            elif self.loss_norm["type"] == "NormByNumPositives":
                normalizer = max(num_total_pos, 1.0) # 651
            elif self.loss_norm["type"] == "NormByNumPosNeg":
                normalizer = self.loss_norm["pos_cls_weight"] * num_total_pos + \
                    self.loss_norm["neg_cls_weight"] * num_total_neg
            elif self.loss_norm["type"] == "dont_norm":  # support ghm loss
                normalizer = batch_size_device
            else:
                raise ValueError(f"unknown loss norm type")
            loc_loss_reduced = loc_loss.sum() / normalizer # eg: 7.5018
            loc_loss_reduced *= self.loss_reg._loss_weight # *0.25

            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels, label_weights)
            cls_pos_loss /= self.loss_norm["pos_cls_weight"] # 正例分类损失
            cls_neg_loss /= self.loss_norm["neg_cls_weight"] # 负例分类损失
            cls_loss_reduced = cls_loss.sum() / normalizer # 23697.7891 / 651 = 36.4021
            cls_loss_reduced *= self.loss_cls._loss_weight # 

            loss = loc_loss_reduced + cls_loss_reduced # 定位损失+类别损失
            # atten loss
            atten_loss = 0.0
            # sem和geo loss
            for value in attention_loss.values():
                if type(value) == list:
                    temp_loss = 0.0
                    norm_fac = len(value)
                    for temp_atten_loss in value:
                        temp_loss = temp_loss + temp_atten_loss # loss累加
                    value = temp_loss * 1.0 / norm_fac # reduce
                atten_loss = atten_loss + value # eg: -5.0781
            loss = loss + atten_loss # eg:38.2775


            loc_loss_elem = [
                loc_loss[:, :, i].sum() / num_total_pos
                for i in range(loc_loss.shape[-1])
            ] # (1, 651, 10) --> List[10]
            ret = {
                "loss": loss,
                "cls_pos_loss": cls_pos_loss.detach().cpu(),
                "cls_neg_loss": cls_neg_loss.detach().cpu(),
                "cls_loss_reduced": cls_loss_reduced.detach().cpu().mean(),
                "loc_loss_reduced": loc_loss_reduced.detach().cpu().mean(),
                "loc_loss_elem": [elem.detach().cpu() for elem in loc_loss_elem],
                "num_pos": torch.tensor([num_total_pos]),
                "num_neg": torch.tensor([num_total_neg]),
            }
            for key, value in attention_loss.items():
                if type(value) == list:
                    temp_loss = 0.0
                    norm_fac = len(value)
                    for temp_atten_loss in value:
                        temp_loss = temp_loss + temp_atten_loss
                    value = temp_loss * 1.0 / norm_fac
                ret.update({key: value.detach().cpu()})
            rets.append(ret)
        # 6个task，进行合并
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    def assign_hotspots(self,
                        cls_scores: torch.Tensor,
                        gt_bboxes: List[np.ndarray],
                        gt_labels: List[np.ndarray]):
        """
        assign hotspots(generate targets)
        Args:
            cls_scores (torch.Tensor, [B, H, W, C]): classification prediction score map
            gt_bboxes (List[np.ndarray], [[M, ndim], [K, ndim], ...]): ground truth bounding box for each batch
            gt_labels (List[np.ndarray], [[M], [K], ...]): ground truth bounding box id for each batch
            cls_scores (torch.Tensor, [B, H, D, C], optional): classification prediction score map for RV.
                Default to None.
        """
        bs_per_gpu = len(gt_bboxes)  # Get the batch size 4

        device = cls_scores.device  # Get the current device
        gt_bboxes = [torch.tensor(box, device=device).float()
                     for box in gt_bboxes]  # [M, 9], all gt_boxes List(Tensor) 4个Tensor
        # [M] all gt_classes,start from 1,( 0 means background)
        gt_labels = [torch.tensor(label, device=device).long() for label in gt_labels] # 和gt_bboxes相同

        # 逐帧assign
        """
            6个list，每个list中包含4个item，表示一帧的信息
            labels: (160, 160)
            label_weights: (160, 160)
            bbox_targets: (324, 3)
            bbox_locs: (324, 10)
            num_pos: eg:324
            num_neg: eg:25276
        """
        labels_list, label_weights_list, bbox_targets_list, bbox_locs_list, num_pos_list, num_neg_list = \
            multi_apply(self.assign_hotspots_bev_single, cls_scores, gt_bboxes, gt_labels)
        
        # 逐帧处理，在bbox_locs_list的头部添加batch索引
        for i in range(bs_per_gpu):
            bbox_locs_list[i][:, 0] = i

        num_total_pos = sum([max(num, 1) for num in num_pos_list]) # 651
        num_total_neg = sum([max(num, 1) for num in num_neg_list]) # 101749

        targets = (labels_list, label_weights_list, bbox_targets_list,
                   bbox_locs_list, num_total_pos, num_total_neg)

        return targets

    def assign_hotspots_bev_single(self,
                                   cls_scores: torch.Tensor,
                                   gt_bboxes: torch.Tensor,
                                   gt_labels: torch.Tensor):
        r"""
        assign hotspots(generate targets) of BEV for a single batch.
        Args:
            cls_scores (torch.Tensor, [H, W, C]): classification prediction score map
            gt_bboxes (torch.Tensor, [M, ndim]): ground truth bounding box
            gt_labels_list (torch.Tensor, [M]): ground truth bounding box id
        """
        h, w = cls_scores.size()[:2]  # Get the size of the feature map of bev view (160, 160)

        # initialize relate labels 全0
        labels = torch.zeros_like(cls_scores[:, :, 0], dtype=torch.long)  # Set up the bev labels (160, 160)
        # Initialize all weights to neg weights--> (160, 160) 全部为2
        label_weights = torch.ones_like(
            cls_scores[:, :, 0], dtype=torch.float) * self.loss_norm["neg_cls_weight"] 
        # initialized to record the positive bbx's location in grid map
        bbox_locs = cls_scores.new_zeros((0, 3), dtype=torch.long) # 空tensor
        # initialized to record the positive bbx's regression targets
        bbox_targets = cls_scores.new_zeros((0, self.box_coder.code_size), dtype=torch.float) # 空tensor

        # scan gt_bboxes
        self.effective_ratio = self.assign_cfg.get("effective_ratio", [1.0, 6.0]) # [1.0, 6.0]
        if len(gt_bboxes > 0):
            effective_boxes = gt_bboxes[:, [0, 1, 3, 4]].clone().detach()  # [M, 4]
            effective_ratio_l = (self.dims[0] / w) / effective_boxes[:, 2]  # [M]
            effective_ratio_w = (self.dims[1] / h) / effective_boxes[:, 3]  # [M]
            effective_ratio_l = effective_ratio_l.clamp(min=self.effective_ratio[0],  # [M]
                                                        max=self.effective_ratio[1])  # [M]
            effective_ratio_w = effective_ratio_w.clamp(min=self.effective_ratio[0],  # [M]
                                                        max=self.effective_ratio[1])  # [M]

            # expand the box'area into a grid if the box is too small,
            # so that this box label can match the center of the correspond box
            # the expanded box called `effective_boxes`
            effective_boxes[:, 2] *= effective_ratio_l
            effective_boxes[:, 3] *= effective_ratio_w

            # get the corners
            angles = gt_bboxes[:, -1]  # [num_box]
            effective_boxes = box_torch_ops.center_to_corner_box2d(
                effective_boxes[:, :2], effective_boxes[:, 2:4], angles)
            ignore_boxes_out = effective_boxes # (17, 4, 2)

            # transfer the hybrid coordinate system to Cartesian coordinate system
            self.box_coder.layout(w, h)

            # read necessary parameters from box_coder
            # center cartesian coordinate, grid coordinate index in hybrid coordinate
            # grid_real_centers - [W * H, 2]
            # w_indices - [W * H]
            # h_indices - [W * H]
            grid_real_centers = self.box_coder.grids_sensor # (25600, 2)
            w_indices = self.box_coder.ww_l # (25600,)
            h_indices = self.box_coder.hh_l # (25600,)

            # scan bounding boxes 逐个bbox处理
            for i in range(len(gt_bboxes)):
                # get the points(hotspots) cover by the bounding box
                pos_mask = points_in_convex_polygon_torch(
                    grid_real_centers, effective_boxes[i].unsqueeze(0))  # (25600, 1)

                # get the raw hotspots
                pos_ind = pos_mask.nonzero()[:, 0] # eg:(19,)

                # NOTE: fix the bugs of targets assignment in bev, while using hybird coordinates,
                #       the `effective_boxes` may not expand enough to cover a grid center,
                #       so we nearest search a grid center as hotspots for this situation
                gt_center = gt_bboxes[i: i + 1, :2]  # (1, 2)
                dist_to_grid_center = torch.norm(grid_real_centers - gt_center, dim=1)  # [W * H]-->(25600,)
                min_ind = torch.argmin(dist_to_grid_center)
                if min_ind not in pos_ind:
                    pos_ind = torch.cat([pos_ind.reshape(-1, 1), min_ind.reshape(-1, 1)],
                                        dim=0).reshape(-1)

                num_hotspots = self.assign_cfg.get("num_hotspots", 28) # 28
                if self.assign_cfg.get("select_hotspots", True): # 选择索引
                    # filter out the verbose hotspots
                    if len(pos_ind) > num_hotspots: # 如果预测gt超过预订数量，则直接取前28个
                        # if the hotspots are too many for the instance
                        # select the num_hotspots-th nearest as valid hotspots
                        points = grid_real_centers[pos_ind, :]
                        diff = gt_bboxes[i, :2] - points
                        diff = torch.norm(diff, dim=1)
                        sorted_ind = torch.argsort(diff)[:num_hotspots]
                        pos_ind = pos_ind[sorted_ind]

                # get the indices of hotspots
                pos_h_indices = h_indices[pos_ind]  # [num_pos] (19,)
                pos_w_indices = w_indices[pos_ind]  # [num_pos] (19,)

                # scan the positive hotspots
                if len(pos_h_indices):
                    if not (labels[pos_h_indices, pos_w_indices] == 0).all():
                        unique_pos_h_indices = pos_h_indices.new_zeros((0,))
                        unique_pos_w_indices = pos_w_indices.new_zeros((0,))
                        unique_pos_ind = pos_ind.new_zeros((0,))
                        # NOTE: assert that each grid's gradient just be affected by one label
                        #       if a grid was covered by other label, eliminate its effects
                        for ph, pw, pi in zip(pos_h_indices, pos_w_indices, pos_ind):
                            if labels[ph, pw] == 0:
                                unique_pos_h_indices = torch.cat(
                                    (unique_pos_h_indices, ph.view((1))))
                                unique_pos_w_indices = torch.cat(
                                    (unique_pos_w_indices, pw.view((1))))
                                unique_pos_ind = torch.cat((unique_pos_ind, pi.view((1))))
                            else:
                                label_weights[ph, pw] = 0
                        pos_h_indices = unique_pos_h_indices
                        pos_w_indices = unique_pos_w_indices
                        pos_ind = unique_pos_ind

                    # fullfill `labels` and `label_weights`
                    labels[pos_h_indices, pos_w_indices] = gt_labels[i] # 为该box内的点分配标签
                    label_weights[pos_h_indices, pos_w_indices] = self.loss_norm["pos_cls_weight"]

                # get the overlap hotspots and set the `label_weights` as 0
                ig_mask = points_in_convex_polygon_torch(
                    grid_real_centers, ignore_boxes_out[i].unsqueeze(0)) # （25600, 1）
                ig_mask = (ig_mask & (~pos_mask)).reshape(-1)  # Get the overlapped grid
                ig_h = h_indices[ig_mask] # 一般为空
                ig_w = w_indices[ig_mask]
                # 1 for hspots in gtbbx, 0 for non-hspots in gtbbx
                label_weights[ig_h, ig_w] = 0

                centers = grid_real_centers[pos_ind] # eg:(19, 2) 取出box内正例点的中心
                shifts = torch.zeros((len(centers), self.box_coder.code_size),
                                     device=cls_scores.device,
                                     dtype=torch.float) # eg:(19, 10)
                # Got the encode bbx target for each positive grid
                shifts = self.box_coder._encode(centers, shifts, gt_bboxes[i]) # eg:(19, 10)

                zeros = torch.zeros_like(pos_w_indices) # (19,)
                locs = torch.stack((zeros, pos_h_indices, pos_w_indices), dim=-1) # （19, 3)

                # get the corresponding bounding boxes
                bbox_locs = torch.cat((bbox_locs, locs), dim=0) # 组合bbox内pos点的索引(all bbox) --> (324, 3)
                bbox_targets = torch.cat((bbox_targets, shifts), dim=0) # 组合bbox内pos的target(all bbox) --> (324, 10)

        # get the ratio os positive and negative examples
        num_pos = bbox_targets.size(0) # 324
        num_neg = label_weights.nonzero().size(0) - num_pos # 25276
        """
            labels: (160, 160)
            label_weights: (160, 160)
            bbox_targets: (324, 3)
            bbox_locs: (324, 10)
            num_pos: eg:324
            num_neg: eg:25276
        """
        return (labels, label_weights, bbox_targets, bbox_locs, num_pos, num_neg)

    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        rets = []
        double_flip = test_cfg.get('double_flip', False) # false
        # 逐个task进行预测
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size_device = example['num_voxels'].shape[0] # 4
            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size_device
            else:
                meta_list = example["metadata"]

            if double_flip:
                assert batch_size_device % 4 == 0, f"batch_size_device: {batch_size_device}"
                batch_size_device = int(batch_size_device / 4)
                meta_list = meta_list[:4 * int(batch_size_device):4]
                batch_box_preds_all = preds_dict["box_preds"]
                batch_cls_preds_all = preds_dict["cls_preds"]
                _, H, W, C = batch_box_preds_all.shape
                batch_box_preds_all = batch_box_preds_all.reshape(
                    int(batch_size_device), 4, H, W, C)
                batch_box_preds_sincos_all = batch_box_preds_all[:, :, :, :, 8:10].clone()
                _, H, W, C = batch_cls_preds_all.shape
                batch_cls_preds_all = batch_cls_preds_all.reshape(
                    int(batch_size_device), 4, H, W, C)

                batch_cls_preds_all[:, 1] = torch.flip(batch_cls_preds_all[:, 1], dims=[1])
                batch_cls_preds_all[:, 2] = torch.flip(batch_cls_preds_all[:, 2], dims=[2])
                batch_cls_preds_all[:, 3] = torch.flip(batch_cls_preds_all[:, 3], dims=[1, 2])
                batch_cls_preds_all = batch_cls_preds_all.sigmoid()
                batch_cls_preds = batch_cls_preds_all.mean(dim=1)

                batch_box_preds_sincos_all[:, 1] = torch.flip(
                    batch_box_preds_sincos_all[:, 1], dims=[1])
                batch_box_preds_sincos_all[:, 2] = torch.flip(
                    batch_box_preds_sincos_all[:, 2], dims=[2])
                batch_box_preds_sincos_all[:, 3] = torch.flip(
                    batch_box_preds_sincos_all[:, 3], dims=[1, 2])

                num_class_with_bg = self.num_classes[task_id]
                if not self.encode_background_as_zeros:
                    num_class_with_bg = self.num_classes[task_id] + 1

                batch_cls_preds = batch_cls_preds.contiguous()

                batch_cls_preds = batch_cls_preds.view(
                    batch_size_device, -1, num_class_with_bg)

                batch_reg_preds = torch.zeros(
                    (int(batch_size_device), 4, H * W, 9), dtype=batch_box_preds_all.dtype, device=batch_box_preds_all.device)

                for i in range(4):
                    batch_box_preds = batch_box_preds_all[:, i, :, :, :]

                    box_ndim = self.box_n_dim
                    h, w = batch_box_preds.size()[1:3]

                    batch_box_preds = batch_box_preds.contiguous()

                    batch_box_preds = batch_box_preds.view(batch_size_device, -1, box_ndim)

                    if i == 1:  # theta = pi-theta
                        batch_box_preds[:, :, -2] = -batch_box_preds[:, :, -2]
                        batch_box_preds_sincos_all[:, i, :, :, 0] = - \
                            batch_box_preds_sincos_all[:, i, :, :, 0]
                    elif i == 2:  # x=-x, theta = 2pi-theta, vx = -vx
                        batch_box_preds[:, :, -1] = -batch_box_preds[:, :, -1]
                        batch_box_preds_sincos_all[:, i, :, :, 1] = - \
                            batch_box_preds_sincos_all[:, i, :, :, 1]
                    elif i == 3:  # x=-x,y=-y, theta=theta-pi, vx=-vx, vy=-vy
                        batch_box_preds[:, :, -1] = -batch_box_preds[:, :, -1]
                        batch_box_preds[:, :, -2] = -batch_box_preds[:, :, -2]
                        batch_box_preds_sincos_all[:, i, :, :, 0] = - \
                            batch_box_preds_sincos_all[:, i, :, :, 0]
                        batch_box_preds_sincos_all[:, i, :, :, 1] = - \
                            batch_box_preds_sincos_all[:, i, :, :, 1]

                    #import pdb; pdb.set_trace()
                    # -pi/2
                    #batch_box_preds[:, :, -2], batch_box_preds[:, :, -1] = batch_box_preds[:, :, -1], -batch_box_preds[:, :, -2]
                    # # +pi/2
                    #batch_box_preds[:, :, -2], batch_box_preds[:, :, -1] = -batch_box_preds[:, :, -1], batch_box_preds[:, :, -2]
                    batch_reg_preds_temp = self.box_coder._decode(
                        batch_box_preds[:, :, :self.box_coder.code_size], w, h
                    )

                    if i == 1:  # y=-y, vy = -vy
                        batch_reg_preds_temp[:, :, 1] = -batch_reg_preds_temp[:, :, 1]
                        batch_reg_preds_temp[:, :, 7] = -batch_reg_preds_temp[:, :, 7]
                    elif i == 2:  # x=-x, vx = -vx
                        batch_reg_preds_temp[:, :, 0] = -batch_reg_preds_temp[:, :, 0]
                        batch_reg_preds_temp[:, :, 6] = -batch_reg_preds_temp[:, :, 6]
                    elif i == 3:  # x=-x,y=-y, vx=-vx, vy=-vy
                        batch_reg_preds_temp[:, :, 1] = -batch_reg_preds_temp[:, :, 1]
                        batch_reg_preds_temp[:, :, 0] = -batch_reg_preds_temp[:, :, 0]
                        batch_reg_preds_temp[:, :, 7] = -batch_reg_preds_temp[:, :, 7]
                        batch_reg_preds_temp[:, :, 6] = -batch_reg_preds_temp[:, :, 6]
                    batch_reg_preds[:, i, :, :] = batch_reg_preds_temp

                batch_box_preds_sincos_all = batch_box_preds_sincos_all.mean(dim=1)
                batch_box_preds_sincos_all = batch_box_preds_sincos_all.reshape(
                    batch_size_device, -1, 2)
                batch_box_preds_rads = torch.atan2(
                    batch_box_preds_sincos_all[:, :, 1], batch_box_preds_sincos_all[:, :, 0])

                batch_reg_preds = batch_reg_preds.reshape(batch_size_device, 4, H, W, 9)
                batch_reg_preds[:, 1] = torch.flip(batch_reg_preds[:, 1], dims=[1])
                batch_reg_preds[:, 2] = torch.flip(batch_reg_preds[:, 2], dims=[2])
                batch_reg_preds[:, 3] = torch.flip(batch_reg_preds[:, 3], dims=[1, 2])
                batch_reg_preds = batch_reg_preds.mean(dim=1)
                batch_reg_preds = batch_reg_preds.reshape(batch_size_device, -1, 9)
                batch_reg_preds[:, :, -1] = batch_box_preds_rads
            else:
                batch_box_preds = preds_dict["box_preds"] # (4, 160, 160, 10)
                batch_cls_preds = preds_dict["cls_preds"].sigmoid() # (4, 160, 160, 10) 

                box_ndim = self.box_n_dim # 10
                h, w = batch_box_preds.size()[1:3] # (160, 160)

                batch_box_preds = batch_box_preds.view(batch_size_device, -1, box_ndim) # (4, 25600, 10)

                num_class_with_bg = self.num_classes[task_id] # 1

                if not self.encode_background_as_zeros:
                    num_class_with_bg = self.num_classes[task_id] + 1

                batch_cls_preds = batch_cls_preds.contiguous() # (4, 160, 160, 10) 

                batch_cls_preds = batch_cls_preds.view(batch_size_device, -1, num_class_with_bg) # (4, 25600, 1)
                batch_reg_preds = self.box_coder._decode(
                    batch_box_preds[:, :, :self.box_coder.code_size], w, h
                )

            batch_dir_preds = [None] * batch_size_device
            rets.append(
                self.get_task_detections(
                    task_id,
                    num_class_with_bg,
                    test_cfg,
                    batch_cls_preds,
                    batch_reg_preds,
                    batch_dir_preds,
                    meta_list,
                )
            )

        num_samples = len(rets[0]) # 4
        ret_list = []
        # 从task预测整理为batch形式
        # 逐帧处理
        for i in range(num_samples):
            ret = {}
            # 逐个task处理
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == "metadata":
                    # metadata
                    ret[k] = rets[0][i][k]
            ret_list.append(ret)
        return ret_list

    def get_task_detections(
        self,
        task_id,
        num_class_with_bg,
        test_cfg,
        batch_cls_preds,
        batch_reg_preds,
        batch_dir_preds=None,
        meta_list=None,
    ):
        predictions_dicts = []
        post_center_range = test_cfg.post_center_limit_range # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds.dtype,
                device=batch_reg_preds.device,
            )
        # 逐帧处理
        for box_preds, cls_preds, dir_preds, meta in zip(
            batch_reg_preds,
            batch_cls_preds,
            batch_dir_preds,
            meta_list,
        ):

            box_preds = box_preds.float() # (25600, 9)
            cls_preds = cls_preds.float() # (25600, 1)


            if self.encode_background_as_zeros:
                # this don't support softmax
                assert self.use_sigmoid_score is True
                total_scores = cls_preds # (25600, 1)
                #total_scores = cls_preds
            else:
                # encode background as first element in one-hot vector
                if self.use_sigmoid_score:
                    total_scores = cls_preds[..., 1:] 
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            feature_map_size_prod = (
                batch_reg_preds.shape[1] // self.num_anchor_per_locs[task_id]
            )
            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1) # (25600,)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long,
                ) # (25600,) 全部为0

            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            if test_cfg.score_threshold > 0.0:
                thresh = torch.tensor(
                    [test_cfg.score_threshold], device=total_scores.device
                ).type_as(total_scores) # 0.1
                top_scores_keep = top_scores >= thresh 
                top_scores = top_scores.masked_select(top_scores_keep) # eg:(268,)

            if top_scores.shape[0] != 0:
                if test_cfg.score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    assert (box_preds[:, 3:6] > 0).cpu().numpy().any()
                    top_labels = top_labels[top_scores_keep] # eg:(268,)

                boxes_for_nms = box_torch_ops.boxes3d_to_bevboxes_lidar_torch(box_preds) # 转换BEV视角 (268, 5)

                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, top_scores,
                                                            thresh=test_cfg.nms.nms_iou_threshold, # 0.2
                                                            pre_maxsize=test_cfg.nms.nms_pre_max_size, # 1000
                                                            post_max_size=test_cfg.nms.nms_post_max_size) # 80
            else:
                selected = []
            # if selected is not None:
            selected_boxes = box_preds[selected] # (15, 9)
            selected_labels = top_labels[selected] # (15,)
            selected_scores = top_scores[selected] # (15,)

            # finally generate predictions. 在范围内处理
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes # (15, 9)
                scores = selected_scores # (15,)
                label_preds = selected_labels # (15,)
                final_box_preds = box_preds # (15, 9)
                final_scores = scores # (15,)
                final_labels = label_preds # (15,)
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    # 通过mask过滤不在设定范围内的值，一般都在
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask], # (15, 9)
                        "scores": final_scores[mask], # (15,)
                        "label_preds": label_preds[mask], # (15,)
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": final_labels,
                        "metadata": meta,
                    }
            # NMS后为空的特殊情况，构造空值处理
            else:
                dtype = batch_reg_preds.dtype
                device = batch_reg_preds.device
                predictions_dict = {
                    "box3d_lidar": torch.zeros([0, box_preds.shape[1]], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros(
                        [0], dtype=top_labels.dtype, device=device
                    ),
                    "metadata": meta,
                }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts
