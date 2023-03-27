import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, PseudoSampler)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult
from projects.mmdet3d_plugin.models.utils.decoder_utils import ImageRCNNBlock, PointRCNNBlock, PositionEmbeddingLearned, TransformerDecoderLayer, FFN
import pdb


@HEADS.register_module()
class DeepInteractionDecoder(nn.Module):
    def __init__(self,
                 num_views=0,
                 out_size_factor_img=4,
                 num_proposals=128,
                 auxiliary=True,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_mmpi=4,
                 num_decoder_layers=1,
                 num_heads=8,
                 learnable_query_pos=False,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 ret_idx=None,
                 ):
        super(DeepInteractionDecoder, self).__init__()

        self.num_classes = num_classes # 10类
        self.num_proposals = num_proposals # 200
        self.auxiliary = auxiliary # True
        self.num_heads = num_heads # 8
        self.num_decoder_layers = num_decoder_layers # 1
        self.bn_momentum = bn_momentum # 0.1
        self.learnable_query_pos = learnable_query_pos # False
        self.initialize_by_heatmap = initialize_by_heatmap # True
        self.nms_kernel_size = nms_kernel_size # 3
        if self.initialize_by_heatmap is True:
            assert self.learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False) # True
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls) # FocalLoss
        self.loss_bbox = build_loss(loss_bbox) # L1Loss
        self.loss_heatmap = build_loss(loss_heatmap) # GaussianFocalLoss

        self.bbox_coder = build_bbox_coder(bbox_coder) # TransFusionBBoxCoder
        self.sampling = False

        if self.initialize_by_heatmap:
            layers = []
            layers.append(ConvModule(
                hidden_channel, # 128
                hidden_channel, # 128
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
            layers.append(build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel, # 128
                num_classes, # 10
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
            self.heatmap_head = nn.Sequential(*layers) # 128-->128-->10
            self.heatmap_head_img = copy.deepcopy(self.heatmap_head)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1) # 10 --> 128
        else:
            # query feature
            self.query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals)) # (1, 128, 200)
            self.query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=learnable_query_pos) # (1, 200, 2)

        # transformer decoder layers for object query with LiDAR feature --> transformer decoder(layersequence)
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation, # 128, 8, 256, 0.1, relu
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel), # 2, 128
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel), # 2, 128
                ))

        # Prediction Head --> init_layer
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads) # dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs))) # 10, 2
            self.prediction_heads.append(FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)) # 128
        # ---------------------------------------------------------
        # DeepInteration特有
        # ---------------------------------------------------------
        self.decode_head = nn.ModuleList()
        self.pred_head = nn.ModuleList()
        self.num_mmpi = num_mmpi # 4
        assert self.num_mmpi % 2 == 0
        self.num_views = num_views # 6
        self.out_size_factor_img = out_size_factor_img # 4
        for i in range(int(self.num_mmpi/2)): # 交叉进行，故/2
            heads = copy.deepcopy(common_heads) # dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            # 图像解码头
            self.decode_head.append(
                ImageRCNNBlock(
                    self.num_views, self.num_proposals, self.out_size_factor_img, self.test_cfg, self.bbox_coder, 
                    hidden_channel, num_heads, dropout
                )
            )
            self.pred_head.append(FFN(hidden_channel * 2, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)) # 256(预测头)
            # 点云解码头
            self.decode_head.append(
                PointRCNNBlock(
                    hidden_channel, num_heads, dropout, self.bbox_coder
                )
            )
            self.pred_head.append(FFN(hidden_channel * 2, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)) # 256(预测头)
        # -------------------------DeepInteration特有----------------------

        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'] # 180
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'] # 180
        self.bev_pos = self.create_2D_grid(x_size, y_size) # (1, 32400, 2)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

        self.init_weights()
        self._init_assigner_sampler() # 初始化assigner和sampler-->HungarianAssigner3D和PseudoSampler
        
        self.ret_idx = ret_idx
                

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]] # [0, 179, 180]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]) # (180, 180)
        batch_x = batch_x + 0.5 # 取中心点
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None] # (1, 2, 180, 180)
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1) # (1, 2, 32400) --> (1, 32400, 2)
        return coord_base # (1, 32400, 2)

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling: # False
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler() # 直接返回正例和负例的索引，伪采样
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner) # HungarianAssigner3D
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward(self, pts_inputs, img_inputs, img_metas):
        """Forward function for CenterPoint.

        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)

        Returns:
            list[dict]: Output results for tasks.
        """
        lidar_feat = pts_inputs[0] # (1, 128, 180, 180)
        batch_size = lidar_feat.shape[0] # 1
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W] -->(1, 128, 180*180)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device) # (1, 32400, 2)

        img_feat = img_inputs  # [BN, C, H, W]-->(6, 128, 112, 200)
        BN, I_C, I_H, I_W = img_feat.shape # 6, 128, 112, 200
        img_h = I_H # 112
        img_w = I_W # 200
        new_lidar_feat = pts_inputs[1] # (1, 128, 180, 180)
        bev_feat = new_lidar_feat.view(batch_size, new_lidar_feat.shape[1], -1) # (1, 128, 32400)
        
        dense_heatmap = self.heatmap_head(lidar_feat) # (1, 10, 180, 180)
        dense_heatmap_img = self.heatmap_head_img(bev_feat.view(lidar_feat.shape))  # (1, 128, 180, 180)-->(1, 10, 180, 180)[BS, num_classes, H, W]
        heatmap = (dense_heatmap.detach().sigmoid() + dense_heatmap_img.detach().sigmoid()) / 2 # 取二者平均值 (1, 10, 180, 180)
        padding = self.nms_kernel_size // 2 # 3 // 2 = 1
        local_max = torch.zeros_like(heatmap) # (1, 10, 180, 180) 全0初始化
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0) # (1, 10, 178, 178)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner # (1, 10, 180, 180) max后的最值
        # for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max) # 巧妙处理
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1) # (1, 10, 32400)

        # top #num_proposals among all classes (1, 324000) --> (1, 200)
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals] # 取出前200个最大值
        top_proposals_class = top_proposals // heatmap.shape[-1] # (1, 200) 分类类别
        top_proposals_index = top_proposals % heatmap.shape[-1] # (1, 200) 表示预测该类的特征图索引(那个特征预测了该类)
        # 根据索引在lidar的bev特征图中抽取query的特征
        # lidar_feat_flatten:(1, 128, 32400)   index:(1, 1, 200)-->(1, 128, 200)   query_feat:(1, 128, 200)
        query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
        # 赋值query的分类类别
        self.query_labels = top_proposals_class # (1, 200) 分类类别

        # add category embedding 将query的类别进行one hot编码，然后计算类别嵌入特征
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1) # (1, 200, 10)-->(1, 10, 200)
        query_cat_encoding = self.class_encoding(one_hot.float()) # 10-->128 : (1, 128, 200)
        query_feat += query_cat_encoding # (1, 128, 200) # 使得query特征融合类别特征
        # bev_pos:(1, 32400, 2)  index:(1, 1, 200)-->(1, 200, 1)-->(1, 200, 2) query_pos:(1, 200, 2)
        query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            # Transformer Decoder Layer
            # query_feat: (1, 128, 200)  
            # lidar_feat_flatten: (1, 128, 32400)
            # query_pos: B Pq 2 (1, 200, 2)
            # bev_pos: (1, 32400, 2)
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)

            # Prediction
            # {'center':(1, 2, 200), 'height':(1, 1, 200), 'dim':(1, 3, 200), 'rot':(1, 2, 200),'vel':(1, 2, 200),'heatmap':(1, 10, 200)}
            res_layer = self.prediction_heads[i](query_feat) # (1, 128, 200)-->dict(6个head输出)
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1) # 加特征图上的点坐标，预测为偏移量 (1, 200, 2) 这里体现了query的重要性
            first_res_layer = res_layer

            # for next level positional embedding 
            # 如果lidar transformer为多层，则为下一层的位置编码做准备，同时也是lidar和image融合的位置编码
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)
        
        
        # (6, 128, 112, 200)-->(1, 6, 128, 22400) image信息
        img_feat_flatten = img_feat.view(batch_size, self.num_views, img_feat.shape[1], -1)
        
        if self.img_feat_pos is None:
            (h, w) = img_inputs.shape[-2], img_inputs.shape[-1]
            img_feat_pos = self.img_feat_pos = self.create_2D_grid(h, w).to(img_feat_flatten.device) # (1, 22400, 2)
        else:
            img_feat_pos = self.img_feat_pos
        
        self.on_the_image_mask = []
        
        # -----------------------------------------------
        #              MMPI-LiDAR和MMPI-image
        # -----------------------------------------------
        for layer_idx in range(self.num_mmpi):
            if layer_idx == 0:
                prev_query_feat = query_feat.detach().clone() # (1, 128, 200)
            if layer_idx == 1:
                prev_query_feat = query_feat.detach().clone()
            if layer_idx == 2:
                pass
            if layer_idx == 3:
                prev_query_feat = query_feat.detach().clone()
            # (1, 2, 200)-->(1, 200, 2)
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1) # 记录上一层的中心点坐标(1, 200, 2)
            if layer_idx != 2:
                query_feat, on_the_image = self.decode_head[layer_idx](
                    query_feat=query_feat, res_layer=res_layer, 
                    new_lidar_feat=new_lidar_feat, bev_pos=bev_pos, img_feat_flatten=img_feat_flatten, img_feat_pos=img_feat_pos, 
                    img_metas=img_metas, img_h=img_h, img_w=img_w
                )
            else:
                query_feat, on_the_image = self.decode_head[layer_idx](
                    query_feat=prev_query_feat, res_layer=res_layer, 
                    new_lidar_feat=new_lidar_feat, bev_pos=bev_pos, img_feat_flatten=img_feat_flatten, img_feat_pos=img_feat_pos, 
                    img_metas=img_metas, img_h=img_h, img_w=img_w
                )
            # 两个query_feat合并进行预测
            res_layer = self.pred_head[layer_idx](torch.cat([query_feat, prev_query_feat], dim=1))
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1) # (1, 200, 2)
            layer_mask = on_the_image != -1 if layer_idx % 2 == 0 else self.on_the_image_mask[-1]
            self.on_the_image_mask.append(layer_mask) # 将该层的mask加入list
            # 逐个head进行处理，将没有进行增强的query使用原始lidar的预测值
            for key, value in res_layer.items():
                pred_dim = value.shape[1]
                # repeat dim次是组合对应维度的mask
                res_layer[key][~self.on_the_image_mask[-1].unsqueeze(1).repeat(1, pred_dim, 1)] = first_res_layer[key][~self.on_the_image_mask[-1].unsqueeze(1).repeat(1, pred_dim, 1)] # (1, 1, 200)
            ret_dicts.append(res_layer) # 将该层的result加入list
        # -----------------------------------------------
        #              MMPI-LiDAR和MMPI-image
        # -----------------------------------------------

        if self.initialize_by_heatmap:
            # heatmap:(1, 10, 32400)
            # top_proposals_index:(1, 200) --> (1, 1, 200) --> (1, 10, 200) 取出query在heatmap上的分数 对heatmap的生成做约束
            # query在BEV heatmap上的特征
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            ret_dicts[0]['dense_heatmap'] = dense_heatmap_img # (1, 10, 180, 180) 记录dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1) # 如果是多层，会在这里进行拼接，如果只取最后一层不进行拼接
            else:
                new_res[key] = ret_dicts[0][key]
        return [[new_res]] # {'center':(1, 2, 800), 'height':(1, 1, 800),..., 'dense_heatmap':(1, 10, 180, 180)}

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        # 逐帧处理, 将pred_dict按帧进行重新整理
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            # 逐个head处理
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1] # 取出该帧的预测结果
            list_of_pred_dict.append(pred_dict) # 将该帧结果加入list

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)
        
        # 逐帧assign target
        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, np.arange(len(gt_labels_3d)))
        # 将各target结果进行拼接
        labels = torch.cat(res_tuple[0], dim=0) # eg:(1, 200) --> (bs, num_proposal)
        label_weights = torch.cat(res_tuple[1], dim=0) # eg:(1, 200)
        bbox_targets = torch.cat(res_tuple[2], dim=0) # eg:(1, 200, 10)
        bbox_weights = torch.cat(res_tuple[3], dim=0) # eg:(1, 200, 10)
        ious = torch.cat(res_tuple[4], dim=0) # eg: (1, 200)
        num_pos = np.sum(res_tuple[5]) # eg:27
        matched_ious = np.mean(res_tuple[6]) # eg:0.0339
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0) # (1, 10, 180, 180)
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap
        else:
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        # ----------------------------------
        # 1.拷贝预测输出并对预测bbox进行编码
        # ----------------------------------
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach()) # (1, 10, 200)
        center = copy.deepcopy(preds_dict['center'].detach()) # (1, 2, 200)
        height = copy.deepcopy(preds_dict['height'].detach()) # (1, 1, 200)
        dim = copy.deepcopy(preds_dict['dim'].detach()) # (1, 3, 200)
        rot = copy.deepcopy(preds_dict['rot'].detach()) # (1, 2, 200)
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach()) # (1, 2, 200)
        else:
            vel = None

        # decode the prediction to real world metric bbox
        # boxes_dict:{'bboxes', 'scores', 'labels'}
        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        
        # ------------------------------------------------
        # 2.根据pred boxes和gt boxes进行target assign和sample
        # ------------------------------------------------
        bboxes_tensor = boxes_dict[0]['bboxes'] # (200, 9)
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device) # eg:(78, 9)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_mmpi
        else:
            num_layer = 1
        
        # ------------------------------------
        # 2.1 assign target
        # ------------------------------------
        assign_result_list = []
        # 逐层分配
        for idx_layer in range(num_layer):
            # pred boxes的几何参数和分类分数
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :] # (200, 9)
            score_layer = score[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)] # (1, 10, 200)

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]), # 合并gt的数量
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]), # 合并gt inds
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]), # 合并iou
            labels=torch.cat([res.labels for res in assign_result_list]), # 合分配的lables
        )

        # ------------------------------------
        # 2.2 pseudo sampler
        # ------------------------------------
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds # 预测bbox中正例bbox的索引 eg:78
        neg_inds = sampling_result.neg_inds # 预测bbox中负例bbox的索引 eg:112
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # ------------------------------------
        # 3.create target for loss computation
        # ------------------------------------
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device) # (200, 10)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device) # (200, 10)
        ious = assign_result_ensemble.max_overlaps # (200,)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) # (200,)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) # (200,)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes # labels全部为初始化为10

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            # 对gt进行编码
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets # 将编码后的gt box赋予pos inds的对应位置
            bbox_weights[pos_inds, :] = 1.0 # 将权重设置为1

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0 # 将正例的权重设置为1
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0 # 此时lable_weights全1

        # ----------------------------------
        # 4.compute dense heatmap targets
        # ----------------------------------
        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device) # (78, 10)
            grid_size = torch.tensor(self.train_cfg['grid_size']) # (1440, 1440, 40)
            pc_range = torch.tensor(self.train_cfg['point_cloud_range']) # [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
            voxel_size = torch.tensor(self.train_cfg['voxel_size']) # [0.075, 0.075, 0.2]
            feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len] (180, 180)
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0]) # 初始化(10, 180, 180)
            # 逐个gt bbox处理,计算heatmap
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3] # bbox的实际宽
                length = gt_bboxes_3d[idx][4] # bbox的实际长
                width = width / voxel_size[0] / self.train_cfg['out_size_factor'] # bbox的特征图宽
                length = length / voxel_size[1] / self.train_cfg['out_size_factor'] # bbox的特征图长
                if width > 0 and length > 0:
                    # 根据bbox的长和宽计算高斯半径
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap']) # gaussian_overlap:0.1
                    radius = max(self.train_cfg['min_radius'], int(radius)) # min_radius:2
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1] # gt bbox的实际中心坐标

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor'] # 计算bbox的特征中心坐标
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    # 在该类别所在的heatmap上根据中心点和半径绘制gaussian圆
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1) # 计算iou的平均值
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]

        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        if self.initialize_by_heatmap:
            # labels: (1, 800)
            # label_weights: (1, 800)
            # bbox_targets: (1, 800, 10)
            # bbox_weights: (1, 800, 10)
            # ious: (1, 800)
            # num_pos: eg:204
            # matched_ious: eg:0.0346
            # heatmap: (1, 10, 180, 180)
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        else:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])

        # 逐层处理，对于找不到投影的box将weight设置为0
        num_pos = [] # eg:[51, 51, 51, 51]
        if hasattr(self, 'on_the_image_mask'):
            for idx_layer in range(self.num_mmpi):
                if idx_layer%2==0:
                    label_weights[...,idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals] = label_weights[...,idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals] * self.on_the_image_mask[idx_layer // 2]
                    bbox_weights[:,idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals,:] = bbox_weights[:,idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals,:] * self.on_the_image_mask[idx_layer // 2][:, :, None]
                num_pos.append(bbox_weights.max(-1).values[...,idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].sum())
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # -------------------------------------
        # image guided query initialization 注释
        # -------------------------------------
        if self.initialize_by_heatmap:
            # compute heatmap loss GaussianFocalLoss
            # preds_dict['dense_heatmap']: (1, 10, 180, 180)
            # heatmap: (1, 10, 180, 180)
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer 逐层计算损失
        for idx_layer in range(self.num_mmpi):
            prefix = f'layer_{idx_layer}'
            
            # 计算分类损失
            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            # 进入分类损失函数 focal loss
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos[idx_layer], 1))

            # 计算回归损失
            layer_center = preds_dict['center'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_height = preds_dict['height'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_rot = preds_dict['rot'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_dim = preds_dict['dim'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None) # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            layer_bbox_weights = bbox_weights[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            # 进入回归损失函数 L1 loss
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos[idx_layer], 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls # 记录分类损失
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox # 记录回归损失

        loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious) # 这里只有记录matched_ious，并不会反传，因为单纯计算没有grad_fn

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        # 逐层处理
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0] # 计算batch size
            batch_score = preds_dict[0]['heatmap'][..., -self.num_proposals:].sigmoid() # (1, 10, 200)
            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1) # 提取query的分类类别并进行one hot编码
            batch_score = batch_score * preds_dict[0]['query_heatmap_score'] * one_hot # 融合初始预测和最终预测分数 (1, 10, 200)

            # 提取预测几何参数
            batch_center = preds_dict[0]['center'][..., -self.num_proposals:] # (1, 2, 200)
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:] # (1, 1, 200)
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:] # (1, 3, 200)
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:] # (1, 2, 200)
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:] # (1, 2, 200)
            
            # 对预测的bbox进行解码
            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ] # 对行人和锥筒单独处理
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'], indices=[0], radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[1], radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'], indices=[2], radius=0.7),
                ]

            ret_layer = []
            # 逐帧处理
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes'] # (200, 9)
                scores = temp[i]['scores'] # (200,)
                labels = temp[i]['labels'] # (200,)
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores) # 初始化保持mask
                    # 逐个任务处理
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1) # 提取中心点和分数
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                ) # 进行circle_nms计算被keep的bbox的索引
                            else:
                                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev) # 构造LidarInstanceBox后转换bev
                                top_scores = scores[task_mask] # 提取对应分数
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize'],
                                ) # 进行NMS计算被keep的bbox的索引
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool() # 保留下的mask
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        # 构造输出结果
        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res
