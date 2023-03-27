# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction. (2, 1, 128, 128)
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution. (2, 1, 128, 128)
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1) # 计算正例的权重 (2, 1, 128, 128)
    # 越靠近正例中心，权重越小，在中心附近分错要减少损失，因为中心附近，虽然为负样本但是因为靠近中心导致分数较高
    # 如果不和pred.pow(alpha)相互制衡，会使得损失被中心附近的负样本占据
    neg_weights = (1 - gaussian_target).pow(gamma) # 计算负例权重 (2, 1, 128, 128) 
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights # 根据pos_weights只计算正例
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    # python矩阵式计算的思想是全部计算，然后根据mask或者权重取对象的值
    return pos_loss + neg_loss


@LOSSES.register_module()
class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha # 2.0
        self.gamma = gamma # 4.0
        self.reduction = reduction # 'mean'
        self.loss_weight = loss_weight # 1.0

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction) # 'mean'
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
