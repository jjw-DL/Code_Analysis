# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.builder import ALGORITHMS
from mmrazor.models.utils import add_prefix
from .base import BaseAlgorithm


@ALGORITHMS.register_module()
class GeneralDistill(BaseAlgorithm):
    """General Distillation Algorithm.

    Args:
        with_student_loss (bool): Whether to use student loss.
            Defaults to True.
        with_teacher_loss (bool): Whether to use teacher loss.
            Defaults to False.
    """

    def __init__(self,
                 with_student_loss=True,
                 with_teacher_loss=False,
                 **kwargs):

        super(GeneralDistill, self).__init__(**kwargs)
        self.with_student_loss = with_student_loss # True
        self.with_teacher_loss = with_teacher_loss # False

    def train_step(self, data, optimizer):
        """"""
        losses = dict()
        if self.with_teacher_loss:
            teacher_losses = self.distiller.exec_teacher_forward(data)
            teacher_losses = add_prefix(teacher_losses, 'teacher')
            losses.update(teacher_losses)
        else:
            # Just to be able to trigger the forward hooks that
            # have been registered
            # data:{'img_meta', 'img', 'gt_sementic_seg'}
            _ = self.distiller.exec_teacher_forward(data)

        if self.with_student_loss:
            student_losses = self.distiller.exec_student_forward(
                self.architecture, data)  # self.architecture: pspnet r18 (mmseg.EncoderDecoder)
            student_losses = add_prefix(student_losses, 'student') # 在原始loss前加loss
            losses.update(student_losses) # 在dict中更新
        else:
            # Just to be able to trigger the forward hooks that
            # have been registered
            _ = self.distiller.exec_student_forward(self.architecture, data)

        distill_losses = self.distiller.compute_distill_loss(data)
        distill_losses = add_prefix(distill_losses, 'distiller') # 添加distiller前缀
        losses.update(distill_losses)

        loss, log_vars = self._parse_losses(losses) # {'student_loss'和'distiller_loss'}
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        return outputs
