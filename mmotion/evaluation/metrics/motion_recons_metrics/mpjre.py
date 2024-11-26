import torch

from mmotion.evaluation.metrics.base_motion_metric import BaseMotionMetric
from mmotion.evaluation.metrics.motion_recons_metrics.mpjpe import keypoint_mpjpe
from mmotion.registry import METRICS
from mmotion.utils.geometry.rotation_convert import rot_convert


@METRICS.register_module(force=True)
class MPJRE(BaseMotionMetric):
    metric='MPJRE'
    """Mean Per Joint Rotation Error metric for motion sequence."""



    def process_result(self, gt, pred, has_hand: bool = True):
        """
        :param gt: t j c
        :param pred: t j c
        :param has_hand: whether take hands into account
        :return:
        """
        gt = rot_convert(gt, self.rot_type, 'euler')
        pred = rot_convert(pred, self.rot_type, 'euler')
        t, j = gt.shape[:2]
        mask = None
        if not has_hand:
            mask = torch.ones((t, j), dtype=torch.bool).to(gt.device)
            mask[:, -30:] = 0.
        return keypoint_mpjpe(pred, gt, mask, 'none')

