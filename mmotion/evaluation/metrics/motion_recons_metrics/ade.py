import torch

from mmotion.evaluation.functional.keypoint_eval import average_distance_error
from mmotion.evaluation.metrics.base_motion_metric import BaseMotionMetric
from mmotion.registry import METRICS

@METRICS.register_module(force=True)
class ADE(BaseMotionMetric):
    """Mean Average Distance Error"""

    metric = 'ADE'

    def process_result(self, gt, pred, has_hand: bool = True):
        """Process an image.
        :param gt: t j c
        :param pred: t j c
        :return:Average distance error
        """
        error = average_distance_error(pred, gt, reduction='mean')

        return error
