import torch
from mmotion.evaluation.metrics.base_motion_metric import BaseMotionMetric
from mmotion.registry import METRICS

@METRICS.register_module(force=True)
class FDE(BaseMotionMetric):
    """Final Average Distance Error"""

    metric = 'FDE'

    def process_result(self, gt, pred, has_hand: bool = True):
        """Process an image.
        :param gt: t j c
        :param pred: t j c
        :return:Average distance error
        """
        error = torch.linalg.norm(pred[..., -1:, :] - gt[..., -1:, :], ord=2, dim=-1)
        error = error.mean()

        return error
