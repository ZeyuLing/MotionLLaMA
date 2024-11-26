from typing import Optional, Union, Tuple, Dict

import torch
from einops import einsum

from mmotion.core.evaluation.mesh_eval import compute_similarity_transform
from mmotion.evaluation.metrics.base_motion_metric import BaseMotionMetric
from mmotion.registry import METRICS


@METRICS.register_module(force=True)
class MPJPE(BaseMotionMetric):
    metric = 'MPJPE'
    """Mean Per Joint Position Error metric for motion sequence."""

    def __init__(self, gt_key: str = 'gt_motion', pred_key: str = 'pred_motion', scaling=1, device='cpu',
                 collect_device: str = 'cpu', prefix: Optional[str] = None, has_hand_key: str = 'has_hand',
                 rot_type: str = 'quaternion', alignment: str = 'none', whole_body: bool = True) -> None:
        assert alignment in ['none', 'procrustes', 'scale'], alignment
        self.alignment = alignment
        if alignment == 'procrustes':
            self.metric = 'P-MPJPE'
        elif alignment == 'scale':
            self.metric = 'N-MPJPE'
        self.whole_body = whole_body
        super().__init__(gt_key, pred_key, scaling, device, collect_device, prefix, has_hand_key, rot_type)

    def process_result(self, gt, pred, has_hand: bool = True):
        """
        :param gt: t j c
        :param pred: t j c
        :param has_hand: whether take hands into account
        :return:
        """
        t, j = gt.shape[:2]
        mask = None
        if not has_hand:
            mask = torch.ones((t, j), dtype=torch.bool).to(gt.device)
            mask[:, -30:] = 0.
        result: Union[float, Tuple[float]] = keypoint_mpjpe(pred, gt, mask, self.whole_body, self.alignment)
        if self.whole_body:
            result: Dict = {
                self.metric: result[0],
                self.metric + '_body': result[1],
                self.metric + '_hand': result[2]
            }
        return result


def keypoint_mpjpe(pred: torch.Tensor,
                   gt: torch.Tensor,
                   mask: torch.Tensor = None,
                   whole_body: bool = False,
                   alignment: str = 'none'):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (P-MPJPE).

    Note:
        - batch_size: N
        - num_keypoints: K
        - keypoint_dims: C

    Args:
        pred (torch.Tensor): Predicted keypoint location with shape [N, K, C].
        gt (torch.Tensor): Groundtruth keypoint location with shape [N, K, C].
        mask (torch.Tensor): Visibility of the target with shape [N, K].
            False for invisible joints, and True for visible.
            Invisible joints will be ignored for accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:

                - ``'none'``: no alignment will be applied
                - ``'scale'``: align in the least-square sense in scale
                - ``'procrustes'``: align in the least-square sense in
                    scale, rotation and translation.

    Returns:
        tuple: A tuple containing joint position errors

        - (float | torch.Tensor): mean per-joint position error (mpjpe).
        - (float | torch.Tensor): mpjpe after rigid alignment with the
            ground truth (p-mpjpe).
    """
    pred = pred.float()
    gt = gt.float()
    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = torch.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = einsum(pred, pred, 'n k c, n k c -> n')
        pred_dot_gt = einsum(pred, gt, 'n k c, n k c -> n')
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = torch.linalg.norm(pred - gt, ord=2, dim=-1)
    if whole_body:
        hand_error = error[:, 22:].mean()
        body_error = error[:, :22].mean()

    if mask is not None:
        error = error[mask]
    error = error.mean()

    if whole_body:
        return error, body_error, hand_error

    return error
