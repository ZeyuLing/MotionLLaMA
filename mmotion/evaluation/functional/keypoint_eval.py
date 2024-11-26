import sys

import numpy as np
import os

import torch
from einops import rearrange

sys.path.append(os.curdir)
from mmotion.evaluation.functional.mesh_eval import compute_similarity_transform


def keypoint_mpjpe(pred: np.ndarray,
                   gt: np.ndarray,
                   mask: np.ndarray = None,
                   alignment: str = 'none'):
    """  Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (P-MPJPE).
    Note:
        - batch_size: N
        - num_keypoints: K
        - keypoint_dims: C`
    :param pred: Predicted keypoint location with shape [N, T, J, C].
    :param gt: Groundtruth keypoint location with shape [N, T, J, C].
    :param mask: N,J. 0 for ignored joints, 1 for others.
    :param alignment: method to align the prediction with the
                      ground truth.
                    Supported options are:
                    - ``'none'``: no alignment will be applied
                    - ``'scale'``: align in the least-square sense in scale
                    - ``'procrustes'``: align in the least-square sense in
                        scale, rotation and translation.
    :return: tuple: A tuple containing joint position errors
        - (float | np.ndarray): mean per-joint position error (mpjpe).
        - (float | np.ndarray): mpjpe after rigid alignment with the
            ground truth (p-mpjpe).
    """
    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')
    dist = np.linalg.norm(pred - gt, ord=2, axis=-1)
    if mask is not None:
        mask = mask.astype(np.bool)
        dist = rearrange(dist, 'b t j -> b j t')
        dist = dist[mask]
        error = dist.mean()
    else:
        error = dist.mean()
    return error


def average_distance_error(pred: torch.Tensor, gt: torch.Tensor, reduction='mean'):
    """
    :param pred: t j c or b t j c
    :param gt: t j c or b t j c
    :return:
    """
    error = torch.linalg.norm(pred[..., :1, :] - gt[..., :1, :], ord=2, dim=-1)
    if reduction == 'mean':
        error = error.mean()
    elif reduction == 'sum':
        error = error.sum()
    return error


def final_distance_error(pred: torch.Tensor, gt: torch.Tensor, reduction='mean'):
    """

    :param pred: t j c or b t j c
    :param gt: t j c or b t j c
    :return:
    """
    error = torch.linalg.norm(pred[..., -1:, :] - gt[..., -1:, :], ord=2, dim=-1)
    if reduction == 'mean':
        error = error.mean()
    elif reduction == 'sum':
        error = error.sum()
    return error


if __name__ == '__main__':
    pred = np.random.random([2, 64, 52, 3])
    gt = np.random.random([2, 64, 52, 3])
    mask = np.random.randint(0, 2, size=[2, 52])
    print(keypoint_mpjpe(pred, gt, mask))
