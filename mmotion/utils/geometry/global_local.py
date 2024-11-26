from typing import List

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, matrix_to_axis_angle

from mmotion.utils.geometry.rotation_convert import quaternion_to_matrix


def local_to_global_matrix(local_matrix: torch.Tensor, parent_indices: List[int]) -> torch.Tensor:
    """
    :param local_matrix: local rotation matrix. t j 3 3
    :param parent_indices: parent of each joint. List
    :return: global_matrix
    """
    global_matrix = torch.zeros_like(local_matrix)

    # for each joint
    for i in range(len(parent_indices)):
        # if root
        if parent_indices[i] == -1:
            global_matrix[:, i] = local_matrix[:, i]
        else:
            # global rotation = global rotation of parent * local rotation of it
            global_matrix[:, i] = global_matrix[:, parent_indices[i]] @ local_matrix[:, i]

    return global_matrix


def local_to_global_matrix_np(local_matrix: np.ndarray, parent_indices: List[int]):
    local_matrix = torch.from_numpy(local_matrix)
    return local_to_global_matrix(local_matrix, parent_indices).numpy()


def local_to_global_quat(local_quat: torch.Tensor, parent_indices: List[int]):
    local_matrix = quaternion_to_matrix(local_quat)
    global_matrix = local_to_global_matrix(local_matrix, parent_indices)
    global_quat = matrix_to_quaternion(global_matrix)
    return global_quat


def local_to_global_quat_np(local_quat: np.ndarray, parent_indices: List[int]):
    local_quat = torch.from_numpy(local_quat)
    return local_to_global_quat(local_quat, parent_indices).numpy()


def local_to_global_axis_angle(local_axis_angle: torch.Tensor, parent_indices: List[int]):
    local_matrix = axis_angle_to_matrix(local_axis_angle)
    global_matrix = local_to_global_matrix(local_matrix, parent_indices)
    global_axis_angle = matrix_to_axis_angle(global_matrix)
    return global_axis_angle


def local_to_global_axis_angle_np(local_axis_angle: np.ndarray, parent_indices: List[int]):
    local_axis_angle = torch.from_numpy(local_axis_angle)
    return local_to_global_axis_angle(local_axis_angle, parent_indices).numpy()
