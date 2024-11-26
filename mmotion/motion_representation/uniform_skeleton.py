import numpy as np

import torch

from mmotion.motion_representation.param_utils import RAW_OFFSETS_DICT, CHAIN_DICT
from mmotion.motion_representation.skeleton import Skeleton, l_idx1, l_idx2, face_joint_idx


def uniform_skeleton(positions, target_offset):
    """
    Uniformly scales a skeleton to match a target offset.

    Args:
        positions (numpy.ndarray): Input skeleton joint positions.
        target_offset (torch.Tensor): Target offset for the skeleton.

    Returns:
        numpy.ndarray: New joint positions after scaling and inverse/forward kinematics.
    """
    n_raw_offsets = RAW_OFFSETS_DICT[positions.shape[-2]]
    kinematic_chain = CHAIN_DICT[positions.shape[-2]]
    # Creating a skeleton with a predefined kinematic chain
    src_skel = Skeleton(n_raw_offsets, kinematic_chain)

    # Calculate the global offset of the source skeleton
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))

    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # Calculate Scale Ratio as the ratio of legs
    src_leg_len = np.abs(src_offset[l_idx1]).max(
    ) + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max(
    ) + np.abs(tgt_offset[l_idx2]).max()

    # Scale ratio for uniform scaling
    scale_rt = tgt_leg_len / src_leg_len
    # Extract the root position of the source skeleton
    src_root_pos = positions[:, 0]
    # Scale the root position based on the calculated ratio
    tgt_root_pos = src_root_pos * scale_rt

    # Inverse Kinematics to get quaternion parameters
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_idx)

    # Forward Kinematics with the new root position and target offset
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints
