import numpy as np
from typing import Optional

from mmotion.motion_representation.skeleton import face_joint_idx
from mmotion.motion_representation.uniform_skeleton import uniform_skeleton
from mmotion.utils.geometry.quaternion import qbetween_np, qrot_np
from mmotion.motion_representation.hm3d.joints2hm3d import foot_detect, get_cont6d_params

def multi_person_joints2interhuman(
        positions: np.ndarray,
        tgt_offsets: np.ndarray,
        feet_thre=0.002,
        interactor_positions: Optional[np.ndarray] = None,
        p1=True
):
    """
    :param positions: positions of P1
    :param tgt_offsets: tgt_offsets
    :param feet_thre: velocity feet contact thresh
    :param interactor_positions: 
    :return: 
    """
    num_joints = positions.shape[-2]
    if not isinstance(positions, np.ndarray):
        positions = positions.numpy()
    if not isinstance(interactor_positions, np.ndarray):
        interactor_positions = interactor_positions.numpy()

    positions = uniform_skeleton(positions, tgt_offsets)
    interactor_positions = uniform_skeleton(interactor_positions, tgt_offsets)

    # assume motion has ever landed onto the floor
    floor_height = min(positions.min(axis=0).min(axis=0)[1],
                       interactor_positions.min(axis=0).min(axis=0)[1])
    positions[:, :, 1] -= floor_height
    interactor_positions[:, :, 1] -= floor_height

    # Center the skeleton at the origin in the XZ plane
    if p1:
        root_pos_init = interactor_positions[0]
    else:
        root_pos_init = positions[0]

    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions -= root_pose_init_xz

    # Ensure the initial facing direction is along Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    # Calculate quaternion for root orientation
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # Extract additional features including root height and root data
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    rot_data = cont_6d_params.reshape(len(cont_6d_params), -1)

    # Get Joint Rotation Invariant Position Represention
    # (seq_len, (positions_num-1)*3) local joint position
    positions = positions.reshape(len(positions), -1)

    global_vel = positions[1:] - positions[:-1]
    global_vel = global_vel.reshape(len(global_vel), -1)

    # Concatenate all features into a single array
    data = np.concatenate([positions[:-1], global_vel, rot_data[:-1], feet_l, feet_r], axis=-1)
    assert data.shape[-1] == num_joints*12+4, (data.shape[-1], num_joints)
    return data, positions

def joints2interhuman(positions: np.ndarray,
                      tgt_offsets: np.ndarray,
                      feet_thre=0.002,
                      interactor_positions: Optional[np.ndarray] = None
                      ):
    """
    :param positions: joints coordinate, [t j c], note that y axis is the vertical axis.
    :param feet_thre: threshold for foot contact detection, if the foot velocity is < feet_thre,
    regard foot on the ground
    :param tgt_offsets: the length of each bone in the skeleton.
    :return: humanml3d motion vector in 263(body only) or 623(body and hands) channels
    """
    num_joints = positions.shape[-2]
    if interactor_positions is not None:
        return multi_person_joints2interhuman(
            positions=positions,
            tgt_offsets=tgt_offsets,
            feet_thre=feet_thre,
            interactor_positions=interactor_positions
        )
    if not isinstance(positions, np.ndarray):
        positions = positions.numpy()

    positions = uniform_skeleton(positions, tgt_offsets)

    # assume motion has ever landed onto the floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # Center the skeleton at the origin in the XZ plane
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions -= root_pose_init_xz

    # Ensure the initial facing direction is along Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    # Calculate quaternion for root orientation
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Quaternion and Cartesian representation'''

    # Extract additional features including root height and root data
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    # Get Joint Rotation Representation
    # (seq_len, (positions_num-1) *6) quaternion for skeleton positions
    rot_data = cont_6d_params.reshape(len(cont_6d_params), -1)

    positions = positions.reshape(len(positions), -1)

    # Get Joint Velocity Representation
    # (seq_len-1, positions_num*3)
    global_vel = positions[1:] - positions[:-1]
    global_vel = global_vel.reshape(len(global_vel), -1)

    # Concatenate all features into a single array
    data = np.concatenate([positions[:-1], global_vel, rot_data[:-1], feet_l, feet_r], axis=-1)
    assert data.shape[-1] == num_joints*12+4, (data.shape[-1], num_joints)
    return data, positions
