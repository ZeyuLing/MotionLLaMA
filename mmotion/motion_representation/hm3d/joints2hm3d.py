import numpy as np

from mmotion.utils.geometry.quaternion import qbetween_np, qrot_np, qmul_np, qinv_np
from mmotion.utils.geometry.rotation_convert import quaternion_to_cont6d
from mmotion.motion_representation.param_utils import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from mmotion.motion_representation.skeleton import face_joint_idx, fid_l, fid_r, Skeleton
from mmotion.motion_representation.uniform_skeleton import uniform_skeleton


def get_rifke(positions, r_rot):
    """
    :param positions:
    :return:
    """
    '''Local pose, xz are relative to the first frame'''
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    '''All pose face Z+'''
    positions = qrot_np(
        np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
    return positions


def foot_detect(positions:np.ndarray, thres:float=2e-3):
    """ Judge whether feet are on the floor
    :param positions: joint positions
    :param thres:
    :return:
    """
    # height factor is not used since the positions are not accurate, hm3d uses velocity to determine
    # whether feet are on floor.

    vel_factor, height_factor = np.array(
        [thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2

    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < vel_factor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2

    feet_r = (((feet_r_x + feet_r_y + feet_r_z)
               < vel_factor)).astype(np.float32)
    return feet_l, feet_r


def get_cont6d_params(positions: np.ndarray,
                      n_raw_offsets=t2m_raw_offsets,
                      kinematic_chain=t2m_body_hand_kinematic_chain):
    """
    :param positions: joint positions
    :param n_raw_offsets: raw offsets, determine the bone length
    :param kinematic_chain: joint topology
    :return:
    """
    # Initialize a skeleton object with a specified kinematic chain
    skel = Skeleton(n_raw_offsets, kinematic_chain)
    # (seq_len, positions_num, 4)
    quat_params = skel.inverse_kinematics_np(
        positions, face_joint_idx, smooth_forward=True)

    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    r_linear_velocity = (positions[1:, 0] - positions[:-1, 0]).copy()

    r_linear_velocity = qrot_np(r_rot[1:], r_linear_velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_angular_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    # (seq_len, positions_num, 4)
    return cont_6d_params, r_angular_velocity, r_linear_velocity, r_rot


def joints2hm3d(positions: np.ndarray, tgt_offsets: np.ndarray, feet_thre=0.002):
    """
    :param positions: joints coordinate, [t j c], note that y axis is the vertical axis.
    :param feet_thre: threshold for foot contact detection, if the foot velocity is < feet_thre,
    regard foot on the ground
    :param tgt_offsets: the length of each bone in the skeleton.
    :return: humanml3d motion vector in 263(body only) or 623(body and hands) channels
    """
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

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / \
                   np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    # Calculate quaternion for root orientation
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    uniform_joints = positions.copy()

    feet_l, feet_r = foot_detect(positions, feet_thre)
    '''Quaternion and Cartesian representation'''

    # Extract additional features including root height and root data
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions, r_rot)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation and linear velocity
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # Get Joint Rotation Representation
    # (seq_len, (positions_num-1) *6) quaternion for skeleton positions
    rot_data = cont_6d_params.reshape(len(cont_6d_params), -1)

    # Get Joint Rotation Invariant Position Represention
    # (seq_len, (positions_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Get Joint Velocity Representation
    # (seq_len-1, positions_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], uniform_joints.shape[1], axis=1),
                        uniform_joints[1:] - uniform_joints[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    # Concatenate all features into a single array
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, uniform_joints, positions, l_velocity
