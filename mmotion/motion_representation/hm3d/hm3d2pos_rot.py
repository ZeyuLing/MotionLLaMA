import torch
from torch import Tensor

from mmotion.motion_representation.skeleton import build_uniform_skeleton
from mmotion.registry import FUNCTIONS
from mmotion.utils.geometry.quaternion import qinv, qrot
from mmotion.utils.geometry.rotation_convert import quaternion_to_cont6d
from mmotion.utils.typing import SampleList

def recover_root_rot_pos(data: Tensor):
    """ Get root joint position and rotation from hm3d vectors
    :param data: hm3d vector
    :return:
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

@FUNCTIONS.register_module()
def hm3d2joints(data: Tensor, data_samples: SampleList=None):
    """ recover joints coordinates from coordinates dims
    :param data: b t c
    :param num_joints: 22 for smpl. 52 for smpl-h
    :return:
    """
    num_joints= int(data_samples.get('num_joints')[0])
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(num_joints - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions

@FUNCTIONS.register_module()
def hm3d2rotation(data: Tensor, data_samples: SampleList=None):

    num_joints = int(data_samples.get('num_joints')[0])
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (num_joints - 1) * 3
    end_indx = start_indx + (num_joints - 1) * 6
    cont6d_params = data[..., start_indx: end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, num_joints, 6)
    return cont6d_params

@FUNCTIONS.register_module()
def hm3d2joints_from_rot(data: Tensor, data_samples:SampleList=None, num_joints=52):
    """ recover joints coordinates from cont6d dims
    :param data: b t c
    :param num_joints: 22 for smpl, 52 for smplh
    :param skeleton: SMPLSkeleton. Used for kinematics forward propagation.
    :param device: only used when skeleton is None.
    :return:
    """
    # get root position and rotation.
    # if skeleton is None:
    #     skeleton = standard_skeleton
    num_joints = int(data_samples.get('num_joints')[0]) or num_joints
    skeleton = build_uniform_skeleton(num_joints)
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (num_joints - 1) * 3
    end_indx = start_indx + (num_joints - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, num_joints, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


