"""
    functions to convert vectors back to joint coordinates
"""
from typing import List

import torch
from einops import rearrange

from torch import Tensor

from mmotion.registry import FUNCTIONS
from mmotion.structures import DataSample
from mmotion.utils.geometry.rotation_convert import cont6d_to_axis_angle
from mmotion.motion_representation.skeleton import Skeleton, build_uniform_skeleton
from mmotion.utils.typing import SampleList


@FUNCTIONS.register_module()
def rel_xz_interhuman2joints(data: Tensor, data_samples: SampleList = None, num_joints: int = None):
    """ only xz coordinates are relative, y is global
    :param data: interhuman vectors
    :param data_samples:
    :return:
    """
    num_joints = num_joints or data_samples.get('num_joints')[0]
    positions = data[..., :num_joints * 3]
    return rel_xz_joints_to_global(positions, data_samples)


@FUNCTIONS.register_module()
def relative_interhuman2joints(data: Tensor, data_samples: SampleList = None, num_joints=52):
    num_joints = data_samples.get('num_joints', [num_joints])
    if isinstance(num_joints, List):
        num_joints = num_joints[0]
    positions = data[..., :num_joints * 3]
    return relative_joints_to_global(positions)


@FUNCTIONS.register_module()
def interhuman2joints(data: Tensor, data_samples: DataSample = None, num_joints=52):
    if data_samples is not None:
        num_joints = data_samples.get('num_joints', [num_joints])
    if isinstance(num_joints, List):
        num_joints = num_joints[0]
    positions = data[..., :num_joints * 3]
    return flatten_joints(positions)


def interhuman2joints_from_rot(data: Tensor, data_samples: SampleList = None, skeleton: Skeleton = None):
    num_joints = int(data_samples.get('num_joints')[0])
    root_pos = data[..., :3]  # first 3 channels
    cont6d_params = data[..., num_joints * 6: num_joints * 12]
    cont6d_params = rearrange(cont6d_params, '... (j c) -> ... j c', j=num_joints, c=6)
    positions = skeleton.forward_kinematics_cont6d(cont6d_params, root_pos)
    return positions


@FUNCTIONS.register_module()
def interhuman2rotation(data: Tensor, data_samples: SampleList = None):
    num_joints = int(data_samples.get('num_joints')[0])
    cont6d_params = data[..., - (num_joints * 6 + 4): -4]
    cont6d_params = rearrange(cont6d_params, '... (j c) -> ... j c', j=num_joints, c=6)
    return cont6d_params


@FUNCTIONS.register_module()
def interhuman2rotation_no_fc(data: Tensor, data_samples: SampleList = None):
    num_joints = int(data_samples.get('num_joints')[0])
    cont6d_params = data[..., - num_joints * 6:]
    cont6d_params = rearrange(cont6d_params, '... (j c) -> ... j c', j=num_joints, c=6)
    return cont6d_params


@FUNCTIONS.register_module()
def flatten_joints(data: Tensor, data_samples: SampleList = None):
    jc = data.shape[-1]
    j = jc // 3
    return rearrange(data, '... (j c) -> ... j c', j=j, c=3)


@FUNCTIONS.register_module()
def rel_xz_joints_to_global(data: Tensor, data_samples: SampleList = None):
    jc = data.shape[-1]
    j = jc // 3
    data = rearrange(data, '... (j c) -> ... j c', j=j, c=3)
    data[..., 1:, 0] += data[..., :1, 0]
    data[..., 1:, 2] += data[..., :1, 2]
    return data


@FUNCTIONS.register_module()
def relative_joints_to_global(data: Tensor, data_samples: SampleList = None):
    jc = data.shape[-1]
    j = jc // 3
    data = rearrange(data, '... (j c) -> ... j c', j=j, c=3)
    data[..., 1:, :] += data[..., :1, :]
    return data


@FUNCTIONS.register_module()
def dummy_vec2rotation(data: Tensor, data_samples: SampleList = None):
    return None


@FUNCTIONS.register_module()
def interhuman2rotation_ik(data: Tensor, data_samples: SampleList = None):
    """ Calculate joint rotation from interhuman format data with Inverse Kinematics
    :param data: b t c, interhuman batch data
    :param data_samples: data samples
    :return: b t j 6, cont6d rotation data
    """
    joints = interhuman2joints(data, data_samples)
    b = joints.shape[0]
    joints = rearrange(joints, 'b t j c -> (b t) j c')
    num_joints = int(data_samples.get('num_joints')[0])
    skeleton = build_uniform_skeleton(num_joints)
    rotation = skeleton.inverse_kinematic_cont6d(joints)

    rotation = rearrange(rotation, '(b t) j c -> b t j c', b=b)
    return rotation


@FUNCTIONS.register_module()
def transl_rotation_fk(data: Tensor, data_samples: SampleList = None):
    """ Forward kinematics to get joint positions
    :param data: 3 channels translation and 6 * num_joints rotation, b t c
    :param data_samples:
    :return:
    """
    num_joints = int(data_samples.get('num_joints')[0])
    b = data.shape[0]
    skeleton = build_uniform_skeleton(num_joints)
    transl = rearrange(data[..., :3], '... c -> (...) c')

    rotation_6d = rearrange(data[..., 3:], '... (j c) -> (...) j c', j=num_joints, c=6)

    # (bt, j c)
    joints = skeleton.forward_kinematics_cont6d(rotation_6d, transl)

    joints = rearrange(joints, '(b t) j c -> b t j c', b=b)
    return joints


@FUNCTIONS.register_module()
def transl_cont6d_to_axis_angle(data: torch.Tensor, data_samples: SampleList = None):
    cont6d = data[..., 3:]
    num_joints = cont6d.shape[-1] // 6
    cont6d = rearrange(cont6d, '... (j c) -> ... j c', j=num_joints, c=6)
    axis_angle = cont6d_to_axis_angle(cont6d)
    return axis_angle
