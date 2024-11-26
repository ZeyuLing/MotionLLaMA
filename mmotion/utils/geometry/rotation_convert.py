import numpy as np
from functools import wraps
from typing import Union

import pytorch3d.transforms
import torch
from einops import rearrange
from pytorch3d.transforms.rotation_conversions import _angle_from_tan, _index_from_letter, axis_angle_to_matrix, \
    matrix_to_rotation_6d, rotation_6d_to_matrix

from mmotion.utils.geometry.quaternion import qmul

rot_dim = {
    'quaternion': 4,
    'euler': 3,
    'rot6d': 6,
    'rotation_6d': 6,
    'matrix': 9,
    'axis_angle': 3,
    'joints': 3
}


def check_matrix_shape(func):
    """ A decorator function to support matrix shape in [..., 9], the original functions only supports [..., 3, 3]
    :param func: original rotation conversion function
    :return:
    """

    @wraps(func)
    def check(matrix: Union[torch.Tensor, np.ndarray], *args, **kwargs):
        if matrix.shape[-1] == 9:
            matrix = rearrange(matrix, '... (a b) -> ... a b', a=3, b=3)
        return func(matrix, *args, **kwargs)

    return check


def compatible_numpy(func):
    def wrapper(rot, *args, **kwargs):
        is_numpy = False
        if isinstance(rot, np.ndarray):
            is_numpy = True
            rot = torch.from_numpy(rot)
        res = func(rot, *args, **kwargs)
        if is_numpy:
            res = res.numpy()
        return res

    return wrapper


@compatible_numpy
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@compatible_numpy
def quaternion_to_cont6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d


@compatible_numpy
def quaternion_to_rotation_6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    rot6d = matrix_to_rotation_6d(rotation_mat)
    return rot6d


@compatible_numpy
def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


@compatible_numpy
@check_matrix_shape
def matrix_to_cont6d(matrix: torch.Tensor) -> torch.Tensor:
    quat = matrix_to_quaternion(matrix)
    cont6d = quaternion_to_cont6d(quat)
    return cont6d


@compatible_numpy
def euler_to_quaternion(e: torch.Tensor, order: str = "XYZ", deg=True):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.view(-1, 3)

    ## if euler angles in degrees
    if deg:
        e = e * torch.pi / 180.

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.sin(y / 2), torch.zeros_like(y)), dim=1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z / 2)), dim=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.view(original_shape)


@compatible_numpy
@check_matrix_shape
def matrix_to_euler(matrix: torch.Tensor, order: str = "XYZ", deg=False) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians or degree.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        order: order string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(order) != 3:
        raise ValueError("order must have 3 letters.")
    if order[1] in (order[0], order[2]):
        raise ValueError(f"Invalid order {order}.")
    for letter in order:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in order string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    ## if euler angles in degrees

    i0 = _index_from_letter(order[0])
    i2 = _index_from_letter(order[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            order[0], order[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            order[2], order[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    o = torch.stack(o, -1)
    if deg:
        o = o / torch.pi * 180.
    return o


@compatible_numpy
def quaternion_to_euler_deg(quat: torch.Tensor, order: str = "XYZ"):
    """
    :param quat: quaternion tensor
    :param order:
    :return:
    """
    matrix = quaternion_to_matrix(quat)
    euler = matrix_to_euler(matrix, order, True)
    return euler


@compatible_numpy
def quaternion_to_euler(quat: torch.Tensor, order: str = "XYZ", deg=False):
    matrix = quaternion_to_matrix(quat)
    euler = matrix_to_euler(matrix, order, deg)
    return euler


@compatible_numpy
def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


@compatible_numpy
def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


@compatible_numpy
def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


@compatible_numpy
def axis_angle_to_euler(axis_angle: torch.Tensor, order: str = "XYZ"):
    """
    :param axis_angle: axis angle tensor
    :param order: XYZ as default
    :return:
    """
    matrix = axis_angle_to_matrix(axis_angle)
    euler = matrix_to_euler(matrix, order, False)
    return euler


@compatible_numpy
def axis_angle_to_euler_deg(axis_angle: torch.Tensor, order: str = "XYZ"):
    """
    :param axis_angle: axis angle tensor
    :param order: XYZ as default
    :return:
    """
    matrix = axis_angle_to_matrix(axis_angle)
    euler = matrix_to_euler(matrix, order, True)
    return euler


@compatible_numpy
@check_matrix_shape
def matrix_to_quaternion(matrix: torch.Tensor):
    return pytorch3d.transforms.matrix_to_quaternion(matrix)


@check_matrix_shape
def matrix_to_axis_angle(matrix: torch.Tensor):
    return pytorch3d.transforms.matrix_to_axis_angle(matrix)


@compatible_numpy
def axis_angle_to_cont6d(axis_angle: torch.Tensor):
    quaternion = axis_angle_to_quaternion(axis_angle)
    cont6d = quaternion_to_cont6d(quaternion)
    return cont6d


@compatible_numpy
def cont6d_to_axis_angle(cont6d: torch.Tensor):
    matrix = cont6d_to_matrix(cont6d)
    axis_angle = matrix_to_axis_angle(matrix)
    return axis_angle


@compatible_numpy
def cont6d_to_quaternion(cont6d: torch.Tensor):
    matrix = cont6d_to_matrix(cont6d)
    quaternion = matrix_to_quaternion(matrix)
    return quaternion


@compatible_numpy
def cont6d_to_euler_deg(contd: torch.Tensor, order: str = "XYZ"):
    """
    :param contd: cont6d tensor
    :param order: XYZ as default
    :return:
    """
    matrix = cont6d_to_matrix(contd)
    euler = matrix_to_euler(matrix, order, True)
    return euler


@compatible_numpy
def cont6d_to_euler(contd: torch.Tensor, order: str = "XYZ"):
    """
    :param contd: cont6d tensor
    :param order: XYZ as default
    :return:
    """
    matrix = cont6d_to_matrix(contd)
    euler = matrix_to_euler(matrix, order, False)
    return euler


@compatible_numpy
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    return pytorch3d.transforms.rotation_6d_to_matrix(d6)


@compatible_numpy
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return pytorch3d.transforms.matrix_to_rotation_6d(matrix)


@compatible_numpy
def axis_angle_to_rotation_6d(axis_angle: torch.Tensor):
    matrix = axis_angle_to_matrix(axis_angle)
    rotation_6d = matrix_to_rotation_6d(matrix)
    return rotation_6d

@compatible_numpy
def rotation_6d_to_axis_angle(rotation_6d: torch.Tensor):
    matrix = rotation_6d_to_matrix(rotation_6d)
    axis_angle = matrix_to_axis_angle(matrix)
    return axis_angle

@compatible_numpy
def rotation_6d_to_quaternion(rotation_6d: torch.Tensor):
    matrix = rotation_6d_to_matrix(rotation_6d)
    quaternion = matrix_to_quaternion(matrix)
    return quaternion

@compatible_numpy
def rotation_6d_to_euler(rotation_6d: torch.Tensor, order='XYZ'):
    matrix = rotation_6d_to_matrix(rotation_6d)
    euler_rad = matrix_to_euler(matrix, order=order, deg=False)
    return euler_rad

@compatible_numpy
def rotation_6d_to_euler_deg(rotation_6d: torch.Tensor, order='XYZ'):
    matrix = rotation_6d_to_matrix(rotation_6d)
    euler_deg = matrix_to_euler(matrix, order=order, deg=True)
    return euler_deg
def rot_convert(from_rot: Union[torch.Tensor, np.ndarray], from_type: str, to_type: str, **kwargs):
    if from_type == to_type:
        return from_rot
    function_name = f'{from_type}_to_{to_type}'
    func = globals()[function_name]
    return func(from_rot, **kwargs)
