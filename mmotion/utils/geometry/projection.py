import torch
import numpy as np

from mmotion.utils.geometry.rotation_convert import matrix_to_quaternion, quaternion_to_matrix, \
    quaternion_to_axis_angle


def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def estimate_translation_np(S,
                            joints_2d,
                            joints_conf,
                            focal_length=5000,
                            img_size=224):
    """Find camera translation that brings 3D joints S closest to 2D the
    corresponding joints_2d.

    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length, focal_length])
    # optical center
    center = np.array([img_size / 2., img_size / 2.])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    OO = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array([
        F * np.tile(np.array([1, 0]), num_joints),
        F * np.tile(np.array([0, 1]), num_joints),
        OO - np.reshape(joints_2d, -1)
    ]).T
    c = (np.reshape(joints_2d, -1) - OO) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000., img_size=224.):
    """Find camera translation that brings 3D joints S closest to 2D the
    corresponding joints_2d.

    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = S.device
    # Use only joints 25:49 (GT joints)
    S = S[:, 25:, :].cpu().numpy()
    joints_2d = joints_2d[:, 25:, :].cpu().numpy()
    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(
            S_i,
            joints_i,
            conf_i,
            focal_length=focal_length,
            img_size=img_size)
    return torch.from_numpy(trans).to(device)


def project_points(points_3d, camera, focal_length, img_res):
    """Perform orthographic projection of 3D points using the camera
    parameters, return projected 2D points in image plane.

    Notes:
        batch size: B
        point number: N
    Args:
        points_3d (Tensor([B, N, 3])): 3D points.
        camera (Tensor([B, 3])): camera parameters with the
            3 channel as (scale, translation_x, translation_y)
    Returns:
        points_2d (Tensor([B, N, 2])): projected 2D points
            in image space.
    """
    batch_size = points_3d.shape[0]
    device = points_3d.device
    cam_t = torch.stack([
        camera[:, 1], camera[:, 2], 2 * focal_length /
                                    (img_res * camera[:, 0] + 1e-9)
    ],
        dim=-1)
    camera_center = camera.new_zeros([batch_size, 2])
    rot_t = torch.eye(
        3, device=device,
        dtype=points_3d.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    keypoints_2d = perspective_projection(
        points_3d,
        rotation=rot_t,
        translation=cam_t,
        focal_length=focal_length,
        camera_center=camera_center)
    return keypoints_2d


def weak_perspective_projection(points, scale, translation):
    """This function computes the weak perspective projection of a set of
    points.

    Input:
        points (bs, N, 3): 3D points
        scale (bs,1): scalar
        translation (bs, 2): point 2D translation
    """
    projected_points = scale.view(-1, 1, 1) * (
            points[:, :, :2] + translation.view(-1, 1, 2))

    return projected_points


def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """convert the camera parameters from the crop camera to the full camera.

    :param crop_cam: shape=(N, 3) weak perspective camera in cropped
       img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - img_w / 2.) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - img_h / 2.) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def projection(pred_joints, pred_camera, iwp_mode=True):
    """Project 3D points on the image plane based on the given camera info,
    Identity rotation and Weak Perspective (IWP) camera is used when
    iwp_mode = True
    """
    batch_size = pred_joints.shape[0]
    if iwp_mode:
        cam_sxy = pred_camera['cam_sxy']
        pred_cam_t = torch.stack([
            cam_sxy[:, 1], cam_sxy[:, 2], 2 * 5000. /
                                          (224. * cam_sxy[:, 0] + 1e-9)
        ],
            dim=-1)

        camera_center = torch.zeros(batch_size, 2)
        pred_keypoints_2d = perspective_projection(
            pred_joints,
            rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(
                pred_joints.device),
            translation=pred_cam_t,
            focal_length=5000.,
            camera_center=camera_center)

    else:
        raise NotImplementedError
    return pred_keypoints_2d


def compute_twist_rotation(rotation_matrix, twist_axis):
    '''
    Compute the twist component of given rotation and twist axis
    https://stackoverflow.com/questions/3684269/component-of-a-quaternion-rotation-around-an-axis
    Parameters
    ----------
    rotation_matrix : Tensor (B, 3, 3,)
        The rotation to convert
    twist_axis : Tensor (B, 3,)
        The twist axis
    Returns
    -------
    Tensor (B, 3, 3)
        The twist rotation
    '''
    quaternion = matrix_to_quaternion(rotation_matrix)

    twist_axis = twist_axis / (
            torch.norm(twist_axis, dim=1, keepdim=True) + 1e-9)

    projection = torch.einsum('bi,bi->b', twist_axis,
                              quaternion[:, 1:]).unsqueeze(-1) * twist_axis

    twist_quaternion = torch.cat([quaternion[:, 0:1], projection], dim=1)
    twist_quaternion = twist_quaternion / (
            torch.norm(twist_quaternion, dim=1, keepdim=True) + 1e-9)

    twist_rotation = quaternion_to_matrix(twist_quaternion)
    twist_aa = quaternion_to_axis_angle(twist_quaternion)

    twist_angle = torch.sum(
        twist_aa, dim=1, keepdim=True) / torch.sum(
        twist_axis, dim=1, keepdim=True)

    return twist_rotation, twist_angle
