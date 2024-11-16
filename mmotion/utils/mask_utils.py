import torch

from mmotion.utils.geometry.rotation_convert import rot_dim

def get_mask_joints(num_frames, has_hand: bool, num_joints=52):
    """ Generate mask tensor for joints position tensor(transl body and hands, no face, 52 joints)
    1 for valid, 0 for invalid.
    :param num_frames: number of frames
    :param has_hand: if the hands are fake
    :param num_joints: 52 as default
    :return:
    """
    dim = 3
    mask = torch.ones([num_frames, num_joints, dim])
    if not has_hand:
        mask[..., -30:, :] = 0.
    return mask

def batch_get_mask_joints(num_frames: int, has_hand: torch.Tensor, num_joints=52):
    """
    :param num_frames: num_frames
    :param has_hand: has_hand for a batch. [B]
    :param num_joints: num of joints. 52 as default(1 root, 21 body, 30 hands)
    :return:
    """
    dim = 3
    if not isinstance(has_hand, torch.Tensor):
        has_hand = torch.tensor(has_hand)
    batch_size = len(has_hand)
    mask = torch.ones([batch_size, num_frames, num_joints, dim])
    mask[~has_hand, :, - 30:] = 0.
    return mask

def get_mask_rotation(num_frames, has_hand: bool, rot_type: str, num_joints=52):
    """ Generate mask tensor for smplx tensor(transl body and hands, no face, 52 joints)
    1 for valid, 0 for invalid.
    :param num_frames: number of frames
    :param has_hand: if the hands are fake
    :param rot_type: rotation type
    :param num_joints: 52 as default
    :return:
    """
    dim = rot_dim[rot_type]
    mask = torch.ones([num_frames, num_joints * dim])
    if not has_hand:
        mask[..., - 30 * dim:] = 0.
    return mask


def batch_get_mask_rotation(num_frames: int, has_hand: torch.Tensor, rot_type: str, num_joints=52):
    """
    :param num_frames: num_frames
    :param has_hand: has_hand for a batch. [B]
    :param rot_type: rotation representation type
    :param num_joints: num of joints. 52 as default(1 root, 21 body, 30 hands)
    :return:
    """
    dim = rot_dim[rot_type]
    if not isinstance(has_hand, torch.Tensor):
        has_hand = torch.tensor(has_hand)
    batch_size = len(has_hand)
    mask = torch.ones([batch_size, num_frames, num_joints * dim])
    mask[~has_hand, :, - 30 * dim:] = 0.
    return mask
