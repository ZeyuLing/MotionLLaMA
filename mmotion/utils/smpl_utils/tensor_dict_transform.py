import torch
from einops import rearrange

from mmotion.utils.geometry.rotation_convert import rot_convert, rot_dim
from mmotion.utils.smpl_utils.smpl_key_const import TRANSL, GLOBAL_ORIENT, BODY_POSE, LEFT_HAND_POSE, zero_param, \
    RIGHT_HAND_POSE, BETAS, LEYE_POSE, REYE_POSE, EXPRESSION, JAW_POSE
from mmotion.utils.smpl_utils.transl import local2global


def tensor_to_smplx_dict(output: torch.Tensor,
                         rot_type: str = 'quaternion',
                         global2local: bool = False,
                         init_loc: torch.Tensor = None, ):
    """ Turn tensor composite with : transl + 52 * rot_dim to smpl dict
    :param output: smpl tensor(transl + pose)
    :param rot_type:
    :param global2local:
    :param init_loc:
    :return:
    """

    t = output.shape[0]
    transl = output[..., :3]
    dim = rot_dim[rot_type]
    if global2local is True:
        transl = local2global(transl, init_loc)

    rot = output[..., 3:]
    if rot_type == 'quaternion':
        rot = torch.clip(rot, -1, 1)
    rot = rearrange(rot, '... (j c) -> ... j c', c=dim)
    rot = rot_convert(rot, rot_type, 'axis_angle')
    rot = rearrange(rot, '... j c -> ... (j c)')
    smplx_dict = {
        TRANSL: transl,
        GLOBAL_ORIENT: rot[..., :3],
        BODY_POSE: rot[..., 3:66],
        LEFT_HAND_POSE: rot[..., 66:66 + 45] if rot.shape[-1] > 66 else zero_param(t, LEFT_HAND_POSE).to(rot),
        RIGHT_HAND_POSE: rot[..., 66 + 45:66 + 90] if rot.shape[-1] > 66 else zero_param(t, RIGHT_HAND_POSE).to(rot),
        BETAS: zero_param(t, BETAS).to(rot),  # controls the body shape. Body shape is static
        LEYE_POSE: zero_param(t, LEYE_POSE).to(rot),
        REYE_POSE: zero_param(t, REYE_POSE).to(rot),
        EXPRESSION: zero_param(t, EXPRESSION).to(rot),  # controls the face expression
        JAW_POSE: zero_param(t, JAW_POSE).to(rot),  # controls the yaw pose
    }
    return smplx_dict
