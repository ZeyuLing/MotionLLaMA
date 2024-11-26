import numpy as np

DIM_JOINT_MAPPING = {
    251: 21,  # KIT-ML
    263: 22,  # humanml3d
    313: 52,  # tomato whole body
    623: 52,  # humanml3d whole body
    628: 52,  # interhuman whole body
}

ROOT_A = 'root_a'
ROOT_XZ = 'root_xz'
ROOT_Y = 'root_y'

POSITION = 'pos'
POSITION_BODY = 'pos_body'
POSITION_HAND = 'pos_hand'

ROTATION = 'rot'
ROTATION_BODY = 'rot_body'
ROTATION_HAND = 'rot_hand'

VELOCITY = 'vel'
VELOCITY_BODY = 'vel_body'
VELOCITY_HAND = 'vel_hand'

FEET_CONTACT = 'fc'

HUMANML3D_COMPONENTS = [ROOT_A, ROOT_XZ, ROOT_Y, POSITION, VELOCITY, ROTATION, FEET_CONTACT]
TOMATO_COMPONENTS = [ROOT_A, ROOT_XZ, ROOT_Y, POSITION_BODY, POSITION_HAND, VELOCITY_BODY, VELOCITY_HAND]
INTERHUMAN_COMPONENTS = [POSITION, VELOCITY, ROTATION, FEET_CONTACT]

BIAS_KEYS=[ROOT_A, ROOT_XZ, ROOT_Y, FEET_CONTACT]
def hm3d2tomato(hm3d: np.ndarray, no_vel: bool = False) -> np.ndarray:
    """ hm3d including ra rx rz ry jp jv jr cf
        tomato including ra rx rz ry jp jv (in our implement, we don't use face)
    :param hm3d: humanml3d vec. [t, c] for smpl, vec is 263-dim, for smpl-h, vec is 623-dim
    :return:
    """
    num_joints = DIM_JOINT_MAPPING[hm3d.shape[-1]]
    tomato = hm3d[:, :-(num_joints - 1) * 6 + 4]
    if no_vel:
        tomato = tomato[:, :-num_joints * 3]
    return tomato
