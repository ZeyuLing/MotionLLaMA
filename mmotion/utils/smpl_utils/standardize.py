import numpy as np

from mmotion.utils.geometry.angle import angle_normalization
from mmotion.utils.smpl_utils.smpl_key_const import TRANSL, ORIGIN, ROTATION_KEYS, JOINTS


def standardize_smplx_dict(smplx_dict, global2local: bool = False, origin=None):
    """ Two things to do:
        1, Set the first frame root position as original point
        2, Make sure every rotation vector is between (-pi, pi)
    :param global2local: Whether transform coordinate system from global to local ones
    :param smplx_dict: smplx dict
    :param origin: if origin is set and global2local is True, set this as the origin of the local coord system
    :return: standard smplx dict
    """
    if global2local:
        if origin is None:
            origin = smplx_dict[TRANSL][0]
        smplx_dict[TRANSL] = smplx_dict[TRANSL] - origin
        if JOINTS in smplx_dict.keys():
            smplx_dict[JOINTS] -= origin
        smplx_dict[ORIGIN] = origin

    for key, value in smplx_dict.items():
        if key in ROTATION_KEYS:
            value = angle_normalization(value)
            smplx_dict[key] = value
    return smplx_dict
