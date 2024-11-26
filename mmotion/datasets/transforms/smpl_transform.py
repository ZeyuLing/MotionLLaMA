from typing import Tuple, List, Union, Dict, Optional

import torch
from einops import rearrange
from mmcv import BaseTransform

from mmotion.registry import TRANSFORMS
from mmotion.utils.geometry.rotation_convert import rot_convert
from mmotion.utils.smpl_utils.smpl_key_const import GLOBAL_ORIENT, BODY_POSE, LEFT_HAND_POSE, RIGHT_HAND_POSE, \
    TRANSL
from mmotion.utils.smpl_utils.transl import global2local

@TRANSFORMS.register_module()
class SmplxDict2SmplTensor(BaseTransform):
    ROT_TYPES = ['quaternion', 'euler', 'matrix', 'cont6d', 'axis_angle', 'rotation_6d']

    def __init__(
            self,
            keys: Tuple[List[str], str] = ['motion'],
            init_loc_key: str = 'init_loc',
            global2local: bool = False,
            rot_type: str = 'quaternion',
            joints_key: str = 'joints'
    ) -> None:
        """ transform smplx dict to transl, body pose hands pose Tensor
        :param keys: key of smplx dict to be transformed
        :param init_loc_key key to save the first frame location
        :param global2local: if transform global translation to local ones
        """
        self.keys = keys if isinstance(keys, List) else [keys]
        self.global2local = global2local
        self.rot_type = rot_type
        self.init_loc_key = init_loc_key
        self.joints_key = joints_key
        assert rot_type in self.ROT_TYPES, f'rot_type should be one of {self.ROT_TYPES}, but got {rot_type}'

    def smplx_dict2tensor(self, smplx_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """ transform smplx dict to a tensor consist of translation and body, hand rotation
        :param smplx_dict: SMPLX dict including transl and pose
        :return: Tensor including transl and rotation information, location at init frame.
        """
        transl = smplx_dict[TRANSL]
        init_loc = transl[0]

        if self.global2local:
            transl, init_loc = global2local(transl)
        rot = torch.concat([
            smplx_dict[GLOBAL_ORIENT],
            smplx_dict[BODY_POSE]], dim=-1)
        rot = rearrange(rot, 't (j c) -> t j c', c=3)
        rot = rot_convert(rot, 'axis_angle', self.rot_type)
        rot = rearrange(rot, 't j c -> t (j c)')
        res = torch.concat([transl, rot], dim=-1)
        return res, init_loc

    def get_joints(self, joints: torch.Tensor):
        """
        :param joints: whole body joints
        :return: only body and hand, get rid of face
        """
        joints = joints[..., :22, :]
        return joints

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        for key in self.keys:
            if key in results.keys():
                results[f'ori_{key}'] = results[key]
                res_tensor, init_location = self.smplx_dict2tensor(results[key])
                results[key] = res_tensor
                if self.init_loc_key is not None:
                    results[self.init_loc_key] = init_location
        results['rot_type'] = self.rot_type
        results['global2local'] = self.global2local

        if self.joints_key in results.keys():
            results[self.joints_key] = self.get_joints(results[self.joints_key])
        return results


@TRANSFORMS.register_module()
class SmplxDict2SmplhTensor(SmplxDict2SmplTensor):

    def smplx_dict2tensor(self, smplx_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """ transform smplx dict to a tensor consist of translation and body, hand rotation
        :param smplx_dict: SMPLX dict including transl and pose
        :return: Tensor including transl and rotation information, location at init frame.
        """
        transl = smplx_dict[TRANSL]
        init_loc = transl[0]

        if self.global2local:
            transl, init_loc = global2local(transl)
        rot = torch.concat([
            smplx_dict[GLOBAL_ORIENT],
            smplx_dict[BODY_POSE],
            smplx_dict[LEFT_HAND_POSE],
            smplx_dict[RIGHT_HAND_POSE]], dim=-1)
        rot = rearrange(rot, 't (j c) -> t j c', c=3)
        rot = rot_convert(rot, 'axis_angle', self.rot_type)
        rot = rearrange(rot, 't j c -> t (j c)')
        res = torch.concat([transl, rot], dim=-1)
        return res, init_loc

    def get_joints(self, joints: torch.Tensor):
        """
        :param joints: whole body joints
        :return: only body and hand, get rid of face
        """
        body_hand = torch.cat(
            [
                joints[:, :22, :],
                joints[:, 25:55, :]
            ],
            dim=-2
        )
        return body_hand
