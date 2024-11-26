"""
    This file enables transform the loaded motion vectors from one data source to another
"""
from typing import Union, List, Dict, Optional, Tuple
from warnings import warn

import torch
from einops import rearrange
from mmcv import BaseTransform

from mmotion.registry import TRANSFORMS
from mmotion.motion_representation import DIM_JOINT_MAPPING


@TRANSFORMS.register_module()
class RelativeRootTransform(BaseTransform):
    """
        Transform the global joints coordinates to ones relative to the root position.
    """

    def __init__(self, keys: Union[str, List[str]]='motion'):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
    @staticmethod
    def global2relative(joints:torch.Tensor, xz=True):
        """
        :param joints:
        :param xz: if true, only modify xz coordinates, else modify xyz
        :return:
        """
        joints = rearrange(joints, 't (j c) -> t j c', c=3)
        if xz:
            joints[:, 1:, [0, 2]] -= joints[:, :1, [0, 2]]
        else:
            joints[:, 1:] -= joints[:, :1]
        joints = rearrange(joints, 't j c -> t (j c)')
        return joints
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        for key in self.keys:
            joints = results[key]
            # relative to the root joint
            joints= self.global2relative(joints)
            results[key] = joints
        results['data_source'] = 'relative_' + results['data_source']
        return results

@TRANSFORMS.register_module()
class InterhumanTransform(BaseTransform):
    """
        1. Transform the joints coordinates to ones relative to the root keypoint
        2. Get rid of any information in the representation, including joints, rotation, velocity and feet contact
    """
    def __init__(self, keys: Union[str, List[str]]='motion',
                 relative_joints: Union[bool, str]=False,
                 no_rotation: bool = False,
                 no_joints: bool=False,
                 root_only=False,
                 no_velocity:bool =False,
                 no_feet_contact: bool=False,
                 ):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.relative_joints = relative_joints
        # xyz or xz
        if self.relative_joints is True:
            self.relative_joints = 'xyz'

        self.no_rotation = no_rotation
        self.no_joints = no_joints
        self.no_velocity = no_velocity
        self.no_feet_contact = no_feet_contact
        self.root_only = root_only

        assert not (root_only and no_joints),"if you want to use the root keypoint, don't set 'no_joints' to True"
        assert not (self.no_joints and self.no_velocity and self.no_rotation), \
            "Joints, Velocity and Rotations should not all be None"

        if self.no_joints and relative_joints:
            warn('No joints information is included, so relative_joints param will be ignored')
    def get_data_source(self):
        data_source='interhuman'
        if self.relative_joints:
            data_source=f'relative_{self.relative_joints}_interhuman'
        if self.root_only:
            data_source='transl'
        if self.no_joints:
            data_source+='_no_joints'
        if self.no_velocity:
            data_source+='_no_velocity'
        if self.no_rotation:
            data_source+='_no_rotation'
        if self.no_feet_contact:
            data_source+='_no_feet_contact'
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        new_data_source = self.get_data_source()
        results['ori_data_source'] = 'interhuman'
        results['data_source'] = new_data_source
        for key in self.keys:
            if key not in results:
                continue
            motion = results[key]
            T, num_channels = motion.shape
            num_joints = DIM_JOINT_MAPPING[num_channels]

            joints = motion[...,:num_joints*3]
            if self.root_only:
                joints = joints[..., :3]
            vel = motion[..., num_joints*3:num_joints*6]
            rotation = motion[..., num_joints*6:num_joints*12]
            feet_contact = motion[..., -4:]

            if self.relative_joints is not False:
                joints = RelativeRootTransform.global2relative(joints, xz=self.relative_joints=='xz')

            new_motion = torch.empty([T, 0])
            if not self.no_joints:
                new_motion = torch.cat([new_motion, joints], dim=-1)
            if not self.no_velocity:
                new_motion = torch.cat([new_motion, vel], dim=-1)
            if not self.no_rotation:
                new_motion = torch.cat([new_motion, rotation], dim=-1)
            if not self.no_feet_contact:
                new_motion = torch.cat([new_motion, feet_contact], dim=-1)
            results[key] = new_motion
            results[f'{key}_data_source'] = new_data_source
        return results


