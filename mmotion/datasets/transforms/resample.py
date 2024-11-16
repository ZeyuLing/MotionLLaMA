import numpy as np
from typing import Union, List, Dict

import torch
from einops import rearrange
from mmcv import BaseTransform
from torch.nn.functional import interpolate

from mmotion.motion_representation.format_convert import DIM_JOINT_MAPPING
from mmotion.registry import TRANSFORMS
from mmotion.utils.geometry.slerp import motion_sphere_interpolate
from mmotion.utils.smpl_utils.smpl_key_const import ROTATION_KEYS


@TRANSFORMS.register_module()
class LinearResampleFPS(BaseTransform):
    def __init__(self,
                 keys: Union[str, List[str]] = 'motion',
                 fps_key: str = 'fps',
                 tgt_fps: int = 20,
                 ori_fps_key: str = 'ori_fps',
                 output_keys: Union[str, List[str]] = None
                 ):
        """
        Args:
            keys: keys of videos need to be downsampled.
            fps_key: key of video fps which u stored in data_samples
            ori_fps_key: after downsampling,
            the transform will save the original fps to ori_fps_key.
            tgt_fps: target fps of downsampled videos.
            output_keys: where to save the downsampled videos,
            if None, the original video will be covered.
        """
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.fps_key = fps_key
        if output_keys:
            if isinstance(output_keys, str):
                output_keys = [output_keys]
            assert len(output_keys) == len(keys)
        else:
            output_keys = keys

        self.output_keys = output_keys

        self.ori_fps_key = ori_fps_key
        self.tgt_fps = tgt_fps

    @staticmethod
    def resample_linear(motion: torch.Tensor, ori_fps: int, tgt_fps: int):
        """
        :param motion: the motion vector to do linear interpolation.
        Note: this function is only used for non-rotation vectors !!!
        U should use resample_slerp() for rotation vectors.
        :param ori_fps: original fps
        :param tgt_fps: target fps
        :return: The resampled motion vector
        """
        if ori_fps % tgt_fps == 0:
            return motion[::int(ori_fps / tgt_fps)]
        original_frames, chans = motion.shape[:2]
        target_frames = int(original_frames * (tgt_fps / ori_fps))
        motion = rearrange(motion, 't c -> 1 c t')
        motion = interpolate(motion, size=(target_frames), mode='linear', align_corners=False)
        motion = rearrange(motion, '1 c t -> t c')
        return motion

    def resample_motion(self, motion: torch.Tensor, ori_fps: int, tgt_fps: int):
        return self.resample_linear(motion, ori_fps, tgt_fps)

    def transform(self, results: Dict) -> Dict:
        cur_fps = results[self.fps_key]
        if cur_fps == self.tgt_fps:
            return results
        results[self.ori_fps_key] = cur_fps
        results[self.fps_key] = self.tgt_fps
        for key, output_key in zip(self.keys, self.output_keys):
            if key in results:
                video = results[key]
                results[output_key] = self.resample_motion(
                    video, cur_fps, self.tgt_fps)
        return results


@TRANSFORMS.register_module()
class SphereResampleFPS(LinearResampleFPS):
    def __init__(self, keys: Union[str, List[str]] = 'motion', fps_key: str = 'fps', tgt_fps: int = 20,
                 ori_fps_key: str = 'ori_fps', output_keys: Union[str, List[str]] = None, rot_type: str = 'quaternion'):
        """
        Args:
            keys: keys of videos need to be downsampled.
            fps_key: key of video fps which u stored in data_samples
            ori_fps_key: after downsampling,
            the transform will save the original fps to ori_fps_key.
            tgt_fps: target fps of downsampled videos.
            output_keys: where to save the downsampled videos,
            if None, the original video will be covered.
        """
        super().__init__(keys, fps_key, tgt_fps, ori_fps_key, output_keys)
        self.rot_type = rot_type

    @staticmethod
    def resample_slerp(motion: torch.Tensor, rot_type: str, ori_fps: int, tgt_fps: int):
        if ori_fps % tgt_fps == 0:
            return motion[::int(ori_fps / tgt_fps)]
        original_frames, chans = motion.shape[:2]
        target_frames = int(original_frames * (tgt_fps / ori_fps))
        return motion_sphere_interpolate(motion, rot_type, target_frames)

    def transform(self, results: Dict) -> Dict:

        cur_fps = results[self.fps_key]
        if cur_fps == self.tgt_fps:
            return results
        results[self.ori_fps_key] = cur_fps
        results[self.fps_key] = self.tgt_fps
        for key, output_key in zip(self.keys, self.output_keys):
            if key in results:
                motion = results[key]
                results[output_key] = self.resample_slerp(
                    motion, self.rot_type, cur_fps, self.tgt_fps)
        return results


@TRANSFORMS.register_module()
class SmplResampleFPS(SphereResampleFPS):

    def resample_smpl(self, smpl_dict: Dict, cur_fps: int, tgt_fps: int) -> Dict:
        for key, value in smpl_dict.items():
            if key in ROTATION_KEYS:
                smpl_dict[key] = self.resample_slerp(value, self.rot_type, cur_fps, tgt_fps)
            else:
                smpl_dict[key] = self.resample_linear(value, cur_fps, tgt_fps)
        return smpl_dict

    def transform(self, results: Dict) -> Dict:

        cur_fps = results[self.fps_key]
        if cur_fps == self.tgt_fps:
            return results
        results[self.ori_fps_key] = cur_fps
        results[self.fps_key] = self.tgt_fps
        for key, output_key in zip(self.keys, self.output_keys):
            if key in results:
                motion = results[key]
                results[output_key] = self.resample_smpl(
                    motion, cur_fps, self.tgt_fps)
        return results


@TRANSFORMS.register_module()
class MotionResampleFPS(BaseTransform):
    def __init__(self,
                 keys: Union[str, List[str]] = 'motion',
                 fps_key: str = 'fps',
                 tgt_fps: int = 30,
                 ori_fps_key: str = 'ori_fps',
                 output_keys: Union[str, List[str]] = None,
                 data_source: str = 'hm3d',
                 ):
        """
        Args:
            keys: keys of videos need to be downsampled.
            fps_key: key of video fps which u stored in data_samples
            ori_fps_key: after downsampling,
            the transform will save the original fps to ori_fps_key.
            tgt_fps: target fps of downsampled videos.
            output_keys: where to save the downsampled videos,
            if None, the original video will be covered.
        """
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.fps_key = fps_key
        if output_keys:
            if isinstance(output_keys, str):
                output_keys = [output_keys]
            assert len(output_keys) == len(keys)
        else:
            output_keys = keys

        self.output_keys = output_keys

        self.ori_fps_key = ori_fps_key
        self.tgt_fps = tgt_fps

        self.data_source = data_source

    def resample_motion(self, vec: Union[np.ndarray, torch.Tensor], ori_fps, tgt_fps):
        """ hm3d is composed of ra rx rz ry jp jv jr cf
        jr should be sphere interpolated,
        cf should be discrete after linear interpolated,
        others should be linear interpolated
        :param vec: hm3d vector
        :return:
        """
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec)

        resample_vec = LinearResampleFPS.resample_linear(vec, ori_fps, tgt_fps)

        if self.data_source == 'hm3d':
            num_joints = DIM_JOINT_MAPPING[vec.shape[-1]]
            resample_vec[..., -4:] = torch.round(resample_vec[..., -4:])
            resample_vec[..., - ((num_joints-1) * 6 + 4): -4] = SphereResampleFPS.resample_slerp(
                vec[..., - ((num_joints-1) * 6 + 4): -4], 'rotation_6d',
                ori_fps, tgt_fps)
        elif self.data_source == 'tomato':
            resample_vec[..., -4:] = torch.round(resample_vec[..., -4:])
        elif self.data_source == 'interhuman':
            num_joints = DIM_JOINT_MAPPING[vec.shape[-1]]
            resample_vec[..., - (num_joints * 6 + 4): -4] = SphereResampleFPS.resample_slerp(
                vec[..., - (num_joints * 6 + 4): -4], 'rotation_6d',
                ori_fps, tgt_fps)


        return resample_vec

    def transform(self, results: Dict) -> Dict:
        cur_fps = results[self.fps_key]
        if cur_fps == self.tgt_fps:
            return results
        results[self.ori_fps_key] = cur_fps
        results[self.fps_key] = self.tgt_fps
        for key, output_key in zip(self.keys, self.output_keys):
            if key in results:
                motion = results[key]
                results[output_key] = self.resample_motion(
                    motion, cur_fps, self.tgt_fps)
                results['num_frames'] = len(results[output_key])
                results[f'{key}_num_frames'] = len(results[output_key])
        return results
