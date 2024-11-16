import warnings
from random import randint, uniform
from typing import Tuple, Dict, Union, List, Optional

import torch
from mmcv import BaseTransform

from mmotion.registry import TRANSFORMS
from mmotion.utils.logger import print_colored_log


@TRANSFORMS.register_module(force=True)
class RandomCrop(BaseTransform):
    def __init__(self,
                 keys='motion',
                 clip_len: int = 64,
                 start_frame_key='start_frame'):
        """
        Args:
            keys: keys of motion sequence to be cropped
            clip_len: clip length
            start_frame_key: key to save start frame information
        """
        assert keys, 'Keys should not be empty.'
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys
        self.clip_len = clip_len
        self.start_frame_key = start_frame_key

    @staticmethod
    def random_crop(motion: torch.Tensor, clip_len: int = 64, start_frame=None) -> Tuple[torch.Tensor, int]:
        if clip_len >= motion.shape[0]:
            return motion, start_frame
        if start_frame is None:
            start_frame = randint(a=0, b=len(motion) - clip_len)
        return motion[start_frame:start_frame + clip_len], start_frame

    def transform(self, results):
        start_frame = None
        for key in self.keys:
            if key in results:
                cropped, start_frame = self.random_crop(results[key], self.clip_len, start_frame)
                results[key] = cropped
                results[self.start_frame_key] = start_frame
                results['num_frames'] = len(cropped)
                results[f'{key}_num_frames'] = len(cropped)
        return results


@TRANSFORMS.register_module()
class MotionAudioRandomCrop(BaseTransform):
    def __init__(self,
                 motion_keys: str = 'motion',
                 audio_keys: str = 'audio',
                 clip_duration: float = 5.,
                 fps: Union[float, str] = 'fps',
                 sr: Union[int, str] = 'sr',
                 duration_diff_threshold: float = 1.5,                 ):
        """
        :param motion_keys: motion key to crop
        :param audio_keys: audio key to crop
        :param clip_duration: clip motion and audio into clip_duration length
        :param fps: key to load fps value, or directly set fps value of motion
        :param sr: key to load sr value, or directly set sr value of audio
        no cropping is performed.
        """
        if isinstance(motion_keys, str):
            motion_keys = [motion_keys]
        if isinstance(audio_keys, str):
            audio_keys = [audio_keys]
        self.motion_keys = motion_keys
        self.audio_keys = audio_keys
        self.fps = fps
        self.sr = sr
        self.clip_duration = clip_duration
        self.duration_diff_thershold = duration_diff_threshold

    def check_crop(self, results):
        """ Check Crop motion or audio only, or motion-audio paris.
        :param results:
        :return:
        """
        crop_motion = any([key in results for key in self.motion_keys])
        crop_audio = any([key in results for key in self.audio_keys])
        return crop_motion, crop_audio

    def align_duration(self, results):
        """ Align duration of audio and motion pairs.
        :param results: contains audio and motions
        :return:
        """
        motion_lengths = [len(results[key]) for key in self.motion_keys if key in results]
        audio_lengths = [len(results[key]) for key in self.audio_keys if key in results]

        assert all([l == motion_lengths[0] for l in motion_lengths])
        assert all([l == audio_lengths[0] for l in audio_lengths])

        motion_lengths = motion_lengths[0]
        audio_lengths = audio_lengths[0]

        motion_duration = motion_lengths / self.fps
        audio_duration = audio_lengths / self.sr
        aligned_duration = min(motion_duration, audio_duration)
        if abs(motion_duration - audio_duration) > self.duration_diff_thershold:
            print_colored_log('Duration of motion and audio are not aligned, '
                              f"filename: {results.get('interhuman_path')},"
                              f"motion_duration: {motion_duration}"
                              f"audio_duration: {audio_duration}")

        for key in self.motion_keys:
            motion = results.get(key)
            if motion is None:
                continue
            motion = motion[:int(aligned_duration * self.fps)]
            results[f'{key}_duration'] = aligned_duration
            results['duration'] = aligned_duration
            results['num_frames'] = len(motion)
            results[f'{key}_num_frames'] = len(motion)
            results[key] = motion

        for key in self.audio_keys:
            audio = results.get(key)
            if audio is None:
                continue
            audio = audio[:int(aligned_duration * self.sr)]
            results[key] = audio
            results['audio_duration'] = aligned_duration
            results['audio_num_frames'] = len(audio)
            results[f'{key}_num_frames'] = len(audio)

        return results

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if isinstance(self.fps, str):
            self.fps = results.get(self.fps, 30)
        if isinstance(self.sr, str):
            self.sr = results.get(self.sr, 24000)

        # make sure motion and audio share the same duration
        if_crop_motion, if_crop_audio = self.check_crop(results)
        if not if_crop_motion and not if_crop_audio:
            return results

        if if_crop_motion and if_crop_audio:
            results = self.align_duration(results)

        cur_duration = results.get('duration') if if_crop_motion else results.get('audio_duration')

        if cur_duration <= self.clip_duration:
            return results

        start_timestamp = uniform(a=0, b=cur_duration - self.clip_duration)

        motion_start_frame = int(start_timestamp * self.fps)
        audio_start_frame = int(start_timestamp * self.sr)

        for key in self.motion_keys:
            motion = results.get(key, None)
            if motion is not None:
                ori_num_frames = motion.shape[0]
                motion = motion[motion_start_frame: motion_start_frame + int(self.fps * self.clip_duration)]
                results[key] = motion
                assert len(motion) > 0, (motion.shape, ori_num_frames, cur_duration, start_timestamp)
                results['num_frames'] = len(motion)
                results[f'{key}_num_frames'] = len(motion)
                results['duration'] = len(motion) / self.fps

        for key in self.audio_keys:
            audio = results.get(key)
            if audio is not None:
                audio_duration = audio.shape[0] / self.sr
                if audio_duration > self.clip_duration:
                    audio = audio[audio_start_frame: audio_start_frame + int(self.sr * self.clip_duration)]
                results[key] = audio
                results['audio_duration'] = audio.shape[0] / self.sr
                results['audio_num_frames'] = audio.shape[0]
                results[f'{key}_num_frames'] = audio.shape[0]

        return results
