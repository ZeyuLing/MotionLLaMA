# Copyright (c) OpenMMLab. All rights reserved.
import random
from os.path import exists
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import torch
import torchaudio
from einops import rearrange, repeat
from mmcv.transforms import BaseTransform
from praatio.textgrid import openTextgrid
from mmotion.registry import TRANSFORMS
from mmotion.utils.dataset_utils.hm3d_utils import hm3d_pattern
from mmotion.utils.files_io.json import read_json
from mmotion.utils.files_io.txt import read_txt
from mmotion.utils.geometry.rotation_convert import rot_convert
from mmotion.motion_representation import hm3d2tomato
from mmotion.utils.smpl_utils.smpl_key_const import SMPLX_KEYS, ROTATION_KEYS, SHAPE_KEYS, TRANSL, zero_param, \
    BODY_POSE, GLOBAL_ORIENT, JAW_POSE, LEFT_HAND_POSE, RIGHT_HAND_POSE, LEYE_POSE, REYE_POSE, param_dim, \
    EXPRESSION, BETAS, ORIGIN
from mmotion.utils.smpl_utils.transl import global2local
from mmotion.utils.task.prompt.prompt_template import sample_template


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


@TRANSFORMS.register_module(force=True)
class LoadHm3dTxt(BaseTransform):


    def __init__(self, keys: Union[str, List[str]] = 'caption', min_duration=0, sr=None):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

        self.sr = sr
        self.min_duration = min_duration

    def transform(self, results: dict) -> dict:
        """Functions to load humanml3d caption text.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded caption, token, etc.
        """
        for key in self.keys:
            filename = results.get(f'{key}_path')
            if filename is None or not exists(filename):
                continue

            caption_list, pos_list, range_list = self.load_caption(filename)
            # 0 <= idx <= num_captions - 1
            select_idx = random.randint(0, len(caption_list) - 1)
            caption = caption_list[select_idx]
            pos = pos_list[select_idx]
            range = range_list[select_idx]

            results[key] = caption
            results[f'{key}_pos'] = pos
            results[f'{key}_range'] = range

            results[f'{key}_list'] = caption_list
            # pos: part of speech
            results[f'{key}_pos_list'] = pos_list
            results[f'{key}_range_list'] = range_list

        return results

    @staticmethod
    def judge_hm3d(content: str):
        """ Judge if the content is a humanml3d type caption file
        :param content: content of file
        :return: True or False
        """
        content = content.strip()

        first_line = content.split('\n')[0]
        if hm3d_pattern.match(first_line):
            return True
        return False

    def load_hm3d_caption(self, content: str):
        caption_list = []
        pos_list = []
        range_list = []

        for line in content.split('\n'):
            caption = line.split('#')[0].strip()
            assert len(caption) > 0, content
            pos = line.split('#')[1].strip()

            range = line.split('#')[-2:]
            range = [float(x) for x in range]
            duration = range[1] - range[0]
            # duration == 0 means no crop occurs.
            if 0 < duration < self.min_duration:
                continue

            caption_list.append(caption)
            pos_list.append(pos)
            range_list.append(range)
        return caption_list, pos_list, range_list

    @staticmethod
    def load_pure_caption(content: str):
        caption_list = []
        pos_list = []
        range_list = []
        for line in content.split('\n'):
            caption = line.strip()

            caption_list.append(caption)

            pos_list.append(None)
            range_list.append([0, 0])
        return caption_list, pos_list, range_list

    def load_caption(self, caption_path: str) -> Tuple:
        """
        :param caption_path: txt path of humanml3d caption file.
        :return: caption list, pos list and range list
        """
        try:
            content = read_txt(caption_path).strip()
        except:
            raise Exception(caption_path)
        is_hm3d = self.judge_hm3d(content)
        if is_hm3d:
            caption_list, pos_list, range_list = self.load_hm3d_caption(content)
        else:
            caption_list, pos_list, range_list = self.load_pure_caption(content)

        return caption_list, pos_list, range_list

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'key={self.key})')

        return repr_str


@TRANSFORMS.register_module(force=True)
class LoadPureTxt(BaseTransform):
    def __init__(self, key: str = 'speech_script'):
        self.key = key

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        filename = results[f'{self.key}_path']

        text = read_txt(filename)

        results[self.key] = text
        return results


@TRANSFORMS.register_module(force=True)
class LoadAudio(BaseTransform):
    def __init__(self, keys: Union[List[str], str] = ['audio'], sr: Union[int, str] = 'sr'):
        """
        :param keys: keys of audio need to be loaded
        :param sr: 1) if sr is str, it means the key of sr in the input dict,
         all loaded sr will be transformed to the sr saved in the dict.
         2) if sr in int, transform all the loaded audio to this sr.
        """
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.sr = sr

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if isinstance(self.sr, str):
            self.sr = results.get(self.sr)

        results['sr'] = self.sr

        for key in self.keys:
            filename = results.get(f'{key}_path')
            if filename is None:
                continue
            # the loaded sr of audio is raw or determined by hyperparameter
            audio, sr = torchaudio.load(filename)
            audio = convert_audio(audio, sr, self.sr, 1).squeeze(0)

            results[key] = audio
            results[f'{key}_ori_sr'] = sr
            results['audio_num_frames'] = audio.shape[0]
            results['audio_duration'] = audio.shape[0] / self.sr
            results['ori_audio_duration'] = audio.shape[0] / self.sr

        return results


@TRANSFORMS.register_module(force=True)
class LoadSmplx322Npy(BaseTransform):
    def __init__(self, key: str = 'motion', duration_key: str = 'duration', fps_key: str = 'fps', save_key: str = None):
        """
        :param key: where to load 322-dim smplx numpy file(In motionx data format)
        """
        self.key = key
        self.duration_key = duration_key
        self.fps_key = fps_key
        self.save_key = save_key or key

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        filename = results[f'{self.key}_path']
        smplx_322 = torch.from_numpy(np.load(filename))
        # when determine dict keys, we reference to smplx repo.
        num_frames = smplx_322.shape[0]
        smplx_dict = {
            GLOBAL_ORIENT: smplx_322[:, :param_dim[GLOBAL_ORIENT]],  # controls the global root orientation
            BODY_POSE: smplx_322[:, 3:3 + param_dim[BODY_POSE]],  # controls the body
            LEFT_HAND_POSE: smplx_322[:, 66:66 + param_dim[LEFT_HAND_POSE]],  # controls the finger articulation
            RIGHT_HAND_POSE: smplx_322[:, 66 + 45:66 + 45 + param_dim[RIGHT_HAND_POSE]],
            # controls the finger articulation
            JAW_POSE: smplx_322[:, 66 + 90:66 + 90 + param_dim[JAW_POSE]],  # controls the yaw pose
            EXPRESSION: smplx_322[:, 159:159 + param_dim[EXPRESSION]],  # controls the face expression
            TRANSL: smplx_322[:, 309:309 + param_dim[TRANSL]],  # controls the global body position
            BETAS: smplx_322[:, 312:312 + param_dim[BETAS]],  # controls the body shape. Body shape is static
            LEYE_POSE: zero_param(num_frames, LEYE_POSE).to(smplx_322),
            REYE_POSE: zero_param(num_frames, REYE_POSE).to(smplx_322),
        }

        results[self.save_key] = smplx_dict
        results[self.duration_key] = num_frames * 1. / results[self.fps_key]
        return results


@TRANSFORMS.register_module(force=True)
class LoadSmpl(BaseTransform):
    KEY_MAPPING = dict()

    def __init__(self, key: str = 'motion',
                 duration_key: str = 'duration',
                 fps_key: str = 'fps',
                 key_mapping: Dict = dict(),
                 rot_type='axis_angle',
                 joints_key: str = 'joints',
                 rot_type_key='rot_type',
                 global2local=False,
                 global2local_key: str = 'global2local',
                 split_from_body_list=[],
                 save_key=None):
        """
        :param key: where to load smplx npz or pkl file.
        """
        self.key = key
        self.save_key = save_key or key
        self.duration_key = duration_key
        self.fps_key = fps_key
        self.KEY_MAPPING = key_mapping or self.KEY_MAPPING
        self.rot_type = rot_type
        self.rot_type_key = rot_type_key
        self.split_from_body_list = split_from_body_list
        self.joints_key = joints_key
        self.global2local = global2local
        self.global2local_key = global2local_key

    def convert_rotation_type(self, smpl_dict):
        for key, param in smpl_dict.items():
            if key in ROTATION_KEYS:
                if key != GLOBAL_ORIENT:
                    param = rearrange(param, 't (j c) -> t j c', c=3)
                param = rot_convert(param, 'axis_angle', self.rot_type)
                if key != GLOBAL_ORIENT:
                    param = rearrange(param, 't j c -> t (j c)')
                smpl_dict[key] = param
        return smpl_dict

    def load(self, input_dict: Dict):
        """
        :param input_dict: load from files(pkl npz json) in dataset.
        :return: standard smplx dict, rotation in axis angle
        """
        transl_key = TRANSL if TRANSL in input_dict.keys() else self.KEY_MAPPING[TRANSL]
        smplx_dict = {
            TRANSL: torch.tensor(input_dict[transl_key])
        }
        num_frames = len(input_dict[transl_key])

        for key in SMPLX_KEYS:
            input_key = key if key in input_dict.keys() else self.KEY_MAPPING.get(key)
            if input_key is None:
                param = zero_param(num_frames, key)
            else:
                param = torch.tensor(input_dict[input_key])

            smplx_dict[key] = param

            if key in SHAPE_KEYS:
                assert len(param.shape) <= 2, f'Shape params of smpl should in 1- or 2-dim, but got {param.shape}'
                if len(param.shape) == 1:
                    param = repeat(param, 'c -> n c', n=num_frames)
                else:
                    if param.shape[0] != num_frames:
                        assert param.shape[0] == 1, param.shape
                        param = repeat(param, '1 c -> n c', n=num_frames)
                smplx_dict[key] = param[..., :param_dim[key]]

        if GLOBAL_ORIENT in self.split_from_body_list:
            smplx_dict[GLOBAL_ORIENT] = smplx_dict[BODY_POSE][..., :3]
            smplx_dict[BODY_POSE] = smplx_dict[BODY_POSE][..., 3:]

        if LEFT_HAND_POSE in self.split_from_body_list:
            smplx_dict[LEFT_HAND_POSE] = smplx_dict[BODY_POSE][..., -90:-45]
            smplx_dict[RIGHT_HAND_POSE] = smplx_dict[BODY_POSE][..., -45:]
            smplx_dict[BODY_POSE] = smplx_dict[BODY_POSE][..., :-90]

        if JAW_POSE in self.split_from_body_list:
            smplx_dict[JAW_POSE] = smplx_dict[BODY_POSE][..., 63:66]
            smplx_dict[LEYE_POSE] = smplx_dict[BODY_POSE][..., 66:69]
            smplx_dict[REYE_POSE] = smplx_dict[BODY_POSE][..., 69:72]
            smplx_dict[BODY_POSE] = torch.concat([smplx_dict[BODY_POSE][..., :63],
                                                  smplx_dict[BODY_POSE][..., 72:]], dim=-1)
        # SMPL body params are
        smplx_dict[BODY_POSE] = smplx_dict[BODY_POSE][..., :63]
        return smplx_dict

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        filename = results[f'{self.key}_path']
        results[self.global2local_key] = self.global2local
        if filename.endswith('.npz') or filename.endswith('.pkl'):
            input_dict = np.load(filename, allow_pickle=True)
        else:
            assert filename.endswith('.json')
            input_dict = read_json(filename)
        results[self.rot_type_key] = self.rot_type
        if self.joints_key in input_dict:
            results[self.joints_key] = input_dict[self.joints_key]

        smplx_dict = self.load(input_dict)
        smplx_dict = self.convert_rotation_type(smplx_dict)
        if self.global2local:
            smplx_dict[TRANSL], results[ORIGIN] = global2local(smplx_dict[TRANSL])

        results[self.save_key] = smplx_dict

        num_frames = len(smplx_dict[GLOBAL_ORIENT])

        results[self.duration_key] = num_frames * 1. / results[self.fps_key]
        return results


@TRANSFORMS.register_module(force=True)
class LoadMotionVector(BaseTransform):
    def __init__(self, keys: Union[str, List] = ['motion'],
                 duration_key: str = 'duration',
                 fps_key: str = 'fps',
                 data_source: str = 'hm3d',
                 num_joints=52,
                 range_key=None,
                 save_keys=None):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

        if isinstance(save_keys, str):
            save_keys = [save_keys]
        self.save_keys = save_keys or keys

        assert len(self.save_keys) == len(keys), f'Number of save keys should be equal to keys'
        self.duration_key = duration_key
        self.fps_key = fps_key
        self.data_source = data_source
        self.range_key = range_key
        self.num_joints = num_joints

    def load(self, raw_vec: np.ndarray) -> np.ndarray:
        """
        :param raw_vec: t, c
        :return:
        """

        if self.data_source in ['hm3d', 'interhuman']:
            return raw_vec
        if self.data_source == 'tomato':
            return hm3d2tomato(raw_vec)
        if self.data_source == 'tomato_no_vel':
            return hm3d2tomato(raw_vec, no_vel=True)
        if self.data_source in ['uniform_joints', 'global_uniform_joints']:
            raw_vec = rearrange(raw_vec, 't j c -> t (j c)')
            return raw_vec
        raise NotImplementedError(f'Unsupported data source {self.data_source}')

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        for key, save_key in zip(self.keys, self.save_keys):
            filename = results.get(f'{key}_path', None)
            if filename is None:
                continue
            motion = np.load(filename)
            assert not np.isnan(motion).any(), filename
            motion = self.load(motion)
            if self.range_key is not None:
                start, end = results[self.range_key]
                if end > 0:
                    start = int(start * results[self.fps_key])
                    end = int(end * results[self.fps_key])
                    motion = motion[start: end]
            assert len(motion) >= 4, (f'{filename} too short, only {len(motion)},'
                                      f' if it is in humanml3d dataset, check all corresponding captions')

            results[save_key] = torch.tensor(motion).float()
            results[f'{save_key}_path'] = filename
            results['data_source'] = self.data_source
            results['num_joints'] = self.num_joints
            num_frames = len(motion)
            results[self.duration_key] = num_frames * 1. / results[self.fps_key]
            results[f'{save_key}_{self.duration_key}'] = num_frames * 1. / results[self.fps_key]
            results['num_frames'] = num_frames
            results[f'{save_key}_num_frames'] = num_frames

        return results


@TRANSFORMS.register_module(force=True)
class LoadScript(BaseTransform):
    def __init__(self, keys: Union[List[str], str] = 'script', tier_name='words'):
        super().__init__()
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.tier_name = tier_name

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        for key in self.keys:
            filename = results.get(f'{key}_path', None)
            if filename is None:
                continue
            text_grid = openTextgrid(filename, includeEmptyIntervals=True, reportingMode='silence')
            intervals = text_grid._tierDict[self.tier_name].entries
            results[key] = intervals
        return results


@TRANSFORMS.register_module(force=True)
class LoadConversation(BaseTransform):
    def __init__(self, key='conversation', task_key='task'):
        self.task_key = task_key
        self.key = key

    def transform(self, results: Dict) -> Dict:
        """
        :param results: data info dict for a single sample
        :return:
        """
        task = results.get(self.task_key)
        conversation_template: List[List[Dict]] = sample_template(task)
        results[self.key] = conversation_template
        return results
