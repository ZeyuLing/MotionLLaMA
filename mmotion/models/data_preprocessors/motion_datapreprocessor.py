import numpy as np
from typing import Union, List, Optional, Tuple, Dict

import torch
from mmengine import is_seq_of
from mmengine.model import BaseDataPreprocessor
from mmengine.model.base_model.data_preprocessor import CastData
from torch import Tensor

from mmotion.models.data_preprocessors.normalizer import BaseMotionNormalizer
from mmotion.registry import MODELS, FUNCTIONS
from mmotion.structures import DataSample
from mmotion.utils.typing import SampleList


@MODELS.register_module()
class MotionDataPreprocessor(BaseDataPreprocessor):
    _MOTION_KEYS = []
    _AUDIO_KEYS = []
    _NON_CONCATENATE_KEYS = ['fps']

    def __init__(self,
                 normalizer: dict = dict(
                     type='BaseMotionNormalizer',
                     norm_path='data/motionhub/statistics/interhuman.pkl'),
                 pad_module: Dict = None,
                 motion_keys: Optional[Tuple[str, List[str]]] = ['motion'],
                 audio_keys: Optional[Tuple[str, List[str]]] = ['audio', 'music'],
                 non_concatenate_keys: Optional[Tuple[str, List[str]]] = ['message'],
                 vec2joints_fn: dict = {'type': 'hm3d2joints'},
                 vec2rotation_fn: dict = {'type': 'hm3d2rotation'},
                 stack_data_sample: bool = True,
                 ):
        """
        :param norm: normalizer config
        :param enable_norm: whether to do normalization for motion data
        :param motion_keys: motion keys
        :param non_concatenate_keys: for non_concatenate data, we do no operations to collate them,
         keep them as a list in input dict and data_samples.
        :param stack_data_sample:
        """
        super().__init__()
        # normalization for motions
        self.normalizer: BaseMotionNormalizer = MODELS.build(normalizer)

        # padding for both motion and audio
        self._enable_pad = pad_module is not None
        if self._enable_pad:
            self.pad_module = MODELS.build(pad_module)

        # motion keys
        if motion_keys is not None:
            if not isinstance(motion_keys, list):
                motion_keys = [motion_keys]
            self._MOTION_KEYS += motion_keys

        # audio keys
        if audio_keys is not None:
            if not isinstance(audio_keys, list):
                audio_keys = [audio_keys]
            self._AUDIO_KEYS += audio_keys

        if non_concatenate_keys is not None:
            if not isinstance(non_concatenate_keys, list):
                non_concatenate_keys = [non_concatenate_keys]
            self._NON_CONCATENATE_KEYS += non_concatenate_keys

        self.stack_data_sample = stack_data_sample

        self.vec2joints_fn = FUNCTIONS.get(vec2joints_fn)
        self.vec2rotation_fn = FUNCTIONS.get(vec2rotation_fn)

    def destruct(self,
                 outputs: Union[Tensor, List[Tensor]],
                 data_samples: Union[SampleList, DataSample, None] = None,
                 key=None) -> Union[list, Tensor]:
        """ undo normalization and padding
        :param key: need to destruct
        :param outputs: motion tensor to destruct. T, J, C or B, T, J, C or list of [T, J, C]
        :param data_samples: data samples
        :return: un-normalized motion tensor
        """
        outputs = self.normalizer.inv_normalize(outputs)
        outputs = self._undo_pad_motion(outputs, data_samples, key)

        return outputs

    def do_pad(self, inputs: Union[torch.Tensor, List[torch.Tensor]],
                data_samples: SampleList, key=None):
        """
        :param inputs: a batch of input needs to be padded
        :param data_samples: data samples
        :return: padded inputs, data samples updated with padding infomation
        """
        if not self._enable_pad:
            return inputs, data_samples
        inputs, pad_infos = self.pad_module(inputs)
        if data_samples is not None:
            for data_sample, pad_info in zip(data_samples, pad_infos):
                if key is not None:
                    pad_info = {f'{key}_{ori_key}': value for ori_key, value in pad_info.items()}

                data_sample.set_metainfo(pad_info)

        return inputs, data_samples

    def _undo_pad(self, inputs: Union[torch.Tensor, List[torch.Tensor]],
                  data_samples: Union[DataSample, SampleList] = None,
                  key=None):
        """
        :param inputs: Padded motion vector. T C or B T C or list of [T, C]
        :param data_samples:
        :return:
        """

        if data_samples is None or not self._enable_pad:
            # if padding info is not provided, return
            return inputs

        is_batch = True
        if isinstance(inputs, torch.Tensor) and inputs.ndim == 2:
            is_batch = False
        if not is_batch:
            inputs = [inputs]

        batch_size = len(inputs)
        num_frames_key = 'num_frames' if key is None else f"{key}_num_frames"
        if isinstance(data_samples, list):
            num_frames = [data_sample.get(num_frames_key, data_sample.get('num_frames'))
                          for data_sample in data_samples]
        else:
            num_frames = data_samples.get(num_frames_key, data_samples.get('num_frames'))
        if isinstance(num_frames, int):
            num_frames = [num_frames] * batch_size

        inputs = [m[:nf] for m, nf in zip(inputs, num_frames)]
        if not is_batch:
            inputs = inputs[0]
        return inputs

    def _undo_pad_motion(self, inputs: Union[torch.Tensor, List[torch.Tensor]],
                         data_samples: Union[DataSample, SampleList] = None,
                         motion_key=None):
        """
        :param inputs: Padded motion vector. T C or B T C or list of [T, C],
         some items in the inputs can be None, If so, the function will keep None at that position.
        :param data_samples:
        :return:
        """

        if data_samples is None or not self._enable_pad:
            # if padding info is not provided, return
            return inputs

        is_batch = True
        if isinstance(inputs, torch.Tensor) and inputs.ndim == 2:
            is_batch = False
        if not is_batch:
            inputs = [inputs]

        batch_size = len(inputs)
        num_frames_key = 'num_frames' if motion_key is None else f"{motion_key}_num_frames"
        if isinstance(data_samples, list):
            num_frames = [data_sample.get(num_frames_key, data_sample.get('num_frames'))
                          for data_sample in data_samples]
        else:
            num_frames = data_samples.get(num_frames_key, data_samples.get('num_frames'))
        if isinstance(num_frames, int):
            num_frames = [num_frames] * batch_size

        inputs = [m[:nf] if m is not None else None for m, nf in zip(inputs, num_frames)]
        if not is_batch:
            inputs = inputs[0]
        return inputs

    def _undo_pad_audio(self, inputs: Union[torch.Tensor, List[torch.Tensor]],
                        data_samples: Union[DataSample, SampleList] = None, audio_key=None):
        """
        :param inputs: Padded motion vector. T C or B T C or list of [T, C]
        :param data_samples:
        :return:
        """

        if data_samples is None or not self._enable_pad:
            # if padding info is not provided, return
            return inputs

        is_batch = True
        if isinstance(inputs, torch.Tensor) and inputs.ndim == 1:
            is_batch = False
        if not is_batch:
            inputs = [inputs]

        batch_size = len(inputs)
        num_frames_key = 'audio_num_frames' if audio_key is None else f"{audio_key}_num_frames"
        if isinstance(data_samples, list):
            num_frames = [data_sample.get(num_frames_key, data_sample.get('audio_num_frames'))
                          for data_sample in data_samples]
        else:
            num_frames = data_samples.get(num_frames_key, data_samples.get('audio_num_frames'))
        if isinstance(num_frames, int):
            num_frames = [num_frames] * batch_size

        inputs = [m[:nf] if m is not None else None for m, nf in zip(inputs, num_frames)]
        if not is_batch:
            inputs = inputs[0]
        return inputs

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, (str, int, float)):
            return data
        return super().cast_data(data)

    def forward(self, data: dict, training: bool = False) -> dict:
        """

        Args:
            data (dict): Input data to process.
            training (bool): Whether to in training mode. Default: False.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        _batch_data_samples: SampleList = data.get('data_samples', None)
        assert isinstance(_batch_inputs, dict) or is_seq_of(_batch_inputs, dict), \
            f'Expected dict or list of dict but got {type(_batch_inputs)}'

        if is_seq_of(_batch_inputs, dict):
            # convert list of dict to dict of list
            keys = _batch_inputs[0].keys()
            assert all([inp.keys() == keys for inp in _batch_inputs]), 'the input list of dicts must have same keys'
            _batch_inputs = {k: [inp[k] for inp in _batch_inputs] for k in keys}

        _batch_inputs, _batch_data_samples = \
            self._preprocess_dict_inputs(
                _batch_inputs, _batch_data_samples)

        data['inputs'] = _batch_inputs

        # process data samples
        if _batch_data_samples:
            _batch_data_samples = self._preprocess_data_sample(_batch_data_samples)
        data['data_samples'] = _batch_data_samples
        return data

    def _preprocess_motion_batch(self,
                                 batch_motion: Union[List[Tensor], Tensor],
                                 data_samples: Optional[SampleList] = None,
                                 input_key=None) -> Tuple[Tensor, SampleList]:
        """
        :param batch_motion: Motion tensor batch to be preprocessed. can be a list or a tensor(determined by dataloader collate_fn)
        :param data_samples: DataSample List to store additional infos for each sample
        :param input_key: if provided, judge whether the input is motion, if not, directly return
        :return:  Tuple[Tensor, List[DataSample]]: The preprocessed motion tensor
                and updated data samples.
        """
        if input_key is not None and input_key not in self._MOTION_KEYS:
            return batch_motion, data_samples

        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(len(batch_motion))]

        dim = batch_motion[0].dim()
        assert all([
            tensor.ndim == dim for tensor in batch_motion
        ]), ('Expected the dimensions of all tensors must be the same, '
             f'but got {[tensor.ndim for tensor in batch_motion]}')

        batch_motion, data_samples = self.do_norm(batch_motion, data_samples, input_key)
        batch_motion, data_samples = self.do_pad(batch_motion, data_samples, input_key)
        if isinstance(batch_motion, list):
            batch_motion = torch.stack(batch_motion, dim=0)
        return batch_motion, data_samples

    def _preprocess_audio_batch(self,
                                batch_audio: Union[List[Tensor], Tensor],
                                data_samples: Optional[SampleList],
                                input_key=None) -> Tuple[Tensor, SampleList]:
        """
        :param batch_audio: audio tensor batch to be preprocessed. can be a list or a tensor(determined by dataloader collate_fn)
        :param data_samples: DataSample List to store additional infos for each sample
        :param input_key: if provided, judge whether the input is motion, if not, directly return
        :return:  Tuple[Tensor, List[DataSample]]: The preprocessed motion tensor
                and updated data samples.
        """
        if input_key is not None and input_key not in self._AUDIO_KEYS:
            return batch_audio, data_samples

        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(len(batch_audio))]

        batch_audio, data_samples = self.do_pad(batch_audio, data_samples, input_key)
        if isinstance(batch_audio, list):
            batch_audio = torch.stack(batch_audio, dim=0)
        return batch_audio, data_samples

    def _preprocess_other_batch(self, batch_inputs, data_samples, input_key=None):
        """Processes the input list based on the following criteria:

        1. If each item in the input list is a tensor or ndarray, it stacks them into a tensor if their shapes are consistent.
        2. If the input list contains numbers, it converts them into a tensor.
        3. If the input list contains dictionaries and all dictionaries have the same keys,
           it converts the list of dictionaries into a dictionary of lists.
        4. In other cases, the input list is returned unchanged.

        :param batch_inputs: a batch of input items in input batch dict
        :param data_samples: data_samples
        :param input_key: the key of the input
        :return: preprocessed list and updated data samples
        """
        if input_key in self._NON_CONCATENATE_KEYS:
            return batch_inputs, data_samples

        if all(isinstance(item, (torch.Tensor, np.ndarray)) for item in batch_inputs):
            shapes = [item.shape for item in batch_inputs]
            if all(shape == shapes[0] for shape in shapes):
                batch_inputs = torch.stack(
                    [torch.tensor(item) if isinstance(item, np.ndarray) else item for item in batch_inputs])

        elif all(isinstance(item, (int, float)) for item in batch_inputs):
            batch_inputs = torch.tensor(batch_inputs)

        elif all(isinstance(item, dict) for item in batch_inputs):
            keys = batch_inputs[0].keys()
            if all(item.keys() == keys for item in batch_inputs):
                batch_inputs = {key: [d[key] for d in batch_inputs] for key in keys}

        return batch_inputs, data_samples

    def _preprocess_dict_inputs(self,
                                batch_inputs: dict,
                                data_samples: Optional[SampleList] = None
                                ) -> Tuple[dict, SampleList]:
        """Preprocess dict type inputs.

        Args:
            batch_inputs (dict): Input dict.
            data_samples (List[DataSample], optional): The data samples
                of corresponding inputs. If not passed, a list of empty data
                samples will be initialized to save metainfo. Defaults to None.

        Returns:
            Tuple[dict, List[DataSample]]: The preprocessed dict and
                updated data samples.
        """
        for k, inputs in batch_inputs.items():
            # handle concentrate for values in list
            inputs, data_samples = self._preprocess_motion_batch(inputs, data_samples, k)
            inputs, data_samples = self._preprocess_audio_batch(inputs, data_samples, k)
            inputs, data_samples = self._preprocess_other_batch(inputs, data_samples, k)

            batch_inputs[k] = inputs

        return batch_inputs, data_samples

    def do_norm(self,
                inputs: Union[Tensor, List[Tensor]],
                data_samples: Union[SampleList, DataSample] = None,
                key=None) -> Tuple[Tensor, Union[SampleList, DataSample]]:

        inputs, mean, std = self.normalizer.normalize(inputs)

        key = f'{key}_' if key is not None else ''
        if data_samples is not None:
            data_process_meta = {
                f'{key}mean': mean,
                f'{key}std': std
            }
            if isinstance(data_samples, list):
                for data_sample in data_samples:
                    data_sample.set_metainfo(data_process_meta)
            else:
                data_samples.set_metainfo(data_process_meta)

        return inputs, data_samples

    def _preprocess_data_sample(self, data_samples: SampleList) -> DataSample:
        """
        Args:
            data_samples (List[DataSample]): A list of data samples to
                preprocess.

        Returns:
            list: The list of processed data samples.
        """

        for data_sample in data_samples:
            for key in self._MOTION_KEYS:
                data = data_sample.get(key)
                if data is None:
                    continue
                data, _ = self.do_norm(data, data_sample, key)

                data_sample.set_data({f'{key}': data})

        if self.stack_data_sample:
            assert is_seq_of(data_samples, DataSample), (
                'Only support \'stack_data_sample\' for DataSample '
                'object. Please refer to \'DataSample.stack\'.')
            return DataSample.stack(data_samples)
        return data_samples

    def vec2joints(self, data: torch.Tensor, data_samples: DataSample):
        if isinstance(data, list):
            return [self.vec2joints_fn(d, data_samples) for d in data]
        return self.vec2joints_fn(data, data_samples)

    def vec2rotation(self, data: torch.Tensor, data_samples: DataSample):
        if isinstance(data, list):
            return [self.vec2rotation_fn(d, data_samples) for d in data]
        return self.vec2rotation_fn(data, data_samples)

    @staticmethod
    def merge_interaction(data_sample: DataSample):
        """ For interaction tasks, merge the persons together for further visualization
        :param data_sample: including predicted motion and condition motion, with keys end in 'joints'
        :return: updated data sample
        """

        pred_b_joints = data_sample.get('pred_interactor_joints', None)
        if pred_b_joints is not None:
            pred_a_joints = data_sample.get('pred_joints',
                                            data_sample.get('joints', None))
            assert pred_a_joints is not None, data_sample
            motion_length = min(len(pred_b_joints), len(pred_a_joints))
            pred_joints = torch.stack([pred_a_joints[:motion_length],
                                       pred_b_joints[:motion_length]], dim=1)
            data_sample.set_field(pred_joints, 'pred_joints')

        # gt
        b_joints = data_sample.get('interactor_joints', None)
        if b_joints is not None:
            a_joints = data_sample.get('joints', None)
            assert a_joints is not None, data_sample
            motion_length = min(len(b_joints), len(a_joints))
            joints = torch.stack([a_joints[:motion_length], b_joints[:motion_length]], dim=1)
            data_sample.set_field(joints, 'joints')

        return data_sample

    def postprocess_data_sample(self, data_samples: DataSample):
        """ Do following postprocessing instructions:
        1) Merge the predicted
        :param data_samples: The data samples after prediction
        data_samples should include:
            pred_motion
        :return:
        """
        # calculate joints positions from motion vectors.
        new_data_samples = []
        data_samples = data_samples.split(allow_nonseq_value=True)
        for data_sample in data_samples:
            for key, value in data_sample.to_dict().items():
                if key.endswith('motion') and value is not None:
                    value = self.destruct(value, data_sample, key)
                    data_sample.set_field(value, key)
                    joints_key = key.replace('motion', 'joints')
                    joints = self.vec2joints(value, data_sample)
                    data_sample.set_field(joints, joints_key)

            data_sample = self.merge_interaction(data_sample)
            new_data_samples.append(data_sample)
        return new_data_samples


def is_dict_or_dict_list(input_data):
    if isinstance(input_data, dict):
        return True
    elif isinstance(input_data, list):
        return all(isinstance(item, dict) for item in input_data)
    return False
