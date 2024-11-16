import numpy as np
from logging import WARNING
from typing import Union, Dict, List, Optional, Tuple
import torch
from einops import rearrange, repeat
from mmengine import is_seq_of, print_log
from mmengine.model import BaseDataPreprocessor
from mmengine.model.base_model.data_preprocessor import CastData
from smplx import SMPLX
from smplx.utils import SMPLXOutput
from torch import Tensor

from mmotion.core.conventions.keypoints_mapping import keypoint_index_mapping
from mmotion.registry import MODELS
from mmotion.structures import DataSample
from mmotion.utils.geometry.rotation_convert import rot_dim, rot_convert
from mmotion.utils.smpl_utils.smpl_key_const import TRANSL, GLOBAL_ORIENT, BODY_POSE, LEFT_HAND_POSE, \
    RIGHT_HAND_POSE, ROTATION_KEYS, NUM_JOINTS_OF_ROT, SMPLX_KEYS, param_dim
from mmotion.utils.smpl_utils.tensor_dict_transform import tensor_to_smplx_dict
from mmotion.utils.typing import SampleList


@MODELS.register_module()
class SmplDataPreprocessor(BaseDataPreprocessor):
    _SMPL_KEYS = None
    _JOINT_KEYS = None
    _NON_CONCATENATE_KEYS = ['fps']

    def __init__(self,
                 norm: Dict = dict(
                     transl=dict(
                         mean=0.,
                         std=2
                     ),
                     rotation=dict(
                         mean=0.,
                         std=torch.pi
                     )
                 ),
                 enable_norm: bool = True,
                 data_source: str = 'smplh',
                 to_tensor_keys: List[str] = [TRANSL, GLOBAL_ORIENT, BODY_POSE, LEFT_HAND_POSE, RIGHT_HAND_POSE],
                 smpl_keys: Optional[Tuple[str, List[str]]] = ['motion'],
                 joint_keys: Optional[Tuple[str, List[str]]] = ['joints'],
                 non_concatenate_keys: Optional[Tuple[str, List[str]]] = [],
                 data_keys: Union[List[str], str] = ['recons_motion', 'gt_motion'],
                 stack_data_sample=True,
                 smplx_model: str = 'smpl_models/smplx/SMPLX_NEUTRAL.npz',
                 ):
        super().__init__()
        self.norm_dict = self.load_norm(norm)
        self._enable_normalize = enable_norm
        self.to_tensor_keys = to_tensor_keys
        self.data_source = data_source
        if smpl_keys is not None:
            if not isinstance(smpl_keys, list):
                smpl_keys = [smpl_keys]
            self._SMPL_KEYS = smpl_keys

        if joint_keys is not None:
            if not isinstance(joint_keys, list):
                joint_keys = [joint_keys]
            self._JOINT_KEYS = joint_keys

        if non_concatenate_keys is not None:
            if not isinstance(non_concatenate_keys, list):
                non_concatenate_keys = [non_concatenate_keys]
            self._NON_CONCATENATE_KEYS += non_concatenate_keys

        self.smplx_model = SMPLX(model_path=smplx_model, ext=smplx_model[-3:],
                                 create_transl=False, create_betas=False, create_expression=False,
                                 create_jaw_pose=False, create_body_pose=False, create_leye_pose=False,
                                 create_reye_pose=False, create_left_hand_pose=False, create_right_hand_pose=False,
                                 create_global_orient=False, use_face_contour=False, use_pca=False,
                                 use_compressed=False).eval()

        self.data_keys = data_keys
        if data_keys is not None and not isinstance(data_keys, list):
            self.data_keys = [data_keys]

        self.stack_data_sample = stack_data_sample

    def load_norm(self, norm_dict: Union[str, Dict]):
        if isinstance(norm_dict, str):
            norm_dict = np.load(norm_dict, allow_pickle=True)
        for key, value in norm_dict.items():
            value['mean'] = torch.tensor(value['mean'])
            value['std'] = torch.tensor(value['std'])
            norm_dict[key] = value
        return norm_dict

    @torch.no_grad()
    def forward_kinematics(self, smplx_dict: Dict, chunk_size=6400) -> torch.Tensor:
        """
        :param smplx_dict: smplx dict
        :param chunk_size: to prevent oom while fk,
         only chunk size frames are taken into fk for each time.
        :return: joint coordinates in shape [T, C]
        """
        joints_list = []
        num_frames = len(smplx_dict['transl'])
        for chunk_idx in range(0, num_frames, chunk_size):
            chunk_dict = {key: value[chunk_idx:chunk_idx + chunk_size] for key, value in smplx_dict.items()}
            chunk_output: SMPLXOutput = self.smplx_model.forward(**chunk_dict)
            joints_list.append(chunk_output.joints.data)
        joints = torch.cat(joints_list, dim=0)
        return joints

    def batch_tensor_to_joints(self,
                               outputs: Tensor,
                               data_samples: Union[SampleList, DataSample, None] = None,
                               chunk_size: int = 6400,
                               num_joints=52
                               ) -> torch.Tensor:
        """ Turn smplx tensor(for training) to joint position tensor
        :param outputs: smplx tensor to destruct. [B T C]
        :param data_samples: data samples
        :return: body joint position Tensor
        """
        rot_type = data_samples.metainfo.get('rot_type')[0]
        global2local = data_samples.metainfo.get('global2local')[0]
        init_loc = data_samples.get('init_loc', None)
        b, t, c = outputs.shape
        if global2local is True and init_loc is not None:
            outputs[..., 1:, :3] = torch.cumsum(outputs[..., 1:, :3], dim=1)
            outputs[..., 0, :3] = init_loc
        outputs = rearrange(outputs, 'b t c -> (b t) c')
        batch_smplx_dict = tensor_to_smplx_dict(outputs,
                                                rot_type=rot_type,
                                                global2local=global2local,
                                                init_loc=init_loc)
        batch_joints = self.forward_kinematics(batch_smplx_dict, chunk_size=chunk_size)
        batch_joints = rearrange(batch_joints, '(b t) j c -> b t j c', b=b)
        batch_joints = torch.cat(
            [
                batch_joints[:, :, :22],
                batch_joints[:, :, 25:num_joints + 3]
            ],
            dim=-2
        )
        return batch_joints

    def destruct(self,
                 outputs: Tensor,
                 data_samples: Union[SampleList, DataSample, None] = None,
                 key: str = 'recons_motion') -> Union[list, Tensor]:
        """ Destruct smpl tensor to smplx dict.
        :param outputs: smpl tensor to destruct
        :param data_samples: data samples
        :param key: key to destruct
        :return: destructed smplx
        """
        output_smpl_dict = self.tensor_to_smpl(outputs, data_samples)
        output_smpl_dict = self._destruct_norm(output_smpl_dict)
        output_smpl_dict = self._destruct_rotation(output_smpl_dict, data_samples)
        output_smpl_dict = self._destruct_to_list(output_smpl_dict)
        return output_smpl_dict

    def tensor_unormalize(self, smpl_tensor: torch.Tensor,
                          data_samples: Union[SampleList, DataSample, None] = None):
        smpl_dict = self.tensor_to_smpl(smpl_tensor, data_samples)
        smpl_dict = self._destruct_norm(smpl_dict)
        smpl_tensor = self.smpl_to_tensor(smpl_dict)
        return smpl_tensor

    def _destruct_norm(self, smpl_dict, data_samples: DataSample = None):
        """
        :param batch_smplx_dict:
        :param data_samples:
        :return:
        """
        if not hasattr(self, 'norm_dict') or self.norm_dict is None or not self._enable_normalize:
            return smpl_dict
        for key, value in smpl_dict.items():
            if key in self.norm_dict.keys():
                mean = self.norm_dict[key]['mean']
                std = self.norm_dict[key]['std']
            elif key in ROTATION_KEYS and 'rotation' in self.norm_dict.keys():
                mean = self.norm_dict['rotation']['mean']
                std = self.norm_dict['rotation']['std']
            else:
                continue
            mean = mean.to(value)
            std = std.to(value)
            if key in ROTATION_KEYS and key != GLOBAL_ORIENT:
                if len(mean.shape):
                    mean = repeat(mean, 'c -> (j c)', j=value.shape[-1] // mean.shape[-1])
                    std = repeat(std, 'c -> (j c)', j=value.shape[-1] // std.shape[-1])
            smpl_dict[key] = value * std + mean
        return smpl_dict

    def _destruct_rotation(self, smpl_dict, data_samples: DataSample):
        rot_type = data_samples.rot_type
        if isinstance(rot_type, List):
            rot_type = rot_type[0]
        for key, value in smpl_dict.items():
            if key in ROTATION_KEYS:
                j = NUM_JOINTS_OF_ROT[key]
                if j > 1:
                    value = rearrange(value, '... (j c) -> ... j c', j=j)
                    value = rot_convert(value, rot_type, 'axis_angle')
                    value = rearrange(value, '... j c -> ... (j c)')
                else:
                    value = rot_convert(value, rot_type, 'axis_angle')
                smpl_dict[key] = value
        return smpl_dict

    def _destruct_to_list(self, smpl_dict, data_samples: DataSample = None) -> List[Dict]:
        """ Separate smpl dict into a list of smpl dicts, which param is in shape (t, c)
        :param smpl_dict: a smpl batch dict, each param is (b, t, c)
        :param data_samples:
        :return:
        """
        batch_size = smpl_dict[TRANSL].shape[0]
        smpl_list = [{key: value[i] for key, value in smpl_dict.items()} for i in range(batch_size)]
        return smpl_list

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
        # process input
        assert isinstance(_batch_inputs, dict), f'Only dicts input are supported yet, but got {type(_batch_inputs)}'
        _batch_inputs, _batch_data_samples = \
            self._preprocess_dict_inputs(
                _batch_inputs, _batch_data_samples)

        data['inputs'] = _batch_inputs
        # process data samples
        if _batch_data_samples:
            _batch_data_samples = self._preprocess_data_sample(
                _batch_data_samples, training)

        data['data_samples'] = _batch_data_samples
        return data

    def _preprocess_smpl_dict(self,
                              smpl_dict: Dict,
                              data_samples: Optional[SampleList]) -> Tuple[Tensor, SampleList]:
        """
        :param smpl_dict: a smpl dict, each param in the dict is a batch list, for exp:
        transl: b * [t 3], global_orient: b * [t 3], body_pose: b * [t 63] ...
        :param data_samples:
        :return:
        """
        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(len(smpl_dict[TRANSL]))]

        for key, value in smpl_dict.items():
            if isinstance(value, List):
                smpl_dict[key] = torch.stack(value, dim=0)

        smpl_dict = self._do_norm(smpl_dict)
        smpl_tensor = self.smpl_to_tensor(smpl_dict)

        return smpl_tensor, data_samples

    def _preprocess_smpl_list(self,
                              smpl_list: List[Dict],
                              data_samples: Optional[SampleList]) -> Tuple[Tensor, SampleList]:
        """Preprocess a list of motion tensor

        Args:
            smpl_list (List[Dict]): A batch of smpl dict
            data_samples (List[DataSample], optional): The data samples
                of corresponding inputs. If not passed, a list of empty data
                samples will be initialized to save metainfo. Defaults to None.
            key (str): The key of tensor list in data samples.
                Defaults to 'img'.

        Returns:
            Tuple[Tensor, List[DataSample]]: The preprocessed motion tensor
                and updated data samples.
        """
        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(len(smpl_list))]

        batch_smpl_dict = {
            key: torch.stack([smpl_dict[key] for smpl_dict in smpl_list], dim=0)
            for key in smpl_list[0].keys()
        }

        batch_smpl_dict = self._do_norm(batch_smpl_dict)
        smpl_tensor = self.smpl_to_tensor(batch_smpl_dict)
        return smpl_tensor, data_samples

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
        pad_size_dict = dict()
        for k, inputs in batch_inputs.items():
            # handle concentrate for values in list
            if isinstance(inputs, list):
                if k in self._NON_CONCATENATE_KEYS:
                    # use the first value
                    assert all([
                        inputs[0] == inp for inp in inputs
                    ]), (f'NON_CONCENTATE_KEY \'{k}\' should be consistency '
                         'among the data list.')
                    batch_inputs[k] = inputs[0]
                else:

                    if k in self._SMPL_KEYS:
                        if isinstance(inputs, List):
                            inputs, data_samples = self._preprocess_smpl_list(
                                inputs, data_samples)

                    if k in self._JOINT_KEYS:
                        if isinstance(inputs, List):
                            inputs, data_samples = self._preprocess_joints_list(
                                inputs, data_samples)

                    else:
                        # only stack
                        inputs = torch.stack(inputs)

                    batch_inputs[k] = inputs
            elif isinstance(inputs, dict):
                if k in self._SMPL_KEYS:
                    inputs, data_samples = self._preprocess_smpl_dict(
                        inputs, data_samples)
                batch_inputs[k] = inputs

        return batch_inputs, data_samples

    def _preprocess_joints_list(self, joints: List[Tensor], data_samples: SampleList):
        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(len(joints))]

        assert isinstance(joints, List), f'a batch of joints must be a list of tensors'
        joints = torch.stack(joints, dim=0)
        index = keypoint_index_mapping(self.data_source, 'smplx')
        joints = joints[..., index, :]
        return joints, data_samples

    def _preprocess_data_sample(self, data_samples: SampleList,
                                training: bool) -> DataSample:
        """Preprocess data samples. When `training` is True, fields belong to
        :attr:`self.data_keys` will be converted to
        :attr:`self.output_channel_order` and then normalized by `self.mean`
        and `self.std`. When `training` is False, fields belongs to
        :attr:`self.data_keys` will be attempted to convert to 'BGR' without
        normalization. The corresponding metainfo related to normalization,
        channel order conversion will be updated to data sample as well.

        Args:
            data_samples (List[DataSample]): A list of data samples to
                preprocess.
            training (bool): Whether in training mode.

        Returns:
            list: The list of processed data samples.
        """

        for data_sample in data_samples:
            if not self.data_keys:
                break
            for key in self.data_keys:
                if not hasattr(data_sample, key):
                    # do not raise error here
                    print_log(f'Cannot find key \'{key}\' in data sample.',
                              'current', WARNING)
                    continue

                data = data_sample.get(key)
                if key in self._SMPL_KEYS:
                    data = self._do_norm(data)
                    data_process_meta = {
                        f'{key}_enable_norm': self._enable_normalize,
                        f'{key}_norm_dict': self.norm_dict
                    }
                    data_sample.set_metainfo(data_process_meta)
                data_sample.set_data({f'{key}': data})

        if self.stack_data_sample:
            assert is_seq_of(data_samples, DataSample), (
                'Only support \'stack_data_sample\' for DataSample '
                'object. Please refer to \'DataSample.stack\'.')
            return DataSample.stack(data_samples)
        return data_samples

    def _do_norm(self, smpl_dict: Dict) -> Dict:

        if not hasattr(self, 'norm_dict') or self.norm_dict is None or not self._enable_normalize:
            return smpl_dict
        for key, value in smpl_dict.items():
            if key in self.norm_dict.keys():
                mean = self.norm_dict[key]['mean']
                std = self.norm_dict[key]['std']
            elif key in ROTATION_KEYS and 'rotation' in self.norm_dict.keys():
                mean = self.norm_dict['rotation']['mean']
                std = self.norm_dict['rotation']['std']
            else:
                continue
            mean = mean.to(value)
            std = std.to(value)
            if key in ROTATION_KEYS and key != GLOBAL_ORIENT:
                if len(mean.shape):
                    mean = repeat(mean, 'c -> (j c)', j=value.shape[-1] // mean.shape[-1])
                    std = repeat(std, 'c -> (j c)', j=value.shape[-1] // std.shape[-1])
            smpl_dict[key] = (value - mean) / std
        return smpl_dict

    def smpl_to_tensor(self, smpl_dict: Dict) -> Tensor:
        smpl_tensor = torch.concatenate([smpl_dict[key] for key in self.to_tensor_keys], dim=-1)
        return smpl_tensor

    def tensor_to_smpl(self, output: Tensor, data_samples: DataSample) -> Dict:
        """
        :param output: B T C
        :param data_samples:
        :return:
        """
        b, t, _ = output.shape
        rot_type = data_samples.rot_type
        if isinstance(rot_type, List):
            rot_type = rot_type[0]
        num_rot_dim = rot_dim[rot_type]
        smpl_dict = {}
        for idx, key in enumerate(self.to_tensor_keys):
            if key == TRANSL:
                smpl_dict[key] = output[..., :3]
                output = output[..., 3:]
            else:
                num_joint = NUM_JOINTS_OF_ROT[key]
                smpl_dict[key] = output[..., :num_rot_dim * num_joint]
                output = output[..., num_rot_dim * num_joint:]
        for key in SMPLX_KEYS:
            if key not in self.to_tensor_keys:
                if key in ROTATION_KEYS:
                    num_joint = NUM_JOINTS_OF_ROT[key]
                    smpl_dict[key] = torch.zeros([b, t, num_rot_dim * num_joint]).to(output)
                else:
                    smpl_dict[key] = torch.zeros([b, t, param_dim[key]]).to(output)
        return smpl_dict
