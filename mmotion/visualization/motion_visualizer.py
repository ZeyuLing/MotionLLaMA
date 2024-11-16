import logging
import re
from typing import Sequence

import torch
from mmengine.visualization import Visualizer
from mmotion.registry import VISUALIZERS
from mmotion.structures import DataSample
from mmotion.utils.logger import print_colored_log
from mmotion.utils.smpl_utils.tensor_dict_transform import tensor_to_smplx_dict


@VISUALIZERS.register_module()
class MotionVisualizer(Visualizer):
    """
        Visualizer for Motion VQ-VAE evaluation
    """

    def __init__(self,
                 fn_key: str = 'smplx_path',
                 motion_keys: Sequence[str] = ['gt_joints', 'recons_joints'],
                 name: str = 'visualizer',
                 start_frame_key='start_frame',
                 is_smpl: bool = False,
                 *args,
                 **kwargs):
        super().__init__(name, *args, **kwargs)
        self.fn_key = fn_key
        if not isinstance(self.fn_key, list):
            self.fn_key = [self.fn_key]

        self.motion_keys = motion_keys
        self.start_frame_key = start_frame_key
        self.is_smpl = is_smpl

    def add_datasample(self, data_sample: DataSample, step=0) -> None:
        """
        :param data_sample: DataSample to
        :param step:
        :return:
        """
        merged_dict = {
            **data_sample.to_dict()
        }
        if 'output' in merged_dict.keys():
            merged_dict.update(**merged_dict['output'])

        for key in self.fn_key:
            if key in merged_dict.keys():
                self.fn_key = key
                break

        fn = merged_dict[self.fn_key]

        if isinstance(fn, list):
            fn = fn[0]
        fn = re.split(r' |/|\\', fn)[-1]
        fn = fn.split('.')[0]

        for k in self.motion_keys:
            start_frame = merged_dict.get(self.start_frame_key, 0)
            if k not in merged_dict:
                print_colored_log(
                    f'Key "{k}" not in data_sample or outputs',
                    level=logging.WARN)
                continue
            motion = merged_dict[k]
            if self.is_smpl and isinstance(motion, torch.Tensor):
                rot_type = merged_dict['rot_type']
                motion = tensor_to_smplx_dict(motion, rot_type=rot_type)
            for vis_backend in self._vis_backends.values():
                vis_backend.add_image(fn, motion, k, step, start_frame=start_frame)  # type: ignore