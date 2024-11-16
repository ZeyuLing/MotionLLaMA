import numpy as np
import os
from os.path import join, exists
from typing import Dict, Union, Optional

import torch
from mmengine.visualization import LocalVisBackend

from mmotion.core.visualization import visualize_kp3d
from mmotion.registry import VISBACKENDS
from mmotion.utils.bvh.joints2bvh_converter import Joint2BVHConvertor


@VISBACKENDS.register_module()
class JointsVisBackend(LocalVisBackend):

    def __init__(self,
                 save_dir: str,
                 img_save_dir: str = 'vis_image',
                 config_save_file: str = 'config.py',
                 scalar_save_file: str = 'scalars.json',
                 data_source='smplh',
                 bvh_template='data/motionhub/template.bvh'):
        super().__init__(save_dir, img_save_dir, config_save_file, scalar_save_file)
        self.data_source = data_source
        self.bvh_convertor = Joint2BVHConvertor(bvh_template, data_source)

    def add_image(self,
                  name: str,
                  motion: Union[torch.Tensor, Dict[str, torch.Tensor]],
                  key: str = 'gt_joints',
                  step: Optional[int] = None,
                  start_frame: Optional[int] = None,
                  fps: int = 20,
                  resolution=(512, 512),
                  text=None,
                  **kwargs) -> str:
        """
        :param name: directory to save the video
        :param motion: the motion tensor or array
        :param key: filename of video
        :param step: current training step
        :param start_frame: where does the motion start in the original motion file.
        :param fps: fps
        :param resolution: resolution of the rendered video
        :param text: text needs to be shown in the video.
        :param kwargs:
        :return:
        """
        if isinstance(motion, torch.Tensor):
            motion = motion.detach().cpu().float().numpy()

        save_dir = join(self._img_save_dir, name, f'{name}')
        if step is not None:
            save_dir = save_dir + f'_{step}'
        if start_frame is not None:
            save_dir = save_dir + f'_{start_frame}'

        os.makedirs(save_dir, exist_ok=True)

        video_save_path = join(save_dir, f'{key}.mp4')
        np_save_path = join(save_dir, f'{key}.npy')
        bvh_save_path = join(save_dir, f'{key}.bvh')
        if not exists(video_save_path):
            visualize_kp3d(
                motion,
                frame_names=text,
                convention='blender',
                output_path=video_save_path,
                resolution=(1024, 1024),
                data_source=self.data_source,
            )
        if not exists(np_save_path):
            np.save(np_save_path, motion)
        if not exists(bvh_save_path):
            if len(motion.shape) == 3:
                self.bvh_convertor.convert(motion, bvh_save_path, fps=fps)
            else:
                assert len(motion.shape) == 4
                p1, p2 = motion[:, 0], motion[:, 1]
                p1_path = join(save_dir, f'{key}_p1.bvh')
                p2_path = join(save_dir, f'{key}_p2.bvh')
                self.bvh_convertor.convert(p1, p1_path, fps=fps)
                self.bvh_convertor.convert(p2, p2_path, fps=fps)

        return video_save_path
