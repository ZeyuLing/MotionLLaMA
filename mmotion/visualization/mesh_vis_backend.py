import numpy as np
import os
from os.path import join, exists
from typing import Dict, Union

import torch
from mmengine.visualization import LocalVisBackend

from mmotion.core.visualization.visualize_smpl import render_smpl
from mmotion.registry import VISBACKENDS, MODELS
from mmotion.utils.smpl_utils.smpl_key_const import BODY_POSE, merge_pose_keys


@VISBACKENDS.register_module()
class MeshVisBackend(LocalVisBackend):

    def __init__(self,
                 save_dir: str,
                 img_save_dir: str = 'vis_image',
                 config_save_file: str = 'config.py',
                 scalar_save_file: str = 'scalars.json',
                 pretrained_smplx: str = 'smpl_models/smplx',
                 device: str = 'cuda'):
        super().__init__(save_dir, img_save_dir, config_save_file, scalar_save_file)
        self.body_model = MODELS.build(
            dict(model_path=pretrained_smplx, type='smplx', gender='neutral', create_betas=False,
                 create_transl=False, create_expression=False, create_body_pose=False,
                 create_reye_pose=False, create_jaw_pose=False, create_leye_pose=False,
                 create_left_hand_pose=False, create_right_hand_pose=False,
                 create_global_orient=False,
                 use_pca=False, use_face_contour=False, keypoint_src='smplx_no_contour'
                 )).to(device)
        self.device = device

    def add_image(self,
                  name: str,
                  smplx_dict: Union[torch.Tensor, Dict[str, torch.Tensor]],
                  key: str = 'gt_motion',
                  step: int = 0,
                  start_frame=0,
                  fps: int = 20,
                  resolution=(512, 512),
                  **kwargs) -> str:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            step (int): Global step value to record. Defaults to 0.
            :param smplx_dict: smplx dict or tensor of smplx dict
        """
        if BODY_POSE in smplx_dict.keys():
            smplx_dict = merge_pose_keys(smplx_dict)
        for k, value in smplx_dict.items():
            if isinstance(value, torch.Tensor):
                smplx_dict[k] = value.to(torch.float32).to(self.device)

        save_dir = join(self._img_save_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        save_file_name = f'{name}_{key}_{step}_{start_frame}.mp4'

        save_path = join(save_dir, save_file_name)
        if not exists(save_path):
            render_smpl(
                **smplx_dict,
                output_path=save_path,
                body_model=self.body_model,
                resolution=(1024, 1024),
                R=np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ]),
                overwrite=True
            )
        return save_path
