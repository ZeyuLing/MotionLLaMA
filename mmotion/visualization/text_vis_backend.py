import numpy as np
import os
from os.path import join, exists
from typing import Dict, Union, Optional

import torch
from mmengine.visualization import LocalVisBackend

from mmotion.core.visualization import visualize_kp3d
from mmotion.registry import VISBACKENDS
from scripts.motion_calibration import write_txt


@VISBACKENDS.register_module()
class TextVisBackend(LocalVisBackend):

    def __init__(self,
                 save_dir: str,
                 img_save_dir: str = 'vis_image',
                 config_save_file: str = 'config.py',
                 scalar_save_file: str = 'scalars.json'):
        super().__init__(save_dir, img_save_dir, config_save_file, scalar_save_file)

    def add_image(self,
                  name: str,
                  text: str,
                  key: str = 'pred_text',
                  step: int = None,
                  start_frame: Optional[int] = None,
                  **kwargs) -> str:

        save_dir = join(self._img_save_dir, name, f'{name}')
        if step is not None:
            save_dir = save_dir+ f'_{step}'
        if start_frame is not None:
            save_dir = save_dir+f'_{start_frame}'

        os.makedirs(save_dir, exist_ok=True)

        save_file_name = f'{key}.txt'
        save_path = join(save_dir, save_file_name)
        write_txt(save_path, text)
        return save_path
