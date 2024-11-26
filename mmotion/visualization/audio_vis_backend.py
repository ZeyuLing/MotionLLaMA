import numpy as np
from os.path import join

import os
from typing import Union, Optional

import torch
import torchaudio
from mmengine.visualization import LocalVisBackend

from mmotion.registry import VISBACKENDS


@VISBACKENDS.register_module()
class AudioVisBackend(LocalVisBackend):
    def add_image(self,
                  name: str,
                  audio: Union[torch.Tensor, np.ndarray],
                  sr: int = 24000,
                  key: str = 'pred_audio',
                  step: int = None,
                  start_frame: Optional[int] = None,
                  **kwargs) -> str:
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio)
        audio = audio.float()
        audio = audio.detach().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        save_dir = join(self._img_save_dir, name, f'{name}')
        if step is not None:
            save_dir = save_dir + f'_{step}'
        if start_frame is not None:
            save_dir = save_dir + f'_{start_frame}'

        os.makedirs(save_dir, exist_ok=True)

        save_file_name = f'{key}.wav'
        save_path = join(save_dir, save_file_name)
        torchaudio.save(save_path, audio, sr)
        return save_path
