from typing import Dict, Union, List, Optional

import inspect

from diffusers import DDIMScheduler
from mmengine.model import BaseModel
from torch import Size
import torch

from mmotion.registry import DIFFUSION_SCHEDULERS


class BaseDiffusionModel(BaseModel):
    def __init__(self,
                 scheduler: Dict,
                 test_scheduler=None,
                 data_preprocessor: Dict = None,
                 init_cfg=None
                 ):
        super(BaseDiffusionModel, self).__init__(data_preprocessor, init_cfg)
        self.scheduler: DDIMScheduler = DIFFUSION_SCHEDULERS.build(scheduler)
        test_scheduler = test_scheduler or scheduler
        self.test_scheduler: DDIMScheduler = DIFFUSION_SCHEDULERS.build(test_scheduler)

    def prepare_noise(self,
                      batch_size,
                      channels: int,
                      num_frames: Union[int, List[int]],
                      dtype,
                      device):
        if isinstance(num_frames, list):
            assert len(num_frames) == batch_size, (f'If num frames in the batch differs,'
                                                   f' length of num_frames should equal to batch size,'
                                                   f'but got {num_frames} and batch size {batch_size}')
            num_frames = max(num_frames)
        shape = (batch_size, num_frames, channels)
        noise = torch.randn(shape, device=device, dtype=dtype)
        noise = noise.to(device)
        noise = noise * self.scheduler.init_noise_sigma
        return noise

    def prepare_extra_step_kwargs(self, generator, eta):
        """prepare extra kwargs for the scheduler step.

        Args:
            generator (torch.Generator):
                generator for random functions.
            eta (float):
                eta (η) is only used with the DDIMScheduler,
                it will be ignored for other schedulers.
                eta corresponds to η in DDIM paper:
                https://arxiv.org/abs/2010.02502
                and should be between [0, 1]

        Return:
            extra_step_kwargs (dict):
                dict contains 'generator' and 'eta'
        """
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        if mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'loss':
            return self.forward_loss(inputs, data_samples)
        else:
            raise NotImplementedError(
                'Forward is not implemented now, please use infer.')

    def sample_noise(self, noise_shape: Size):
        noise = torch.randn(size=noise_shape)
        return noise
