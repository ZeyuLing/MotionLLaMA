import numpy as np
import os
import sys
from typing import Dict, List

import random

import torch

import torch.nn.functional as F
from tqdm import tqdm

from mmotion.models.generators.mdm.mdm import MDM
from mmotion.utils.task.task_lib import MotionPrediction, MotionInbetween
from mmotion.utils.typing import SampleList

sys.path.append(os.curdir)
from mmotion.registry import MODELS
from mmotion.structures import DataSample


def create_completion_mask(motion: torch.Tensor,
                           num_frames: List[int],
                           task, past_ratio: float, future_ratio: float):
    """ For prediction and inbetween tasks, create a mask for the area need to predict,
    1 for need to predict, 0 for given condition and padding
    :param motion: gt motion
    :param num_frames: valid frame nums
    :param task: MotionPrediction or MotionInbetween
    :param past_ratio: 0.4 for prediction and 0.2 for inbetween as default
    :param future_ratio: None for prediction and 0.2 for inbetween as default
    :return:
    """
    B, T, _ = motion.shape
    range_tensor = torch.arange(T).view(1, T).to(motion.device)
    if task == MotionPrediction:
        infer_begin = torch.tensor([int(nf * past_ratio) for nf in num_frames], device=motion.device).unsqueeze(-1)
        infer_end = torch.tensor(num_frames, device=motion.device).unsqueeze(-1)
    else:
        assert task == MotionInbetween, 'task should be either prediction or inbetween'
        infer_begin = torch.tensor([int(nf * past_ratio) for nf in num_frames], device=motion.device).unsqueeze(-1)
        infer_end = torch.tensor([int(nf * (1 - future_ratio)) for nf in num_frames], device=motion.device).unsqueeze(
            -1)
    mask = (((range_tensor >= infer_begin) & (range_tensor < infer_end))
            .to(device=motion.device, dtype=torch.bool))
    return mask


@MODELS.register_module()
class CompletionMDM(MDM):

    @torch.no_grad()
    def infer(self,
              ori_motion: torch.Tensor,
              num_frames: List[int],
              task=MotionPrediction,
              past_ratio=0.2,
              future_ratio=0.2,
              caption: List[str] = None,
              num_inference_steps: int = 50,
              guidance_scale: float = 1.,
              show_progress=True,
              generator=None,
              eta=0.
              ):
        if isinstance(num_frames, int):
            num_frames = [num_frames]

        if isinstance(caption, str):
            caption = [caption]
        elif caption is None:
            caption = [''] * len(num_frames)

        batch_size = len(caption)

        cond = self.encode_text(caption)
        do_classifier_free_guidance = guidance_scale > 1.0
        assert not do_classifier_free_guidance, 'not implemented for mdm'

        self.test_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.test_scheduler.timesteps

        x_t = self.prepare_noise(
            batch_size,
            self.nfeats,
            num_frames,
            torch.bfloat16,
            'cuda'
        )

        completion_mask = create_completion_mask(ori_motion, num_frames, task, past_ratio, future_ratio).unsqueeze(-1)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if show_progress:
            timesteps = tqdm(timesteps)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            x_t = self.test_scheduler.scale_model_input(x_t, t)

            # predict the noise residual
            model_output = self.forward_trans(x_t, t, cond, num_frames)
            model_output = ori_motion * ~completion_mask + model_output * completion_mask
            step_result = self.test_scheduler.step(
                model_output, t, x_t, **extra_step_kwargs)
            x_t = step_result['prev_sample']
        return x_t

    def forward_loss(self, inputs: Dict, data_samples: DataSample):
        raise NotImplementedError('Completion is training free')

    @torch.no_grad()
    def forward_predict(self, inputs, data_samples: DataSample) -> SampleList:
        task = data_samples.get('task')[0]
        motion = inputs['motion']
        caption = data_samples.get('caption')

        if task == MotionPrediction:
            past_ratio = data_samples.get('past_ratio', [0.4])[0]
        else:
            past_ratio = data_samples.get('past_ratio', [0.2])[0]
        future_ratio = data_samples.get('future_ratio', [0.2])[0]

        num_frames = data_samples.get('num_frames')
        # assert num_frames to be generated are the same.
        pred_entire_motion = self.infer(motion, num_frames, task, past_ratio, future_ratio, caption)
        data_samples.set_field(pred_entire_motion, 'pred_motion')
        data_samples.set_data(inputs)

        data_samples = self.data_preprocessor.postprocess_data_sample(data_samples)
        return data_samples
