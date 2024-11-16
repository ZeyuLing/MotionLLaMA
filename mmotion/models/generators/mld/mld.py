from typing import Dict, List

import os

import random

import torch
import torch.nn.functional as F
from mmengine import Config
from torch.nn.modules.module import T
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torch import nn

from mmotion.models.generators.gesture_vae import GestureVAE
from mmotion.models.generators.mld.mld_vae import MldVAE
from mmotion.models.generators.base_diffusion_model import BaseDiffusionModel
from mmotion.models.generators.mld.mld_denoiser import MLDDenoiser
from mmotion.registry import MODELS
from mmotion.structures import DataSample


@MODELS.register_module()
class MLD(BaseDiffusionModel):
    def train(self: T, mode: bool = True) -> T:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.vae.eval()
        self.vae.requires_grad_(False)

    def __init__(self, vae_cfg: Dict,
                 denoiser_cfg: Dict,
                 scheduler: Dict,
                 clip_dim=512,
                 latent_dim=256,
                 cond_mask_prob=.1,
                 clip_path='checkpoints/vit_base_patch32',
                 test_scheduler=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(scheduler, test_scheduler, data_preprocessor, init_cfg)
        self.vae: GestureVAE = self.build_vae(vae_cfg)
        self.latent_dim = latent_dim
        self.embed_text = nn.Linear(clip_dim, latent_dim)
        self.denoiser: MLDDenoiser = MODELS.build(denoiser_cfg)
        self.load_and_freeze_clip(clip_path)
        self.cond_mask_prob = cond_mask_prob

    def build_vae(self, vae_cfg: Dict):
        """
        :param mm_tokenizer_cfg: vqvae config
        :return: Vqvae module.
        """
        type = vae_cfg['type']
        if os.path.isfile(type):
            # allow using config file path as type, simplify config writing.
            init_cfg = vae_cfg.pop('init_cfg', None)
            vae_cfg = Config.fromfile(type)['model']
            if init_cfg is not None:
                vae_cfg['init_cfg'] = init_cfg

        vae: GestureVAE = MODELS.build(vae_cfg).eval()
        if vae_cfg.get('init_cfg', None) is not None:
            vae.init_weights()
        vae.requires_grad_(False)
        return vae.eval()

    def random_mask_cond(self, cond):
        batch_size = len(cond)
        for i in range(batch_size):
            if random.random() < self.cond_mask_prob:
                cond[i] = ''
        return cond

    def encode_text(self, raw_text):
        inputs = self.clip_processor(raw_text, return_tensors="pt", padding=True,
                                     truncation=True, max_length=77).to('cuda')
        texts = self.clip_model.get_text_features(**inputs)
        texts = self.embed_text(texts)
        texts = texts.unsqueeze(1)
        return texts

    def load_and_freeze_clip(self, clip_path):
        self.clip_model = CLIPModel.from_pretrained(clip_path)  # Must set jit=False for training
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        # Freeze CLIP weights
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

    def forward_loss(self, inputs: Dict, data_samples: DataSample):
        motion = inputs['motion']
        lengths = data_samples.get('num_frames')

        motion_dist, motion_latent = self.vae.encode_motion(motion, lengths)
        motion_latent = motion_latent.unsqueeze(1)

        caption = data_samples.get('caption')
        caption = self.random_mask_cond(caption)
        cond = self.encode_text(caption)
        batch_size = len(motion)

        noise = self.sample_noise(motion_latent.shape).to(motion_latent)
        timesteps = torch.randint(
            low=0,
            high=self.scheduler.num_train_timesteps, size=(batch_size,),
            device=noise.device).long()
        noisy_latent = self.scheduler.add_noise(motion_latent, noise, timesteps)
        if self.scheduler.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(motion_latent, noise, timesteps)
        elif self.scheduler.prediction_type == 'sample':
            gt = motion_latent
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.prediction_type}')

        model_output: torch.Tensor = self.denoiser(
            noisy_latent,
            timesteps,
            cond)
        loss_dict = dict()
        loss_mse = F.mse_loss(model_output, gt, reduction='mean')
        loss_dict['loss_mse'] = loss_mse
        return loss_dict

    def infer(self,
              caption: List[str],
              num_frames: List[int],
              num_inference_steps: int = 50,
              guidance_scale: float = 7.5,
              show_progress=True,
              generator=None,
              eta=0.
              ):
        if isinstance(caption, str):
            caption = [caption]
        if isinstance(num_frames, int):
            num_frames = [num_frames]

        batch_size = len(caption)

        cond = self.encode_text(caption)
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            uncond = self.encode_text([''] * batch_size)
        self.test_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.test_scheduler.timesteps

        x_t = self.prepare_noise(
            batch_size,
            self.latent_dim,
            1,
            torch.bfloat16,
            'cuda'
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if show_progress:
            timesteps = tqdm(timesteps)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            x_t = self.test_scheduler.scale_model_input(x_t, t)
            t_tensor = torch.tensor([t] * batch_size).to(x_t.device)
            # predict the noise residual
            model_output = self.denoiser(x_t, t_tensor, cond)
            if do_classifier_free_guidance:
                uncond_model_output = self.denoiser(x_t, t_tensor, uncond)
                model_output = (uncond_model_output +
                                guidance_scale * (model_output - uncond_model_output))
            step_result = self.test_scheduler.step(
                model_output, t, x_t, **extra_step_kwargs)
            x_t = step_result['prev_sample']
        # vae decoding
        x_t = x_t.squeeze(1)
        pred_motion = self.vae.motiondecoder(x_t, num_frames)
        return pred_motion

    def forward_predict(self, inputs, data_samples):
        caption = data_samples.get('caption')
        num_frames = data_samples.get('num_frames')
        # assert num_frames to be generated are the same.
        pred_motion = self.infer(caption, num_frames)
        data_samples.set_field(pred_motion, 'pred_motion')
        data_samples.set_field(num_frames, 'pred_motion_num_frames')
        data_samples.set_data(inputs)
        data_samples = self.data_preprocessor.postprocess_data_sample(data_samples)
        return data_samples
