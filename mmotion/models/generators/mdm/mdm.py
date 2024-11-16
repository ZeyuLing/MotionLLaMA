import numpy as np
import os
import sys
from typing import Dict, List

import random

import torch
import torch.nn as nn

import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from mmotion.models.generators.base_diffusion_model import BaseDiffusionModel
from mmotion.utils.typing import SampleList

sys.path.append(os.curdir)
from mmotion.registry import MODELS
from mmotion.structures import DataSample


def create_src_key_padding_mask(lengths):
    batch_size = len(lengths)
    max_len = max(lengths)
    seq_range = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len)
    lengths_tensor = torch.tensor(lengths).unsqueeze(1)
    mask = seq_range >= lengths_tensor
    return mask.to(torch.bool)  # dtype: torch.bool


@MODELS.register_module()
class MDM(BaseDiffusionModel):
    def __init__(self,
                 scheduler: Dict,
                 test_scheduler=None,
                 nfeats=156,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.1,
                 activation="gelu",
                 cond_mask_prob: float = 0.1,
                 clip_dim=512,
                 clip_path='clip-b',
                 data_preprocessor: Dict = None,
                 init_cfg=None
                 ):
        super().__init__(scheduler, test_scheduler, data_preprocessor, init_cfg)

        self.nfeats = nfeats
        self.input_process = InputProcess(nfeats, latent_dim)

        self.cond_mask_prob = cond_mask_prob

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          batch_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)

        self.embed_text = nn.Linear(clip_dim, latent_dim)

        self.output_process = OutputProcess(nfeats, latent_dim)
        self.load_and_freeze_clip(clip_path)

    def load_and_freeze_clip(self, clip_path):
        self.clip_model = CLIPModel.from_pretrained(clip_path)  # Must set jit=False for training
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        # Freeze CLIP weights
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

    def encode_text(self, raw_text):
        inputs = self.clip_processor(raw_text, return_tensors="pt", padding=True,
                                     truncation=True, max_length=77).to('cuda')
        texts = self.clip_model.get_text_features(**inputs)
        texts = self.embed_text(texts)
        return texts

    def random_mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs,
                                                                                                  1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    @torch.no_grad()
    def infer(self,
              caption: List[str],
              num_frames: List[int],
              num_inference_steps: int = 50,
              guidance_scale: float = 1.,
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

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if show_progress:
            timesteps = tqdm(timesteps)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            x_t = self.test_scheduler.scale_model_input(x_t, t)

            # predict the noise residual
            model_output = self.forward_trans(x_t, t, cond, num_frames)
            step_result = self.test_scheduler.step(
                model_output, t, x_t, **extra_step_kwargs)
            x_t = step_result['prev_sample']
        return x_t

    def forward_loss(self, inputs: Dict, data_samples: DataSample):
        motion = inputs['motion']
        # used for making attention mask
        num_frames = data_samples.get('num_frames')
        caption = data_samples.get('caption')

        cond = self.encode_text(caption)
        cond = self.random_mask_cond(cond)

        batch_size = len(motion)
        noise = self.sample_noise(motion.shape).to(motion)
        timesteps = torch.randint(
            low=0,
            high=self.scheduler.num_train_timesteps, size=(batch_size,),
            device=noise.device).long()

        noisy_motion = self.scheduler.add_noise(motion, noise, timesteps)
        if self.scheduler.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(motion, noise, timesteps)
        elif self.scheduler.prediction_type == 'sample':
            gt = motion
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.prediction_type}')

        model_output: torch.Tensor = self.forward_trans(
            noisy_motion,
            timesteps,
            cond,
            num_frames)
        mask = create_src_key_padding_mask(num_frames).unsqueeze(-1).to(gt.device)
        loss_dict = dict()
        model_output = model_output * ~mask
        gt = gt * ~mask
        loss_mse = F.mse_loss(model_output, gt, reduction='mean')
        loss_dict['loss_mse'] = loss_mse
        return loss_dict

    @torch.no_grad()
    def forward_predict(self, inputs, data_samples: DataSample) -> SampleList:
        caption = data_samples.get('caption')
        num_frames = data_samples.get('num_frames')
        # assert num_frames to be generated are the same.
        pred_motion = self.infer(caption, num_frames)
        data_samples.set_field(pred_motion, 'pred_motion')
        data_samples.set_data(inputs)
        # calculate joints positions from motion vectors.

        for key, value in data_samples.to_dict().items():
            if key.endswith('motion') and value is not None:
                value = self.data_preprocessor.destruct(value, data_samples)
                data_samples.set_field(value, key)
                joints_key = key.replace('motion', 'joints')
                joints = self.data_preprocessor.vec2joints(value, data_samples)
                data_samples.set_field(joints, joints_key)
        data_samples = data_samples.split(allow_nonseq_value=True)
        return data_samples

    def forward_trans(self, x, timesteps, cond, num_frames=None):
        """
        :param x: b n d
        :param timesteps: list of timestep
        :param cond: b c
        :param num_frames: list of num_frames
        :return:
        """
        if len(cond.shape) < 3:
            cond = cond.unsqueeze(1)
        mask = None
        if num_frames is not None:
            # take embedding token into account
            mask = create_src_key_padding_mask([n + 1 for n in num_frames]).to(x.device)
        emb = self.embed_timestep(timesteps)  # [bs, 1, d]

        emb = emb + cond

        x = self.input_process(x)
        xseq = torch.cat((emb, x), dim=1)  # [bs, seqlen+1, d]
        xseq = self.sequence_pos_encoder(xseq)  # [bs, seqlen+1, d]
        output = self.seqTransEncoder(xseq, src_key_padding_mask=mask)[:,
                 1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # b t c
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # n 1 c
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        pe = rearrange(self.pe, 'n 1 c -> 1 n c')
        x = x + pe[:, :x.shape[1]]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        embed = self.time_embed(self.sequence_pos_encoder.pe[timesteps])
        return embed


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseEmbedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseFinal = nn.Linear(latent_dim, input_feats)

    def forward(self, output):
        output = self.poseFinal(output)
        return output


if __name__ == '__main__':
    model = MDM(
        scheduler=dict(
            type='EditDDIMScheduler',
            variance_type='learned_range',
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            set_alpha_to_one=False,
            clip_sample=False),
        clip_path='checkpoints/vit_base_patch32/',
    )

    motion = torch.rand([2, 32, 156])
    cond = ['a man is very good', 'a man is very good']
    cond = model.encode_text(cond)
    print(model.forward_trans(motion, timesteps=[322, 999], cond=cond, num_frames=[15, 32]))
