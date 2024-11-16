from typing import Dict, List

import math
import random
import torch
from diffusers.models.controlnet import zero_module
from torch import nn
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from mmotion.models.generators.base_diffusion_model import BaseDiffusionModel
from mmotion.models.generators.motion_diffuse.attention import LinearTemporalDiffusionTransformerDecoderLayer
from mmotion.registry import MODELS
from mmotion.structures import DataSample


def create_src_key_padding_mask(lengths):
    batch_size = len(lengths)
    max_len = max(lengths)
    seq_range = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len)
    lengths_tensor = torch.tensor(lengths).unsqueeze(1)
    mask = seq_range >= lengths_tensor
    return mask.to(torch.bool)  # dtype: torch.bool


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if isinstance(timesteps, list):
        timesteps = torch.tensor(timesteps)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


@MODELS.register_module()
class MotionDiffuse(BaseDiffusionModel):
    def __init__(self,
                 scheduler: Dict,
                 test_scheduler=None,
                 input_feats=156,
                 max_num_frames=1000,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu",
                 num_text_layers=4,
                 text_latent_dim=256,
                 text_ff_size=2048,
                 text_num_heads=4,
                 cond_mask_prob: float = 0.,
                 clip_path='clip-b',
                 clip_dim=512,
                 data_preprocessor: Dict = None,
                 init_cfg=None):
        super().__init__(scheduler, test_scheduler, data_preprocessor, init_cfg)
        time_embed_dim = latent_dim * 4
        self.nfeats = input_feats
        self.latent_dim = latent_dim

        self.learnable_pos_emb = nn.Parameter(torch.randn(max_num_frames, latent_dim))

        self.cond_mask_prob = cond_mask_prob
        # Text Transformer
        self.load_and_freeze_clip(clip_path)

        self.text_pre_proj = nn.Linear(clip_dim, text_latent_dim) \
            if text_latent_dim != clip_dim else nn.Identity()

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=num_text_layers)

        self.text_ln = nn.LayerNorm(text_latent_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_latent_dim, time_embed_dim)
        )

        # Input Embedding
        self.input_embed = nn.Linear(input_feats, latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList([
            LinearTemporalDiffusionTransformerDecoderLayer(
                latent_dim=latent_dim,
                text_latent_dim=text_latent_dim,
                time_embed_dim=time_embed_dim,
                ffn_dim=ff_size,
                num_head=num_heads,
                dropout=dropout) for _ in range(num_layers)]
        )

        self.out = zero_module(nn.Linear(latent_dim, input_feats))

    def load_and_freeze_clip(self, clip_path):
        self.clip_model = CLIPModel.from_pretrained(clip_path)  # Must set jit=False for training
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        # Freeze CLIP weights
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

    def encode_text(self, text):
        with torch.no_grad():
            inputs = self.clip_processor(text, return_tensors="pt", padding=True,
                                         truncation=True, max_length=77).to('cuda')
            input_ids = inputs['input_ids']
            # not pass through the final layer norm in clip
            last_hidden_states = self.clip_model.text_model(**inputs)[0]  # last layer hidden states for all tokens
        # T, B, D
        last_hidden_states = self.text_pre_proj(last_hidden_states)
        last_hidden_states = self.textTransEncoder(last_hidden_states)
        last_hidden_states = self.text_ln(last_hidden_states)
        # B C
        pooled_output = self.text_proj(last_hidden_states[
                                           torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
                                           (input_ids.to(dtype=torch.int32,
                                                         device=last_hidden_states.device) == self.clip_model.text_model.eos_token_id)
                                       .to(torch.int32)
                                       .argmax(dim=-1),
                                       ])

        return last_hidden_states, pooled_output

    def forward_trans(self,
                      x,
                      timesteps,
                      text_last_feature,
                      text_pooled_feature,
                      num_frames=None):

        B, T, D = x.shape
        emb = (self.time_embed(timestep_embedding(timesteps, self.latent_dim).to(x))
               + text_pooled_feature)
        h = self.input_embed(x)
        h = h + self.learnable_pos_emb[:T][None]

        if num_frames is not None:
            src_mask = create_src_key_padding_mask(num_frames).to(x.device)
        else:
            src_mask = torch.zeros([B, T]).to(x)
        for layer in self.temporal_decoder_blocks:
            h = layer(h, text_last_feature, emb, src_mask)

        output = self.out(h).view(B, T, -1).contiguous()
        return output

    def random_mask_cond(self, cond):
        batch_size = len(cond)
        for i in range(batch_size):
            if random.random() < self.cond_mask_prob:
                cond[i] = ''
        return cond

    def infer(self,
              caption: List[str],
              num_frames: List[int],
              num_inference_steps: int = 50,
              guidance_scale: float = 1.,
              show_progress=True,
              generator=None,
              eta=1.
              ):
        if isinstance(caption, str):
            caption = [caption]
        if isinstance(num_frames, int):
            num_frames = [num_frames]

        batch_size = len(caption)

        text_last_feature, text_pooled_feature = self.encode_text(caption)
        do_classifier_free_guidance = guidance_scale > 1.0
        assert not do_classifier_free_guidance, 'not implemented for motion diffuse'

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
            model_output = self.forward_trans(x_t, [t] * batch_size,
                                              text_last_feature, text_pooled_feature,
                                              num_frames)
            # mix condition guidance result and unconditional ones

            # compute the previous noisy sample x_t -> x_t-1
            step_result = self.test_scheduler.step(
                model_output, t, x_t, **extra_step_kwargs)
            x_t = step_result['prev_sample']
        return x_t

    def forward_loss(self, inputs, data_samples):
        x = inputs['motion']
        B, T, _ = x.shape
        text = data_samples.get('caption')
        text = self.random_mask_cond(text)
        num_frames = data_samples.get('num_frames')
        device = x.device

        noise = self.sample_noise(x.shape).to(device)
        timesteps = torch.randint(
            low=0,
            high=self.scheduler.num_train_timesteps, size=(B,),
            device=x.device).long()

        text_last_feature, text_pooled_feature = self.encode_text(text)

        if self.scheduler.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(x, noise, timesteps)
        elif self.scheduler.prediction_type == 'sample':
            gt = x
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.prediction_type}')

        x_t = self.scheduler.add_noise(x, noise, timesteps)
        model_output = self.forward_trans(
            x_t,
            timesteps,
            text_last_feature,
            text_pooled_feature,
            num_frames,
        )
        mask = create_src_key_padding_mask(num_frames).unsqueeze(-1).to(gt.device)
        model_output = model_output * ~mask
        gt = gt * ~mask
        return {
            'loss': F.mse_loss(model_output, gt, reduction='mean')
        }

    def forward_predict(self, inputs, data_samples: DataSample):
        caption = data_samples.get('caption')
        num_frames = data_samples.get('num_frames')

        pred_motion = self.infer(caption, num_frames)
        data_samples.pred_motion = pred_motion
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
