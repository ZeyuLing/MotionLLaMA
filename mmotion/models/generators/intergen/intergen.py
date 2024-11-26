from typing import Dict, List
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torch import nn
import torch.nn.functional as F
from mmotion.models.generators.base_diffusion_model import BaseDiffusionModel
from mmotion.models.generators.intergen.intergen_denoiser import InterDenoiser
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
class InterGen(BaseDiffusionModel):
    def __init__(self,
                 cond_mask_prob=.1,
                 clip_path: str = 'checkpoints/clip-vit-large-patch14',
                 decoder_cfg: Dict = None,
                 scheduler: Dict = None,
                 test_scheduler=None,
                 data_preprocessor: Dict = None,
                 init_cfg=None):
        super().__init__(scheduler, test_scheduler, data_preprocessor, init_cfg)
        self.cond_mask_prob = cond_mask_prob
        self.decoder: InterDenoiser = MODELS.build(decoder_cfg)
        self.nfeats = self.decoder.input_feats * 2
        self.load_and_freeze_clip(clip_path)
        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2)
        self.clip_ln = nn.LayerNorm(768)

    def encode_text(self, text):
        with torch.no_grad():
            inputs = self.clip_processor(text, return_tensors="pt", padding=True,
                                         truncation=True, max_length=77).to('cuda')
            input_ids = inputs['input_ids']
            # not pass through the final layer norm in clip
            last_hidden_states = self.clip_model.text_model(**inputs)[0]  # last layer hidden states for all tokens
        # T, B, D
        last_hidden_states = self.clipTransEncoder(last_hidden_states)
        last_hidden_states = self.clip_ln(last_hidden_states)
        # B C
        pooled_output = last_hidden_states[
            torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
            (input_ids.to(dtype=torch.int32,
                          device=last_hidden_states.device) == self.clip_model.text_model.eos_token_id)
            .to(torch.int32)
            .argmax(dim=-1),
        ]

        return pooled_output

    def load_and_freeze_clip(self, clip_path):
        self.clip_model = CLIPModel.from_pretrained(clip_path)  # Must set jit=False for training
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        # Freeze CLIP weights
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

    def mask_cond(self, cond, cond_mask_prob=0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond).to(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs).to(cond) * cond_mask_prob).view(
                [bs] + [1] * len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), (1. - mask)
        else:
            return cond, None

    def forward_loss(self, inputs, data_samples):
        x_a = inputs['motion']
        x_b = inputs['interactor_motion']
        x = torch.cat([x_a, x_b], dim=-1)
        B, T, _ = x.shape
        text = data_samples.get('union_caption')
        num_frames = data_samples.get('num_frames')
        # 0 for valid, 1 for pad
        mask = create_src_key_padding_mask(num_frames).to(x.device)
        device = x.device

        noise = self.sample_noise(x.shape).to(device)
        timesteps = torch.randint(
            low=0,
            high=self.scheduler.num_train_timesteps, size=(B,),
            device=x.device).long()

        text_pooled_feature = self.encode_text(text)
        text_pooled_feature = self.mask_cond(text_pooled_feature, self.cond_mask_prob)[0]

        if self.scheduler.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(x, noise, timesteps)
        elif self.scheduler.prediction_type == 'sample':
            gt = x
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.prediction_type}')

        noise = self.sample_noise(x.shape).to(x)
        x_t = self.scheduler.add_noise(x, noise, timesteps)
        model_output = self.decoder(
            x_t,
            timesteps,
            mask,
            text_pooled_feature
        )
        loss_dict = dict()
        mask = create_src_key_padding_mask(num_frames).unsqueeze(-1).to(gt.device)
        model_output = model_output * ~mask
        gt = gt * ~mask
        loss_mse = F.mse_loss(model_output, gt, reduction='mean')
        loss_dict['loss_mse'] = loss_mse
        return loss_dict

    def forward_predict(self, inputs, data_samples: DataSample):
        caption = data_samples.get('union_caption')
        num_frames = data_samples.get('num_frames')

        pred_motion, pred_interactor_motion = self.infer(caption, num_frames)
        data_samples.set_field(pred_motion, 'pred_motion')
        data_samples.set_field(pred_interactor_motion, 'pred_interactor_motion')
        data_samples.set_data(inputs)
        data_samples = data_samples.split(allow_nonseq_value=True)

        new_data_samples = []
        for data_sample in data_samples:
            for key, value in data_sample.to_dict().items():
                if key.endswith('motion') and value is not None:
                    value = self.data_preprocessor.destruct(value, data_sample)
                    data_sample.set_field(value, key)
                    joints_key = key.replace('motion', 'joints')
                    joints = self.data_preprocessor.vec2joints(value, data_sample)
                    data_sample.set_field(joints, joints_key)

            data_sample = self.data_preprocessor.merge_completion_interaction(data_sample)
            new_data_samples.append(data_sample)
        return new_data_samples

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

        text_pooled_feature = self.encode_text(caption)

        do_classifier_free_guidance = guidance_scale > 1.0
        assert not do_classifier_free_guidance, 'intergen doesnt support guidance'
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
            mask = create_src_key_padding_mask(num_frames).to(x_t.device)
            model_output = self.decoder(x_t,
                                        batch_size * [t],
                                        mask,
                                        text_pooled_feature)
            # compute the previous noisy sample x_t -> x_t-1
            step_result = self.test_scheduler.step(
                model_output, t, x_t, **extra_step_kwargs)
            x_t = step_result['prev_sample']
        pred_a, pred_b = x_t.chunk(2, dim=-1)
        return pred_a, pred_b
