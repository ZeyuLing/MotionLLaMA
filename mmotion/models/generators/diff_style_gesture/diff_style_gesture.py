from typing import Dict, List

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from torch import nn
import torch.nn.functional as F
from mmotion.models.generators.mdm.mdm import MDM, create_src_key_padding_mask
from mmotion.models.generators.diff_style_gesture.local_attention import LocalAttention
from mmotion.models.generators.diff_style_gesture.rotary_emb import RotaryEmbedding
from mmotion.models.generators.diff_style_gesture.wav_encoder import WavEncoder
from einops import repeat, rearrange
from tqdm import tqdm
from mmotion.structures import DataSample
from mmotion.registry import MODELS


@MODELS.register_module()
class DiffStyleGesture(MDM):
    def __init__(self,
                 scheduler: Dict,
                 wav_model_path: str = 'checkpoints/wav2vec2-base-960h',
                 audio_hidden_size: int = 64,
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
                 init_cfg=None):
        super(DiffStyleGesture, self).__init__(
            scheduler=scheduler,
            test_scheduler=test_scheduler,
            ff_size=ff_size,
            activation=activation,
            num_heads=num_heads,
            clip_dim=clip_dim,
            num_layers=num_layers,
            nfeats=nfeats,
            latent_dim=latent_dim,
            dropout=dropout,
            cond_mask_prob=cond_mask_prob,
            clip_path=clip_path,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
        self.audio_hidden_size = audio_hidden_size
        self.load_wav_model(wav_model_path)

        self.rotary_emb = RotaryEmbedding(latent_dim)
        self.cross_local_attention = LocalAttention(
            window_size=15,  # window size. 512 is optimal, but 256 or 128 yields good enough results
            causal=True,  # auto-regressive or not
            look_backward=1,  # each window looks at the window before
            look_forward=0,
            # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
            dropout=0.1,  # post-attention dropout
            exact_windowsize=False
            # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
        )

        self.input_process2 = nn.Linear(latent_dim * 2 + audio_hidden_size, latent_dim)

    def load_wav_model(self, wav_model_path):
        self.wav_processor = Wav2Vec2Processor.from_pretrained(wav_model_path)
        self.wav_model = Wav2Vec2Model.from_pretrained(wav_model_path)
        self.audio_encoder = WavEncoder(self.wav_model.config.hidden_size, self.audio_hidden_size)
        self.wav_model.eval()
        self.wav_model.requires_grad_(False)

    def encode_audio(self, audio, target_frames):
        with (torch.no_grad()):
            input_values = self.wav_processor(audio.float(), sampling_rate=16000, return_tensors="pt", padding='longest'
                                              ).input_values.squeeze(0).to(audio)
            audio = self.wav_model(input_values).last_hidden_state
        audio = rearrange(audio, 'b n c -> b c n')
        audio = F.interpolate(audio, size=target_frames, mode='linear', align_corners=False)
        audio = rearrange(audio, 'b c n -> b n c')
        audio = self.audio_encoder(audio)
        return audio

    def forward_trans(self, x, timesteps, text: torch.Tensor, audio: torch.Tensor, num_frames=None):
        mask = None
        B, T, C = x.shape
        emb = self.embed_timestep(timesteps)  # [bs, 1, d]
        if text.ndim == 2:
            text = text.unsqueeze(1)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        emb = emb + text

        x = self.input_process(x)
        # merge conditions and motion in channel dimension
        x = torch.cat([x,
                       audio,
                       repeat(emb, 'b 1 c -> b t c', t=T)
                       ], dim=-1)
        x = self.input_process2(x)
        x = self.rotary_emb(x)
        # local attention use 1 as valid, 0 as pad

        if num_frames is not None:
            mask = create_src_key_padding_mask(num_frames).to(x.device)

        x = self.cross_local_attention(x, x, x, ~mask)

        if num_frames is not None:
            mask = create_src_key_padding_mask([n + 1 for n in num_frames]).to(x.device)
        x = torch.cat([emb, x], dim=1)
        x = self.rotary_emb(x)
        x = self.seqTransEncoder(x, src_key_padding_mask=mask)[:, 1:]
        x = self.output_process(x)
        return x

    def forward_loss(self, inputs: Dict, data_samples: DataSample):
        motion = inputs['motion']

        audio = inputs['audio']
        text = data_samples.get('caption')
        num_frames = data_samples.get('num_frames')
        batch_size, T, _ = motion.shape

        audio = self.encode_audio(audio, T)
        text = self.encode_text(text)

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
            text,
            audio,
            num_frames)

        mask = create_src_key_padding_mask(num_frames).unsqueeze(-1).to(gt.device)
        model_output = model_output * ~mask
        gt = gt * ~mask
        loss_dict = dict()
        loss_mse = F.mse_loss(model_output, gt, reduction='mean')
        loss_dict['loss_mse'] = loss_mse
        return loss_dict

    def forward_predict(self, inputs, data_samples: DataSample):
        caption = data_samples.get('caption')
        num_frames = data_samples.get('num_frames')
        audio = inputs.get('audio')

        pred_motion = self.infer(caption, num_frames, audio)
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
            elif key.endswith('music') or key.endswith('audio'):
                value = self.data_preprocessor._undo_pad_audio(value, data_samples)
                data_samples.set_field(value, 'key')
        data_samples = data_samples.split(allow_nonseq_value=True)
        return data_samples

    def infer(self,
              caption: List[str],
              num_frames: List[int],
              audio: torch.Tensor = None,
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
        audio = self.encode_audio(audio, max(num_frames))
        do_classifier_free_guidance = guidance_scale > 1.0
        assert not do_classifier_free_guidance, 'not implemented for mcm'

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
                                              text_pooled_feature, audio,
                                              num_frames)
            # mix condition guidance result and unconditional ones

            # compute the previous noisy sample x_t -> x_t-1
            step_result = self.test_scheduler.step(
                model_output, t, x_t, **extra_step_kwargs)
            x_t = step_result['prev_sample']
        return x_t
