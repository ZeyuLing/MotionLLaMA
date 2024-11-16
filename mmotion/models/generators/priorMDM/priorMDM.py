from typing import Dict, List

import torch
from tqdm import tqdm

from mmotion.models.generators.mdm import MDM
from mmotion.models.generators.mdm.mdm import create_src_key_padding_mask
from mmotion.models.generators.priorMDM.multi_person_block import MultiPersonBlock
from mmotion.registry import MODELS
from torch import nn
import torch.nn.functional as F

from mmotion.structures import DataSample


@MODELS.register_module()
class PriorMDM(MDM):
    def __init__(self,
                 scheduler: Dict,
                 num_end_layers: int = 8,
                 num_mp_layers=2,
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
        super(PriorMDM, self).__init__(
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

        self.requires_grad_(False)

        self.multi_person_block = MultiPersonBlock(
            latent_dim=latent_dim,
            num_layers=num_mp_layers,
        )
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          batch_first=True,
                                                          activation=activation)
        self.seqTransEncoder_end = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=num_end_layers)

        self.multi_person_block.requires_grad_(True)
        self.seqTransEncoder_end.requires_grad_(True)

    def forward_trans(self, x_a, x_b, timesteps, cond, num_frames=None):
        if len(cond.shape) < 3:
            cond = cond.unsqueeze(1)
        mask = None
        if num_frames is not None:
            # take embedding token into account
            mask = create_src_key_padding_mask([n + 1 for n in num_frames]).to(x_a.device)
        emb = self.embed_timestep(timesteps)  # [bs, 1, d]
        emb = emb + cond

        x_a = self.input_process(x_a)
        x_b = self.input_process(x_b)

        x_a = torch.cat([emb, x_a], dim=1)
        x_b = torch.cat([emb, x_b], dim=1)

        x_a = self.sequence_pos_encoder(x_a)
        x_b = self.sequence_pos_encoder(x_b)

        x_a = self.seqTransEncoder(x_a, src_key_padding_mask=mask)
        x_b = self.seqTransEncoder(x_b, src_key_padding_mask=mask)
        delta_xb = self.multi_person_block(x_a, x_b, mask)

        x_b = self.seqTransEncoder_end(x_b + delta_xb, src_key_padding_mask=mask)[:, 1:]
        x_a = self.output_process(x_a[:, 1:])
        x_b = self.output_process(x_b)

        return x_a, x_b

    def forward_loss(self, inputs: Dict, data_samples: DataSample):
        motion_a = inputs.get('motion')
        motion_b = inputs.get('interactor_motion')
        union_caption = data_samples.get('union_caption')
        num_frames = data_samples.get('num_frames')

        cond = self.encode_text(union_caption)

        batch_size = len(motion_a)
        noise_a = self.sample_noise(motion_a.shape).to(motion_a)
        noise_b = self.sample_noise(motion_b.shape).to(motion_b)

        timesteps = torch.randint(
            low=0,
            high=self.scheduler.num_train_timesteps, size=(batch_size,),
            device=noise_a.device).long()

        noisy_a = self.scheduler.add_noise(motion_a, noise_a, timesteps)
        noisy_b = self.scheduler.add_noise(motion_b, noise_b, timesteps)

        if self.scheduler.prediction_type == 'epsilon':
            gt = noise_b
        elif self.scheduler.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(motion_b, noise_b, timesteps)
        elif self.scheduler.prediction_type == 'sample':
            gt = motion_b
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.prediction_type}')

        _, model_output_b = self.forward_trans(noisy_a, noisy_b, timesteps, cond, num_frames)
        mask = create_src_key_padding_mask(num_frames).unsqueeze(-1).to(gt.device)
        model_output_b = model_output_b * ~mask

        loss_dict = {}
        gt = gt * ~mask
        loss_mse = F.mse_loss(model_output_b, gt, reduction='mean')
        loss_dict['loss_mse'] = loss_mse
        return loss_dict

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
            self.nfeats * 2,
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
            xt_a, xt_b = torch.chunk(x_t, 2, dim=-1)
            model_output_a, model_output_b = self.forward_trans(xt_a, xt_b,
                                                                batch_size * [t],
                                                                text_pooled_feature,
                                                                num_frames)
            model_output = torch.cat([model_output_a, model_output_b], dim=-1)
            step_result = self.test_scheduler.step(
                model_output, t, x_t, **extra_step_kwargs)
            x_t = step_result['prev_sample']
        pred_a, pred_b = x_t.chunk(2, dim=-1)
        return pred_a, pred_b

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
