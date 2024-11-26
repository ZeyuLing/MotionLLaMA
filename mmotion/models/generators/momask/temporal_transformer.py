from typing import Dict, Optional, Union

import os
import torch
from mmengine import Config
from mmengine.model import BaseModel
from torch import nn
from torch.distributions import Categorical
from transformers import CLIPModel, CLIPProcessor

from mmotion.models.generators.motion_tokenizer import MotionVQVAE
from mmotion.models.generators.mdm.mdm import PositionalEncoding
from mmotion.models.generators.momask.in_out_process import OutputProcess_Bert
from mmotion.models.generators.momask.momask_utils import uniform, cosine_schedule, get_mask_subset_prob, \
    cal_performance, top_k
from mmotion.models.generators.tma.utils import lengths_to_mask
import torch.nn.functional as F

from mmotion.registry import MODELS
from mmotion.structures import DataSample


@MODELS.register_module()
class MomaskTemporalTransformer(BaseModel):
    def __init__(self, rvq_cfg: Dict, vocab_size: int = 1024,
                 code_dim=512, latent_dim=384, ff_size=1024, num_layers=8,
                 num_heads=6, dropout=0.2, clip_dim=512, cond_drop_prob=.1,
                 clip_path: str = 'checkpoints/vit_base_patch32',
                 data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.noise_schedule = cosine_schedule
        self.rvq = self.build_rvq(rvq_cfg)
        self.cond_drop_prob = cond_drop_prob
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.pad_id = vocab_size + 1
        self.token_emb = nn.Embedding(vocab_size + 2, code_dim)
        self.text_emb = nn.Linear(clip_dim, latent_dim)
        self.input_process = nn.Linear(code_dim, latent_dim) if code_dim != latent_dim else nn.Identity()
        self.position_enc = PositionalEncoding(latent_dim, dropout)
        self.output_process = OutputProcess_Bert(out_feats=vocab_size, latent_dim=latent_dim)
        self.load_and_freeze_clip(clip_path)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu',
                                                          batch_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

    def build_rvq(self, rvq_cfg: Dict):
        """
        :param mm_tokenizer_cfg: vqvae config
        :return: Vqvae module.
        """
        type = rvq_cfg['type']
        if os.path.isfile(type):
            # allow using config file path as type, simplify config writing.
            init_cfg = rvq_cfg.pop('init_cfg', None)
            rvq_cfg = Config.fromfile(type)['model']
            if init_cfg is not None:
                rvq_cfg['init_cfg'] = init_cfg

        rvq: MotionVQVAE = MODELS.build(rvq_cfg).eval()
        if rvq_cfg.get('init_cfg', None) is not None:
            rvq.init_weights()
        return rvq

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
        texts = self.text_emb(texts)
        return texts

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def forward_trans(self,
                      motion_ids: torch.Tensor,
                      cond: torch.Tensor,
                      padding_mask: torch.Tensor,
                      force_mask: bool = False):
        """
        :param motion_ids: b n c
        :param cond: b c
        :param padding_mask: b n, 1 for padding, 0 for valid.
        :param force_mask: mask the condition token
        :return:
        """
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.token_emb(motion_ids)
        x = self.input_process(x)
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=1)  # ( b, seq_len+1,latent_dim)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask]
                                 , dim=1)  # (b, seqlen+1)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[:, 1:]
        logits = self.output_process(output)  # (seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale: float = 4):

        logits = self.forward_trans(motion_ids, cond_vector, padding_mask, force_mask=False)
        if cond_scale == 1:
            return logits

        aux_logits = self.forward_trans(motion_ids, cond_vector, padding_mask, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    def mask_forward(self, motion_ids, cond_token, len_tokens):
        """ Raw text as condition, motion_ids as target output.
        :param motion_ids: target motion ids
        :param cond_token: condition tokens extracted from text encoder
        :param len_tokens: valid lengths of target tokens
        :return:
        """
        bs, ntokens = motion_ids.shape

        # 1 for valid, 0 for padded
        non_pad_mask = lengths_to_mask(len_tokens, motion_ids.device, ntokens)  # (b, n)
        ids = torch.where(non_pad_mask, motion_ids, self.pad_id)

        masked_ids, mask = self.rand_mask_ids(ids, non_pad_mask)

        # only predict the masked positions
        labels = torch.where(mask, ids, self.mask_id)

        logits = self.forward_trans(masked_ids, cond_token, ~non_pad_mask, False)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        return ce_loss, pred_id, acc

    def rand_mask_ids(self, input_ids: torch.Tensor, non_pad_mask: torch.Tensor):
        """ As in Bert, firstly, randomly masked 10% tokens with an incorrect token.
        Secondly, in the mentioned 10%, 90%
        :param input_ids:
        :param non_pad_mask:
        :return:
        """
        bs, ntokens = input_ids.shape
        device = input_ids.device
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask &= non_pad_mask
        x_ids = input_ids.clone()

        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.vocab_size)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)

        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        return x_ids, mask

    @torch.no_grad()
    def generate_tokens(self,
                        cond_vector: torch.Tensor,
                        num_tokens,
                        timesteps: int = 10,
                        cond_scale: float = 4,
                        temperature=1,
                        topk_filter_thres=0.9
                        ):
        device = next(self.parameters()).device
        seq_len = max(num_tokens)

        # 1 for padding, 0 for valid
        padding_mask = ~lengths_to_mask(num_tokens, device, seq_len)
        # print(padding_mask.shape, )

        # Start from all not padding tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * torch.tensor(num_tokens, device=device)).clamp(
                min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask,
                              torch.tensor(self.mask_id, dtype=ids.dtype, device=ids.device),
                              ids)

            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale)

            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            temperature = starting_temperature

            probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
            pred_ids = Categorical(probs).sample()  # (b, seqlen)

            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids

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

    def forward_loss(self, inputs: torch, data_samples: DataSample):
        motion = inputs['motion']
        num_frames = data_samples.get('num_frames')
        caption = data_samples.get('caption')
        caption_token = self.encode_text(caption)
        top_motion_ids: torch.Tensor = self.rvq.encode(motion)[0]  # [n_codebook, b, n]
        num_tokens = [nf // self.rvq.downsample_rate for nf in num_frames]
        ce_loss, pred_id, acc = self.mask_forward(top_motion_ids, caption_token, num_tokens)

        return {'loss': ce_loss}

    @torch.no_grad()
    def forward_predict(self, inputs: Dict, data_samples: DataSample):
        num_frames = data_samples.get('num_frames')
        caption = data_samples.get('caption')
        caption_token = self.encode_text(caption)
        num_tokens = [nf // self.rvq.downsample_rate for nf in num_frames]

        pred_top_rvq_motion_idx = self.generate_tokens(caption_token, num_tokens)
        pred_motion = self.rvq.decode(pred_top_rvq_motion_idx, is_idx=True)
        data_samples.set_field(pred_motion, 'pred_motion')
        data_samples.set_data(inputs)
        for key, value in data_samples.to_dict().items():
            if key.endswith('motion') and value is not None:
                value = self.data_preprocessor.destruct(value, data_samples)
                data_samples.set_field(value, key)
                joints_key = key.replace('motion', 'joints')
                joints = self.data_preprocessor.vec2joints(value, data_samples)
                data_samples.set_field(joints, joints_key)
        data_samples = data_samples.split(allow_nonseq_value=True)
        return data_samples
