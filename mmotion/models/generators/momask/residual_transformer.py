from functools import partial
from typing import Dict

from einops import repeat, einsum, rearrange
from torch import nn
import torch.nn.functional as F
import torch
from mmotion.models.generators.momask import MomaskTemporalTransformer
from mmotion.models.generators.momask.in_out_process import OutputProcess_Bert
from mmotion.models.generators.momask.momask_utils import q_schedule, cal_performance, top_k, gumbel_sample
from mmotion.models.generators.tma.utils import lengths_to_mask
from mmotion.registry import MODELS
from mmotion.structures import DataSample


@MODELS.register_module()
class MomaskResidualTransformer(MomaskTemporalTransformer):
    def __init__(self, rvq_cfg: Dict, num_codebooks=3, vocab_size: int = 1024,
                 code_dim=512, latent_dim=384, ff_size=1024, num_layers=8,
                 num_heads=6, dropout=0.2, clip_dim=512, cond_drop_prob=.1,
                 clip_path: str = 'checkpoints/vit_base_patch32',
                 data_preprocessor=None, init_cfg=None):
        super().__init__(rvq_cfg, vocab_size, code_dim, latent_dim,
                         ff_size, num_layers, num_heads, dropout,
                         clip_dim, cond_drop_prob, clip_path,
                         data_preprocessor, init_cfg)
        self.num_codebooks = num_codebooks
        self.output_process = OutputProcess_Bert(out_feats=code_dim, latent_dim=latent_dim)
        self.pad_id = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, code_dim)
        self.embed_proj_shared_weight = nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(num_codebooks - 2, vocab_size + 1, code_dim)))
        self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, vocab_size + 1, code_dim)))
        self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, vocab_size + 1, code_dim)))
        self.output_proj_bias = None
        self.registered = False

        self.encode_quant = partial(F.one_hot, num_classes=num_codebooks)
        self.quant_emb = nn.Linear(num_codebooks, latent_dim)

        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def process_embed_proj_weight(self):
        # q-1 c d
        self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
        self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)

    def output_project(self, logits, qids):
        '''
        :logits: (bs, seqlen, code_dim)
        :qids: (bs)

        :return:
            -logits (bs, seqlen, codebook size)
        '''
        # (num_qlayers-1, num_token, code_dim) -> (bs, num_token, code_dim)
        output_proj_weight = self.output_proj_weight[qids]
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]
        output = einsum(output_proj_weight, logits, 'b c d, b n d -> b n c')
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output

    def forward_trans(self, motion_codes, qids, cond, padding_mask, force_mask=False):
        '''
        :param motion_codes: (b, seqlen, d)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :return:
            -logits: (b, num_token, seqlen)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)

        x = self.input_process(motion_codes)

        q_onehot = self.encode_quant(qids).float().to(x.device)
        q_emb = self.quant_emb(q_onehot).unsqueeze(1)  # (b, 1, latent_dim)
        cond = cond.unsqueeze(1)  # (b, 1, latent_dim)

        x = self.position_enc(x)
        xseq = torch.cat([cond, q_emb, x], dim=1)  # (seqlen+2, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)  # (b, seqlen+2)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[:, 2:]
        logits = self.output_process(output)
        return logits

    def mask_forward(self, all_indices, cond_vector, m_lens):
        """
        :param all_indices: (q, b, n)
        :param cond_vector b d
        :param m_lens: (b,)
        :return:
        """

        self.process_embed_proj_weight()

        num_quant_layers, bs, ntokens = all_indices.shape
        all_indices = rearrange(all_indices, 'q b n -> b n q')
        device = all_indices.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, device, ntokens)  # (b, n)

        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id)  # (b, n, q)

        # randomly sample quantization layers to work on, [1, num_q)
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        # print(self.token_embed_weight.shape, all_indices.shape)
        token_embed = repeat(self.token_embed_weight, 'q c d -> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])

        all_codes = token_embed.gather(1, gather_indices)  # (b, n, d, q-1)
        cumsum_codes = torch.cumsum(all_codes, dim=-1)  # (b, n, d, q-1)

        active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # (b, n)
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        logits = self.forward_trans(history_sum, active_q_layers, cond_vector, ~non_pad_mask)
        logits = self.output_project(logits, active_q_layers - 1)
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc

    def forward_loss(self, inputs: torch, data_samples: DataSample):
        motion = inputs['motion']
        num_frames = data_samples.get('num_frames')
        caption = data_samples.get('caption')
        if caption is None:
            caption = [''] * len(motion)
        caption_token = self.encode_text(caption)
        motion_ids: torch.Tensor = self.rvq.encode(motion)  # [n_codebook, b, n]
        num_tokens = [nf // self.rvq.downsample_rate for nf in num_frames]
        ce_loss, pred_id, acc = self.mask_forward(motion_ids, caption_token, num_tokens)

        return {'loss': ce_loss}

    def forward_with_cond_scale(self,
                                motion_codes,
                                q_id,
                                cond_vector,
                                padding_mask,
                                cond_scale=5, ):
        """
        :param motion_codes: predicted latent motion codes of current codebooks, [b, n, d]
        :param q_id:
        :param cond_vector: text condition vector [b, d]
        :param padding_mask:
        :param cond_scale: Control the mixing ratio between conditional logits and unconditional logits
        :return:
        """
        bs = motion_codes.shape[0]
        # if cond_scale == 1:
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)

        logits = self.forward_trans(motion_codes, qids, cond_vector, padding_mask)
        logits = self.output_project(logits, qids - 1)
        if cond_scale == 1:
            return logits

        aux_logits = self.forward_trans(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
        aux_logits = self.output_project(aux_logits, qids - 1)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    def generate_tokens(self,
                        top_motion_ids,
                        cond_vector: torch.Tensor,
                        m_lens,
                        temperature=1,
                        topk_filter_thres=0.9,
                        cond_scale=5
                        ):
        """
        :param top_motion_ids: first layer idx predicted by rvq
        :param cond_vector: text condition token
        :param m_lens:
        :param temperature:
        :param topk_filter_thres:
        :param cond_scale:
        :return:
        """

        self.process_embed_proj_weight()

        batch_size, seq_len = top_motion_ids.shape

        padding_mask = ~lengths_to_mask(m_lens, top_motion_ids.device, seq_len)
        # print(padding_mask.shape, motion_ids.shape)
        motion_ids = torch.where(padding_mask, self.pad_id, top_motion_ids)
        all_indices = [motion_ids]
        history_sum = 0

        for i in range(1, self.num_codebooks):
            token_embed = self.token_embed_weight[i - 1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=0)
        all_indices = torch.where(all_indices == self.pad_id, 0, all_indices)
        return all_indices

    def forward_predict(self, inputs: Dict, data_samples: DataSample):
        motion = inputs['motion']
        num_frames = data_samples.get('num_frames')
        caption = data_samples.get('caption')
        if caption is None:
            caption = [''] * len(motion)
        caption_token = self.encode_text(caption)
        # q b n
        motion_ids = self.rvq.encode(motion)
        top_motion_ids = motion_ids[0]
        num_tokens = [nf // self.rvq.downsample_rate for nf in num_frames]
        all_indices = self.generate_tokens(top_motion_ids, caption_token, num_tokens)
        pred_motion = self.rvq.decode(all_indices, is_idx=True)
        data_samples.set_field(pred_motion, 'pred_motion')
        data_samples.set_field(num_frames, 'pred_motion_num_frames')
        data_samples = self.data_preprocessor.postprocess_data_sample(data_samples)
        return data_samples
