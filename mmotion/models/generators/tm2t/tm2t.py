from typing import Dict, Union, List

import os
import torch
from mmengine import Config

from mmengine.model import BaseModel
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration, AutoConfig, BatchEncoding, \
    GenerationConfig
from torch import nn
from mmotion.models.generators.motion_llm.tokenizers import MotionVQVAE
from mmotion.registry import MODELS
from mmotion.structures import DataSample
from mmotion.utils.typing import SampleList


def create_attention_mask_from_lengths(input_ids, lengths: Union[List[int], torch.Tensor]):
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.long, device=input_ids.device)

    range_tensor = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)

    mask = range_tensor < lengths.unsqueeze(1)

    return mask.to(torch.bool)


def trim_to_eos(output_ids, eos_token_id):
    output_ids = output_ids.cpu().tolist()

    trimmed = []
    for seq in output_ids:
        if eos_token_id in seq:
            eos_index = seq.index(eos_token_id)
            if eos_index == -1:
                trimmed.append(seq)
            else:
                trimmed.append(seq[:eos_index])
        else:
            trimmed.append(seq)
    return trimmed


def modify_t5_embeddings(model: T5ForConditionalGeneration, part: str, new_vocab_size: int):
    assert part in ['encoder', 'decoder'], 'part should be encoder or decoder'
    assert not model.config.tie_word_embeddings, \
        'tie_word_embeddings should be False since the src and target vocab is not same'
    if part == 'encoder':
        old_embed = model.encoder.embed_tokens
        old_vocab_size, embed_dim = old_embed.weight.shape

        new_embed = nn.Embedding(new_vocab_size, embed_dim)
        with torch.no_grad():
            new_embed.weight.normal_(mean=0.0, std=embed_dim ** -0.5)

        model.encoder.embed_tokens = new_embed

    elif part == 'decoder':
        old_embed = model.decoder.embed_tokens
        old_vocab_size, embed_dim = old_embed.weight.shape

        new_embed = nn.Embedding(new_vocab_size, embed_dim)
        with torch.no_grad():
            new_embed.weight.normal_(mean=0.0, std=embed_dim ** -0.5)

        model.decoder.embed_tokens = new_embed

        # head
        new_lm_head = nn.Linear(embed_dim, new_vocab_size, bias=False)

        with torch.no_grad():
            new_lm_head.weight.normal_(mean=0.0, std=embed_dim ** -0.5)

        model.lm_head = new_lm_head

    model.config.vocab_size = new_vocab_size
    model.config.decoder_start_token_id = model.config.decoder_start_token_id  # 确保配置一致


@MODELS.register_module()
class TM2T(BaseModel):
    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.motion_tokenizer.train(False)
        self.motion_tokenizer.requires_grad_(False)

    def __init__(self,
                 motion_tokenizer: Dict,
                 pretrained_tokenizer: str,
                 transformer_config: str,
                 t2m: bool = True,
                 generation_config=None,
                 data_preprocessor: Dict = None,
                 init_cfg=None
                 ):
        super(TM2T, self).__init__(data_preprocessor, init_cfg)
        self.is_t2m = t2m
        self.motion_tokenizer: MotionVQVAE = self.build_motion_tokenizer(motion_tokenizer)

        self.text_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_tokenizer)
        self.transformer: T5ForConditionalGeneration = T5ForConditionalGeneration._from_config(
            AutoConfig.from_pretrained(transformer_config)
        )

        self.text_tokenizer.add_tokens('<|end_of_motion|>', special_tokens=True)
        self.motion_eos_token_id = self.text_tokenizer.convert_tokens_to_ids('<|end_of_motion|>')
        if t2m:
            # set an eos token for motion
            modify_t5_embeddings(self.transformer, 'decoder', len(self.text_tokenizer.get_vocab()) + 1)
        else:
            modify_t5_embeddings(self.transformer, 'encoder', len(self.text_tokenizer.get_vocab()) + 1)
        self.set_generation_config(generation_config)

    def build_motion_tokenizer(self, motion_tokenizer_cfg: Dict) -> MotionVQVAE:
        """
        :param motion_tokenizer_cfg: vqvae config
        :return: Vqvae module.
        """
        type = motion_tokenizer_cfg['type']
        if os.path.isfile(type):
            # allow using config file path as type, simplify config writing.
            init_cfg = motion_tokenizer_cfg.pop('init_cfg', None)
            motion_tokenizer_cfg = Config.fromfile(type)['model']
            if init_cfg is not None:
                motion_tokenizer_cfg['init_cfg'] = init_cfg

        motion_tokenizer: MotionVQVAE = MODELS.build(motion_tokenizer_cfg).eval()
        if motion_tokenizer_cfg.get('init_cfg', None) is not None:
            motion_tokenizer.init_weights()
        motion_tokenizer.requires_grad_(False)
        return motion_tokenizer

    def set_generation_config(self, generation_config: Dict = None):
        if self.is_t2m:
            eos_token_id = [
                self.motion_eos_token_id
            ]
        else:
            eos_token_id = [
                self.text_tokenizer.eos_token_id,
                self.text_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        default_generation_config = dict(
            max_length=200,
            do_sample=True,
            eos_token_id=eos_token_id,
            top_k=100,
            num_beams=2,

        )

        if generation_config is not None:
            default_generation_config.update(generation_config)

        self.generation_config = GenerationConfig(
            **default_generation_config
        )

    def forward(self,
                inputs: Dict,
                data_samples: Union[DataSample, SampleList] = None,
                mode: str = 'loss') -> Union[Dict[str, torch.Tensor], list]:
        if mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'loss':
            return self.forward_loss(inputs, data_samples)
        elif mode == 'tensor':
            return self.forward_tensor(inputs, data_samples)
        else:
            raise NotImplementedError(
                'Forward is not implemented now, please use infer.')

    def forward_loss(self, inputs: Dict, data_samples: DataSample):
        batch_input = self.prepare_input(inputs, data_samples)

        batch_output = self.transformer(**batch_input)
        return {
            'loss': batch_output.loss
        }

    def prepare_input(self, inputs, data_samples):
        if self.is_t2m:
            return self.prepare_input_t2m(inputs, data_samples)
        else:
            return self.prepare_input_m2t(inputs, data_samples)

    def prepare_input_m2t(self, inputs, data_samples):
        motion = inputs['motion']
        num_frames = data_samples.get('num_frames')
        batch_size = len(motion)
        input_ids = self.motion_tokenizer.encode(motion)
        input_ids = torch.cat([input_ids, torch.zeros([batch_size, 1]).to(input_ids)], dim=-1)
        # add eos of motion
        num_valid_tokens = [(nf // self.motion_tokenizer.downsample_rate) for nf in num_frames]
        rows = torch.arange(batch_size)
        input_ids[rows, num_valid_tokens] = self.motion_eos_token_id
        num_valid_tokens = [item + 1 for item in num_valid_tokens]

        # create motion attention mask
        attention_mask = create_attention_mask_from_lengths(input_ids, num_valid_tokens)

        labels = None
        caption = data_samples.get('caption')
        if caption is not None:
            text_inputs: BatchEncoding = self.text_tokenizer(
                caption, padding=True, padding_side='right',
                return_tensors='pt', add_special_tokens=True).to(input_ids.device)
            labels = text_inputs['input_ids']
            labels[labels == self.text_tokenizer.pad_token_id] = -100
        batch_input = BatchEncoding(
            {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        )
        return batch_input

    def prepare_input_t2m(self, inputs, data_samples):
        caption = data_samples.get('caption')
        num_frames = data_samples.get('num_frames')
        batch_size = len(caption)
        text_inputs: BatchEncoding = self.text_tokenizer(
            caption, padding=True, padding_side='right',
            return_tensors='pt', add_special_tokens=True).to('cuda')
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']

        labels = None
        motion = inputs.get('motion')
        if motion is not None:
            motion_ids = self.motion_tokenizer.encode(motion)
            motion_ids = torch.cat([motion_ids, torch.zeros([batch_size, 1]).to(motion_ids)], dim=-1)
            # add eos of motion
            num_valid_tokens = [(nf // self.motion_tokenizer.downsample_rate) for nf in num_frames]

            rows = torch.arange(batch_size)
            motion_ids[rows, num_valid_tokens] = self.motion_eos_token_id
            num_valid_tokens = [item + 1 for item in num_valid_tokens]
            # create motion attention mask

            motion_attn_mask = create_attention_mask_from_lengths(motion_ids, num_valid_tokens)
            motion_ids[~motion_attn_mask] = -100
            labels = motion_ids
        batch_input = BatchEncoding(
            {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        )
        return batch_input

    @torch.no_grad()
    def forward_predict(self, inputs, data_samples):
        batch_input = self.prepare_input(inputs, data_samples)
        output_ids = self.transformer.generate(**batch_input,
                                               generation_config=self.generation_config,
                                               synced_gpus=True)
        if self.is_t2m:
            device = output_ids.device
            # the last token output from transformer is eos
            output_ids = trim_to_eos(output_ids, self.motion_eos_token_id)
            output_ids = [seq[1:] for seq in output_ids]
            valid_length = [len(seq) for seq in output_ids]
            output_ids = [torch.tensor(seq, dtype=torch.int64, device=device) for seq in
                          output_ids]
            output_ids = pad_sequence(output_ids, batch_first=True, padding_value=0).to(torch.int64)

            pred_motion = self.motion_tokenizer.decode(output_ids, is_idx=True)

            unpadded_motions = []

            for m, valid_l in zip(pred_motion, valid_length):
                valid_l = int(self.motion_tokenizer.downsample_rate * valid_l)
                m = m[:valid_l]
                unpadded_motions.append(m)

            data_samples.set_field(unpadded_motions, 'pred_motion')
        else:
            output_text = self.text_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            data_samples.set_field(output_text, 'pred_caption')

        data_samples.set_data(inputs)
        data_samples = self.destruct_motion(data_samples)
        return data_samples

    def destruct_motion(self, data_samples: DataSample):
        # calculate joints positions from motion vectors.
        new_data_samples = []
        data_samples = data_samples.split(allow_nonseq_value=True)
        for data_sample in data_samples:
            for key, value in data_sample.to_dict().items():
                if key.endswith('motion') and value is not None:
                    if 'pred' not in key:
                        # need to destruct padding
                        value = self.data_preprocessor.destruct(value, data_sample)
                    else:
                        value = self.data_preprocessor.destruct(value)
                    data_sample.set_field(value, key)
                    joints_key = key.replace('motion', 'joints')
                    joints = self.data_preprocessor.vec2joints(value, data_sample)
                    data_sample.set_field(joints, joints_key)

            data_sample = self.data_preprocessor.merge_completion_interaction(data_sample)
            new_data_samples.append(data_sample)
        return new_data_samples
