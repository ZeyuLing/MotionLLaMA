from collections import defaultdict
from typing import Dict, Type, Union, List, Tuple

import os

from mmengine import Config
from mmengine.model import BaseModel
from peft import get_peft_model, LoraConfig, TaskType
from tokenizers import AddedToken
from torch import Tensor
from transformers import (PreTrainedTokenizerFast, GenerationConfig, BatchEncoding,
                          AutoModelForCausalLM, LlamaForCausalLM)

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn

from mmotion.models.generators.motion_tokenizer.vqvaes import MotionVQVAE
from mmotion.models.generators.wavtokenizer import WavTokenizer

from mmotion.models.generators.motion_llm.data_collator_utils import DataCollatorForCompletionOnlyLM
from mmotion.models.generators.motion_llm.partial_frozen_embedding import PartialFrozenEmbedding, \
    PartialFrozenLinear
from mmotion.registry import MODELS
from mmotion.structures import DataSample
from mmotion.utils.logger import print_colored_log
from mmotion.utils.param_utils import get_parameter_stats
from mmotion.utils.task import name_modal_mapping, Modality, all_modalities, locatable_modalities
from mmotion.utils.task.modality import is_modal, Audio, Motion, replace_placeholders, Duration
from mmotion.utils.task.prompt.chat_template import DEFAULT_RESPONSE_TEMPLATE, LLAMA_3_TEMPLATE_NO_SYSTEM
from mmotion.utils.task.prompt.prompt_template import get_system_user, get_assistant
from mmotion.utils.typing import SampleList


@MODELS.register_module()
class MotionCausalLM(BaseModel):
    lm: LlamaForCausalLM
    generation_config: GenerationConfig = None
    text_tokenizer: PreTrainedTokenizerFast
    motion_tokenizer: MotionVQVAE
    audio_tokenizer: WavTokenizer
    modal_list = []
    data_collator: DataCollatorForCompletionOnlyLM

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        if hasattr(self, 'motion_tokenizer'):
            self.motion_tokenizer.train(False)
            self.motion_tokenizer.requires_grad_(False)
        if hasattr(self, 'audio_tokenizer'):
            self.audio_tokenizer.train(False)
            self.audio_tokenizer.requires_grad_(False)
        return self

    def __init__(self,
                 text_tokenizer_cfg: Dict = None,
                 mm_tokenizer_cfg: Dict = None,
                 freeze_emb: Union[str, bool] = False,
                 pretrained_lm: str = 'checkpoints/flan-t5-base',
                 lora_config: Dict = None,
                 generation_config: Dict = None,
                 is_pretrain_stage=False,
                 data_preprocessor: Dict = None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.is_pretrain_statge = is_pretrain_stage
        self.freeze_emb = freeze_emb
        self.build_text_tokenizer(pretrained_lm, text_tokenizer_cfg)
        self.build_lm(pretrained_lm)
        self.extend_tokenizer(mm_tokenizer_cfg)
        self.set_lora(lora_config)
        self.build_data_collator()
        self.set_generation_config(generation_config)

    def build_data_collator(self):
        self.data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=self.text_tokenizer,
            response_template=DEFAULT_RESPONSE_TEMPLATE
        )

    def build_lm(self, pretrained_lm: str):
        self.lm = AutoModelForCausalLM.from_pretrained(pretrained_lm,
                                                       attn_implementation='flash_attention_2',
                                                       device_map='cuda')

    def set_lora(self, lora_config: Dict = None):
        """ Set LoRA for causal lm
        :param lora_config: peft style lora config
        :return: None
        """
        peft_config = None
        self.use_lora = False
        if lora_config:
            self.use_lora = True
            print_colored_log('Only LoRA modules trainable')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                peft_type='LORA',
                **lora_config
            )
            self.lm = get_peft_model(self.lm, peft_config, autocast_adapter_dtype=True)

        else:
            self.lm.requires_grad_(True)
            print_colored_log('directly finetune')

        self.freeze_embedding_layers()
        if peft_config:
            self.lm.print_trainable_parameters()

    def freeze_embedding_layers(self):
        if isinstance(self.freeze_emb, bool):
            self.freeze_emb = 'no' if self.freeze_emb is False else 'all'
        assert self.freeze_emb in ['no', 'ori', 'all']
        if self.freeze_emb == 'ori':
            self.lm.get_input_embeddings().requires_grad_(False)
            self.lm.get_output_embeddings().requires_grad_(False)
            self.lm.set_input_embeddings(PartialFrozenEmbedding(
                self.lm.get_input_embeddings(), self.ori_vocab_size))
            self.lm.set_output_embeddings(PartialFrozenLinear(
                self.lm.get_output_embeddings(), self.ori_vocab_size, False))

            all_in_emb_params, trainable_in_emb_params, _ = get_parameter_stats(self.lm.get_input_embeddings())
            all_out_emb_params, trainable_out_emb_params, _ = get_parameter_stats(self.lm.get_output_embeddings())

            print_colored_log('The text part input and output embeddings are frozen')
            print_colored_log(f'Trainable Embedding Params: {all_in_emb_params + all_out_emb_params}, '
                              f'Trainable Embedding Params {trainable_in_emb_params + trainable_out_emb_params}, '
                              f'Trainable rate: {(trainable_in_emb_params + trainable_out_emb_params) / (all_in_emb_params + all_out_emb_params) * 100}%')
        elif self.freeze_emb == 'all':

            self.lm.set_input_embeddings(PartialFrozenEmbedding(
                self.lm.get_input_embeddings(), self.ori_vocab_size))
            self.lm.set_output_embeddings(PartialFrozenLinear(
                self.lm.get_output_embeddings(), self.ori_vocab_size, False))

            self.lm.get_input_embeddings().requires_grad_(False)
            self.lm.get_output_embeddings().requires_grad_(False)
            all_in_emb_params, trainable_in_emb_params, _ = get_parameter_stats(self.lm.get_input_embeddings())
            all_out_emb_params, trainable_out_emb_params, _ = get_parameter_stats(self.lm.get_output_embeddings())

            print_colored_log('The text part input and output embeddings are frozen')
            print_colored_log(f'Trainable Embedding Params: {all_in_emb_params + all_out_emb_params}, '
                              f'Trainable Embedding Params {trainable_in_emb_params + trainable_out_emb_params}, '
                              f'Trainable rate: {(trainable_in_emb_params + trainable_out_emb_params) / (all_in_emb_params + all_out_emb_params) * 100}%')
            print_colored_log('All embedding layers are frozen')
        else:
            self.lm.get_input_embeddings().requires_grad_(True)
            self.lm.get_output_embeddings().requires_grad_(True)
            print_colored_log('All embedding layers are optimized')

    def build_text_tokenizer(self, pretrained_lm: str, text_tokenizer_cfg=None):
        """ Set text tokenizer, and determine the padding strategy
        :param pretrained_lm: path of pretrained llm
        :param text_tokenizer_cfg: tokenizer configuration
        :return:
        """
        self.text_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_lm)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        default_text_tokenizer_config = dict(
        )
        if text_tokenizer_cfg is not None:
            default_text_tokenizer_config.update(text_tokenizer_cfg)
        for key, value in default_text_tokenizer_config.items():
            setattr(self.text_tokenizer, key, value)

    def build_mm_tokenizer(self, mm_tokenizer_cfg: Dict) -> Union[nn.Module, MotionVQVAE, WavTokenizer]:
        """
        :param mm_tokenizer_cfg: tokenizer config
        :return: tokenizer module.
        """
        type = mm_tokenizer_cfg['type']
        if os.path.isfile(type):
            # allow using config file path as type, simplify config writing.
            init_cfg = mm_tokenizer_cfg.pop('init_cfg', None)
            mm_tokenizer_cfg = Config.fromfile(type)['model']
            if init_cfg is not None:
                mm_tokenizer_cfg['init_cfg'] = init_cfg

        mm_tokenizer: nn.Module = MODELS.build(mm_tokenizer_cfg).eval()
        if mm_tokenizer_cfg.get('init_cfg', None) is not None:
            mm_tokenizer.init_weights()
        # return mm_tokenizer.bfloat16().cuda()
        return mm_tokenizer

    def extend_tokenizer(self, mm_tokenizer_cfg: Dict):
        self.modal_need_tokenizer = []
        self.ori_vocab_size = len(self.text_tokenizer)
        add_modal_beos = mm_tokenizer_cfg.pop('add_modal_beos', True)

        for key, value in mm_tokenizer_cfg.items():
            if isinstance(value, dict) and 'type' in value:
                mm_tokenizer_name = f'{key}_tokenizer'
                mm_tokenizer = self.build_mm_tokenizer(value).eval()
                setattr(self, mm_tokenizer_name, mm_tokenizer)
                modal: Type[Modality] = name_modal_mapping[key]
                self.modal_need_tokenizer.append(modal)
                self.add_tokens(modal=modal, num_tokens=mm_tokenizer.codebook_size)

        if add_modal_beos:
            for modality in all_modalities:
                if modality.bos is not None:
                    bos = AddedToken(modality.bos, lstrip=False, rstrip=False, single_word=False, special=True)
                    self.text_tokenizer.add_tokens(bos, special_tokens=True)
                    print_colored_log(
                        f"BOS of {modality.name}, {modality.bos}: {self.text_tokenizer.convert_tokens_to_ids(modality.bos)}")
                if modality.eos is not None:
                    eos = AddedToken(modality.eos, lstrip=False, rstrip=False, single_word=False, special=True)
                    self.text_tokenizer.add_tokens(eos, special_tokens=True)
                    print_colored_log(
                        f"EOS of {modality.name}, {modality.bos}: {self.text_tokenizer.convert_tokens_to_ids(modality.eos)}")

        print_colored_log(f"Original vocabulary size: {self.ori_vocab_size},"
                          f" expanded vocabulary size: {len(self.text_tokenizer.get_vocab())}")

        self.lm.resize_token_embeddings(len(self.text_tokenizer.get_vocab()))

    def add_tokens(self, modal: Type[Modality], num_tokens: int = 512):
        """ Add multi-modal tokens to original llama tokenizer.
        :param modal: motion, audio, ...
        :param num_tokens: equals to corresponding tokenizer codebook size
        :return:
        """

        for i in range(num_tokens):
            new_token = AddedToken(modal.token_format.format(i),
                                   lstrip=True, rstrip=True, single_word=True, special=False)

            self.text_tokenizer.add_tokens(new_token, special_tokens=False)

    def set_generation_config(self, generation_config: Dict = None):
        default_generation_config = dict(
            max_length=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=[
                self.text_tokenizer.eos_token_id,
                self.text_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
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

    @torch.no_grad()
    def fill_conversation(self, conversation_templates: List[List[Dict]], inputs: Dict, data_samples: DataSample) \
            -> Tuple[List[List[Dict]], DataSample]:
        """ Quantize each modality into ids,
        and make ids into multi-modal tokens like <|MOTION_123|>,
         then fill templates with multi-modal tokens
        :param conversation_templates: A batch of conversation templates.
        for exp when batch size is 2:
                [{'role': 'system',
                'content': 'message1'},
                {'role': 'user',
                'content': 'user_message1'},
                {'role': 'assistant',
                'content': 'ass_message1'},
                {'role': 'system',
                'content': 'message2'},
                {'role': 'user',
                'content': 'user_message2'},
                {'role': 'assistant',
                'content': 'ass_message2'}]
        :param inputs: input dict
        :param data_samples: a batch of data samples
        :return: conversations filled with motion, caption or audio.
        """

        task = data_samples.get('task')[0]

        modalities = task.all_modality()
        holder_string_mapping = {}
        for modal in modalities:
            holder = modal.holder
            if holder_string_mapping.get(holder, None):
                continue
            key = [key for key in modal.data_keys if key in inputs.keys() or key in data_samples.all_keys()]
            assert len(key), (task.abbr, modal.name, modal.data_keys, inputs.keys(), data_samples.all_keys())
            key = key[0]
            value = inputs.get(key, data_samples.get(key))

            if is_modal(modal, Motion):
                _, value, data_samples = self.tokenize_motion(value, data_samples, key)
            elif is_modal(modal, Audio):
                _, value, data_samples = self.tokenize_audio(value, data_samples, key)
            elif is_modal(modal, Duration):
                value = [f"{v:.1f}" for v in value]

            for v in value:
                assert len(v), (modal, key, data_samples.get('past_motion_text'), data_samples.all_keys())

            value = [modal.bos + v + modal.eos for v in value]
            holder_string_mapping[holder] = value

        conversation_batch = replace_placeholders(conversation_templates, holder_string_mapping)
        return conversation_batch, data_samples

    def prepare_input_train(self, conversations: List[List[Dict]], data_samples: Union[DataSample, SampleList]) -> \
            Tuple[
                BatchEncoding, DataSample]:
        """ Turn dialog messages into input_ids, labels, attention_mask, ...
        Do padding and truncate to the input
        :param message: A batch of input dialog messages.
        :return:
        """
        texts = self.text_tokenizer.apply_chat_template(conversations,
                                                        tokenize=False,
                                                        chat_template=LLAMA_3_TEMPLATE_NO_SYSTEM)
        data_samples.set_field(texts, 'text')
        if not self.is_pretrain_statge:
            # instruction stage, labels are needed
            texts = [self.text_tokenizer(text,
                                         add_special_tokens=False,
                                         padding=False,
                                         return_overflowing_tokens=False,
                                         return_length=False
                                         ) for text in texts]

            lm_input = self.data_collator(texts).to('cuda')
        else:
            # pretrain stage all tokens are available except padding
            lm_input = self.text_tokenizer(
                texts, add_special_tokens=False, padding=True,
                return_overflowing_tokens=False, return_length=False,
                return_attention_mask=True,
                return_tensors='pt'
            ).to('cuda')
            lm_input['labels'] = lm_input['input_ids']

        return lm_input, data_samples

    def prepare_input_predict(self, conversations: List[List[Dict]], data_samples: DataSample) \
            -> Tuple[BatchEncoding, DataSample]:
        batch_inputs = self.text_tokenizer.apply_chat_template([get_system_user(m) for m in conversations],
                                                               chat_template=LLAMA_3_TEMPLATE_NO_SYSTEM,
                                                               add_generation_prompt=True,
                                                               tokenize=False)
        data_samples.input = batch_inputs
        batch_inputs = self.text_tokenizer(batch_inputs, add_special_tokens=False,
                                           padding='longest', return_tensors='pt',
                                           return_attention_mask=True)

        has_gt = any([seq['role'] == 'assistant' for seq in conversations[0]])
        if has_gt:
            output = self.text_tokenizer.apply_chat_template([get_assistant(m) for m in conversations],
                                                             chat_template=LLAMA_3_TEMPLATE_NO_SYSTEM,
                                                             tokenize=False)
            output = [a.replace(DEFAULT_RESPONSE_TEMPLATE, '') for a in output]

            data_samples.text = [inp + out for inp, out in zip(data_samples.input, output)]
        # batch_inputs['length'] = batch_inputs['input_ids'].shape[1]
        for key, value in batch_inputs.items():
            if isinstance(value, Tensor):
                batch_inputs[key] = value.cuda()
        return batch_inputs, data_samples

    @torch.no_grad()
    def tokenize_audio(self, audio: Tensor,
                       data_samples: DataSample = None,
                       audio_key: str = None) -> Tuple[
        Tensor, List[str], DataSample]:
        """ Quantize the audio vector into indexes with tokenizer
        :param audio: audio vector, [t]
        :param data_samples: data sample for a single sample(not a batch)
        :return: indexes, audio strings, data_samples
        """
        if not hasattr(self, 'audio_tokenizer'):
            return
        if audio_key:
            num_frames = data_samples.get(f'{audio_key}_num_frames')
            num_frames = [nf // self.audio_tokenizer.downsample_rate for nf in num_frames]
            audio_ids = self.audio_tokenizer.encode(audio.detach())[1]
            audio_ids = [ids[:nf] for ids, nf in zip(audio_ids, num_frames)]
            audio_strings = [Audio.index_to_string(ids) for ids in audio_ids]
            data_samples.set_field(audio_ids, f'{audio_key}_ids')
            data_samples.set_field(audio_strings, f'{audio_key}_text')
        else:
            audio_ids = self.audio_tokenizer.encode(audio.detach())[1]
            audio_strings = [Audio.index_to_string(ids) for ids in audio_ids]
        return audio_ids, audio_strings, data_samples

    @torch.no_grad()
    def tokenize_motion(self,
                        motion: Tensor,
                        data_samples: DataSample = None,
                        motion_key: str = None) -> Tuple[Tensor, List[str], DataSample]:
        """ Quantize the motion vector into indexes with tokenizer, and save the idx to data_samples.
        Some idx will be used in Completion tasks.
        :param motion: A motion sample, [1, t, c]
        :param data_samples: data sample for a single sample(not a batch)
        :return: indexes, motion strings, data_samples
        """
        # encode
        # [1, n] -> [n]
        if motion_key:
            num_frames = data_samples.get(f'{motion_key}_num_frames')
            # num_frames = [nf // self.motion_tokenizer.downsample_rate for nf in num_frames]
            motion_ids = [self.motion_tokenizer.encode(m[:nf][None])[0] for m, nf in zip(motion, num_frames)]

            motion_string = [Motion.index_to_string(ids) for ids in motion_ids]
            data_samples.set_field(motion_ids, f'{motion_key}_ids')
            data_samples.set_field(motion_string, f'{motion_key}_text')

        else:
            motion_ids = self.motion_tokenizer.encode(motion.detach())
            motion_string = [Motion.index_to_string(ids) for ids in motion_ids]

        recons_motion, num_frames = self.detokenize_motion(motion_ids)
        data_samples.set_field(recons_motion, f'recons_{motion_key}')
        data_samples.set_field(num_frames, f'recons_{motion_key}_num_frames')
        return motion_ids, motion_string, data_samples

    def forward_lm(self, inputs, data_samples) -> CausalLMOutputWithPast:
        # input_ids, labels, attention_masks
        conversations: List[List[Dict]] = data_samples.get('conversation')
        conversations, data_samples = self.fill_conversation(conversations, inputs, data_samples)
        lm_input, data_samples = self.prepare_input_train(conversations, data_samples)
        output: CausalLMOutputWithPast = self.lm(**lm_input)
        pred_ids = torch.argmax(output['logits'], dim=-1)
        output['pred_ids'] = pred_ids
        return output

    def forward_loss(self, inputs, data_samples):
        loss_dict = {
            'loss': self.forward_lm(inputs, data_samples).loss
        }
        return loss_dict

    def forward_predict(self, inputs: Dict, data_samples: DataSample):

        conversations: List[List[Dict]] = data_samples.get('conversation')
        conversations, data_samples = self.fill_conversation(conversations, inputs, data_samples)
        lm_inputs, data_samples = self.prepare_input_predict(conversations, data_samples)

        input_output_ids = self.lm.generate(
            **lm_inputs,
            generation_config=self.generation_config
        )
        input_length = lm_inputs['input_ids'].shape[1]

        pred_text = self.text_tokenizer.batch_decode(input_output_ids, skip_special_tokens=False)
        pred_output = self.text_tokenizer.batch_decode(
            input_output_ids[:, input_length:], skip_special_tokens=False)

        data_samples.set_field(pred_text, 'pred_text')
        data_samples: DataSample = self.multimodal_detokenize(pred_output, data_samples)
        data_samples: SampleList = self.post_process(inputs, data_samples)

        return data_samples

    def post_process(self, inputs: Dict, data_samples: DataSample) -> SampleList:
        """ Process the ground truth text and gt text to corresponding modality like
         caption, motion, audio, ...
        :param data_samples: including all data information.
        :return:
        """
        # save gt/condition to data samples for convenient evaluation and visualization
        data_samples.set_data(inputs)
        data_samples = self.data_preprocessor.postprocess_data_sample(data_samples)
        return data_samples

    @torch.no_grad()
    def multimodal_detokenize(self, batch_text: List[str], data_samples: DataSample) -> DataSample:
        """
        Decode multi-modal text into specific modalities like motion and audio.
        Batch decode all sequences to ensure synchronization across GPUs.

        :param batch_text: Output text from the LLM.
        :param data_samples: Data samples containing tasks and other information.
        :return: Decoded data samples with predictions for each modality.
        """
        mm_text = self.extract_mm_text(batch_text)

        for modal, text in mm_text.items():
            if is_modal(modal, Motion):
                # Motion or Interactor Motion
                pred_motion_ids = [Motion.string_to_index(t) for t in text]
                pred_motion, pred_motion_num_frames = self.detokenize_motion(pred_motion_ids)
                data_samples.set_field(pred_motion, f'pred_{modal.name}')
                data_samples.set_field(pred_motion_num_frames, f'pred_{modal.name}_num_frames')

            elif is_modal(modal, Audio):
                # Audio or Music
                if hasattr(self, 'audio_tokenizer'):
                    pred_audio_ids = [modal.string_to_index(t) for t in text]
                    pred_audio, pred_audio_num_frames = self.detokenize_audio(pred_audio_ids)
                    data_samples.set_field(pred_audio, f'pred_{modal.name}')
                    data_samples.set_field(pred_audio_num_frames, f'pred_{modal.name}_num_frames')
            else:
                data_samples.set_field(text, f'pred_{modal.name}')

        return data_samples

    def extract_mm_text(self, batch_text: List[str]) -> Dict:
        """
            1, Firstly, fetch special tokens from the output of causal lm
            2, The predicted modal of each sample may differ in each batch,
            once a sample A has modal X, other samples should make dummy index
            to keep synchronization with sample A. We use [0] as the dummy index
        :param batch_text: causal lm predicted text
        :return: Modality -> corresponding sub string in the LLM output text.
        """
        batch_modal_dict = defaultdict(list)
        for text in batch_text:
            for modal in locatable_modalities:
                match_text = modal.locate_modality(text)
                if len(match_text) and len(match_text[0]):
                    batch_modal_dict[modal].append(match_text[0])
                else:
                    batch_modal_dict[modal].append(None)
        return batch_modal_dict

    @torch.no_grad()
    def detokenize_motion(self, codes: List[Union[torch.Tensor, None]]) -> Tuple[List[Tensor], List[int]]:
        """
        :param codes: a batch of causal lm predicted codes
        :return:
        """
        valid_length = [len(x) if x is not None else 0 for x in codes]
        motions = []
        for code, l in zip(codes, valid_length):
            if code is None or l == 0:
                motions.append(None)
            else:
                motions.append(self.motion_tokenizer.decode(code[:l][None], is_idx=True)[0])
        pred_num_frames = [self.motion_tokenizer.downsample_rate * l if l else None for l in valid_length]

        return motions, pred_num_frames

    @torch.no_grad()
    def detokenize_audio(self, codes: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[List[Tensor], List[int]]:
        valid_length = [len(x) if x is not None else 0 for x in codes]
        audios = []
        for code, l in zip(codes, valid_length):
            if code is None or l == 0:
                audios.append(None)
            else:
                audios.append(self.audio_tokenizer.decode(code[:l][None], is_idx=True)[0])

        pred_num_frames = [self.audio_tokenizer.downsample_rate * l if l else None for l in valid_length]
        audios = [audio if l else None for audio, l in zip(audios, valid_length)]

        return audios, pred_num_frames
