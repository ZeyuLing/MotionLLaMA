from typing import List, Dict, Union, Type, Tuple

import torch
from tokenizers import AddedToken
from torch import Tensor
from transformers import T5ForConditionalGeneration, BatchEncoding, PreTrainedTokenizerFast

from mmotion.models.generators.motion_llm.motion_causal_lm import MotionCausalLM
from mmotion.registry import MODELS
from mmotion.structures import DataSample
from mmotion.utils.task import name_modal_mapping, Modality, all_modalities
from mmotion.utils.task.prompt.chat_template import DEFAULT_CHAT_TEMPLATE, DEFAULT_RESPONSE_TEMPLATE
from mmotion.utils.task.prompt.prompt_template import get_system_user, get_assistant
from mmotion.utils.typing import SampleList


@MODELS.register_module()
class MotionGPT(MotionCausalLM):
    def build_text_tokenizer(self, pretrained_lm: str, text_tokenizer_cfg=None):
        """ Set text tokenizer, and determine the padding strategy
        :param pretrained_lm: path of pretrained llm
        :param text_tokenizer_cfg: tokenizer configuration
        :return:
        """
        self.text_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_lm)
        default_text_tokenizer_config = dict(
            padding_side='right',
            pad_token=self.text_tokenizer.eos_token,
            pad_token_id=self.text_tokenizer.eos_token_id
        )
        if text_tokenizer_cfg is not None:
            default_text_tokenizer_config.update(text_tokenizer_cfg)
        for key, value in default_text_tokenizer_config.items():
            setattr(self.text_tokenizer, key, value)

    def build_lm(self, pretrained_lm):
        # AutoModel Cannot choose the right class for T5
        self.lm: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(pretrained_lm,
                                                                                         attn_implementation='eager',
                                                                                         device_map='cuda')

    def prepare_input_train(self, conversations: List[List[Dict]], data_samples: Union[DataSample, SampleList]):
        batch_size = len(conversations)
        # input Text, including assistant bos
        batch_input = self.text_tokenizer.apply_chat_template([get_system_user(m) for m in conversations],
                                                              chat_template=DEFAULT_CHAT_TEMPLATE,
                                                              add_generation_prompt=True,
                                                              tokenize=False)

        # ground truth output Text, doesn't include assistant bos, include assistant eos
        # During inference period, no ground truth is provided.
        batch_output = self.text_tokenizer.apply_chat_template([get_assistant(m) for m in conversations],
                                                               chat_template=DEFAULT_CHAT_TEMPLATE,
                                                               tokenize=False)
        batch_output = [a.replace(DEFAULT_RESPONSE_TEMPLATE, '') for a in batch_output]
        encoding = self.text_tokenizer(
            batch_input,
            padding='longest',
            return_tensors="pt"
        )

        with self.text_tokenizer.as_target_tokenizer():
            target_encoding = self.text_tokenizer(
                batch_output,
                padding='longest',
                return_tensors="pt"
            )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = target_encoding['input_ids']
        labels[labels == self.text_tokenizer.pad_token_id] = -100
        lm_input = BatchEncoding({
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int64)
        })
        lm_input = lm_input.to('cuda')
        return lm_input, data_samples

    def prepare_input_predict(self, conversations: List[List[Dict]], data_samples: DataSample) \
            -> Tuple[BatchEncoding, DataSample]:
        batch_inputs = self.text_tokenizer.apply_chat_template([get_system_user(m) for m in conversations],
                                                               chat_template=DEFAULT_CHAT_TEMPLATE,
                                                               add_generation_prompt=True,
                                                               tokenize=False)
        data_samples.input = batch_inputs
        batch_inputs = self.text_tokenizer(batch_inputs, add_special_tokens=False,
                                           padding='longest', return_tensors='pt',
                                           return_attention_mask=True)

        has_gt = any([seq['role'] == 'assistant' for seq in conversations[0]])
        if has_gt:
            output = self.text_tokenizer.apply_chat_template([get_assistant(m) for m in conversations],
                                                             chat_template=DEFAULT_CHAT_TEMPLATE,
                                                             tokenize=False)
            output = [a.replace(DEFAULT_RESPONSE_TEMPLATE, '') for a in output]
            output_ids = self.text_tokenizer(output, add_special_tokens=False,
                                             padding=False)['input_ids']
            data_samples.output = output
            data_samples.output_ids = output_ids
        # batch_inputs['length'] = batch_inputs['input_ids'].shape[1]
        for key, value in batch_inputs.items():
            if isinstance(value, Tensor):
                batch_inputs[key] = value.cuda()
        return batch_inputs, data_samples

    def extend_tokenizer(self, mm_tokenizer_cfg: Dict):
        self.ori_vocab_size = len(self.text_tokenizer)
        bos_token = AddedToken('<|begin_of_text|>',
                               lstrip=False, rstrip=False, single_word=False, special=True)
        boh_token = AddedToken('<|start_header_id|>',
                               lstrip=False, rstrip=False, single_word=False, special=True)
        eoh_token = AddedToken('<|end_header_id|>',
                               lstrip=False, rstrip=False, single_word=False, special=True)
        eot_token = AddedToken('<|eot_id|>',
                               lstrip=False, rstrip=False, single_word=False, special=True)
        self.text_tokenizer.add_tokens([bos_token, boh_token, eoh_token, eot_token])
        self.text_tokenizer.bos_token = bos_token

        self.modal_need_tokenizer = []

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
                if modality.eos is not None:
                    eos = AddedToken(modality.eos, lstrip=False, rstrip=False, single_word=False, special=True)
                    self.text_tokenizer.add_tokens(eos, special_tokens=True)
        self.lm.resize_token_embeddings(len(self.text_tokenizer.get_vocab()))
        self.lm.get_input_embeddings().requires_grad_(True)
        self.lm.get_output_embeddings().requires_grad_(True)
        self.freeze_embedding_layers()

    def forward_predict(self, inputs: Dict, data_samples: DataSample):
        conversations: List[List[Dict]] = data_samples.get('conversation')
        conversations, data_samples = self.fill_conversation(conversations, inputs, data_samples)
        lm_inputs, data_samples = self.prepare_input_predict(conversations, data_samples)
        lm_inputs.pop('token_type_ids', None)
        lm_inputs.pop('length', None)
        pred_output_ids = self.lm.generate(
            **lm_inputs,
            generation_config=self.generation_config,
            synced_gpus=True
        )
        pred_output = self.text_tokenizer.batch_decode(pred_output_ids)
        data_samples.set_field(pred_output, 'pred_text')

        data_samples: DataSample = self.multimodal_detokenize(pred_output, data_samples)

        data_samples: SampleList = self.post_process(inputs, data_samples)
        return data_samples
