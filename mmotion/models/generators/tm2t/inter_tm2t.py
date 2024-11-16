from typing import Dict, Union, List

import os
import torch



from transformers import BatchEncoding
from mmotion.models.generators.tm2t.tm2t import TM2T
from mmotion.models.generators.tm2t.tm2t import create_attention_mask_from_lengths
from mmotion.registry import MODELS
from mmotion.structures import DataSample


@MODELS.register_module()
class InterTM2T(TM2T):

    def prepare_input_m2t(self, inputs, data_samples):
        motion_a = inputs['motion']
        motion_b = inputs['interactor_motion']
        num_frames = data_samples.get('num_frames')
        batch_size = len(motion_a)
        input_ids_a = self.motion_tokenizer.encode(motion_a)
        input_ids_a = torch.cat([input_ids_a, torch.zeros([batch_size, 1]).to(input_ids_a)], dim=-1)

        num_valid_tokens = [(nf // self.motion_tokenizer.downsample_rate) for nf in num_frames]
        rows = torch.arange(batch_size)
        input_ids_a[rows, num_valid_tokens] = self.motion_eos_token_id

        input_ids_b = self.motion_tokenizer.encode(motion_b)
        input_ids_b = torch.cat([input_ids_b, torch.zeros([batch_size, 1]).to(input_ids_b)], dim=-1)
        input_ids_b[rows, num_valid_tokens] = self.motion_eos_token_id
        num_valid_tokens = [(item + 1) for item in num_valid_tokens]

        input_ids = torch.cat([input_ids_a, input_ids_b], dim=-1)

        # create motion attention mask
        attention_mask = create_attention_mask_from_lengths(input_ids_a, num_valid_tokens)
        attention_mask = torch.cat([attention_mask, attention_mask], dim=-1)

        labels = None
        caption = data_samples.get('union_caption')
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
        raise NotImplementedError()
    @torch.no_grad()
    def forward_predict(self, inputs, data_samples):
        batch_input = self.prepare_input(inputs, data_samples)
        output_ids = self.transformer.generate(**batch_input,
                                               generation_config=self.generation_config,
                                               synced_gpus=True)
        if self.is_t2m:
            raise NotImplementedError()
        else:
            output_text = self.text_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            data_samples.set_field(output_text, 'pred_union_caption')

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
