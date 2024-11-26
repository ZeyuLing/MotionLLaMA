import re
from abc import ABC
from typing import List, Type, Union, Tuple

import torch
from torch import Tensor

from mmotion.utils.task.default_anno_keys import motion_keys, audio_keys, music_keys, caption_keys, \
    union_caption_keys, script_keys, duration_keys, genre_keys

"""
    Some special tokens representing different modal prompt sequence
"""

AUDIO_HOLDER = '<<AUDIO_HOLDER>>'
MUSIC_HOLDER = '<<MUSIC_HOLDER>>'
CAPTION_HOLDER = '<<CAPTION_HOLDER>>'  # for caption

MOTION_HOLDER = '<<MOTION_HOLDER>>'  # for motion

INTERACTOR_MOTION_HOLDER = '<<INTERACTOR_MOTION_HOLDER>>'  # for multi-person motion synthesis
INTERACTOR_CAPTION_HOLDER = '<<INTERACTOR_CAPTION_HOLDER>>'
UNION_CAPTION_HOLDER = '<<UNION_CAPTION_HOLDER>>'  # a single caption for multi-person
# for prediction task

PAST_MOTION_HOLDER = '<<PAST_MOTION_HOLDER>>'  # for prediction and inbetween
FUTURE_MOTION_HOLDER = '<<FUTURE_MOTION_HOLDER>>'  # for prediction and inbetween
MIDDLE_MOTION_HOLDER = '<<MIDDLE_MOTION_HOLDER>>'  # for inbetween

DURATION_HOLDER = '<<DURATION_HOLDER>>'
SCRIPT_HOLDER = '<<SCRIPT_HOLDER>>'
GENRE_HOLDER = '<<GENRE_HOLDER>>'


def replace_placeholders(conversation_templates_batch, holder_string_mapping):
    """ Replace placeholders in the content of conversations with actual values.
    :param conversation_templates_batch: List of conversations (each conversation is a list of messages).
    :param holder_string_mapping: List of conversations with placeholders replaced.
    :return: new_conversation_batch, replaced with actual values
    """

    # Number of conversations in the batch
    batch_size = len(conversation_templates_batch)

    # Initialize the new conversation batch
    new_conversation_batch = []

    # Iterate over each conversation in the batch
    for i in range(batch_size):
        conversation = conversation_templates_batch[i]
        new_conversation = []

        # For each message in the conversation
        for message in conversation:
            new_message = message.copy()
            content = new_message['content']

            # Replace each placeholder with its corresponding value
            for placeholder, values in holder_string_mapping.items():
                if placeholder in content:
                    # Ensure we have a value for the current conversation
                    if i < len(values):
                        replacement = str(values[i])
                        content = content.replace(placeholder, replacement)
                    else:
                        raise IndexError(f"No replacement value for placeholder '{placeholder}' in conversation {i}.")

            new_message['content'] = content
            new_conversation.append(new_message)

        new_conversation_batch.append(new_conversation)

    return new_conversation_batch


class Modality(ABC):
    """
    Abstract class to define the utilized modalities of Multi-modal Motion tasks.
    name: the name of the modality
    token_format: used in Motion LLM, defines the string format of multi-modal tokens. Only motion and audio used it.
    holder: placeholder of the modality showing in Prompt templates
    bos: special token which stands for begin of modality substring in conversation
    eos: special token which stands for end of modality substring in conversation
    data_keys: Keys to save the modality into DataSample
    load_keys: Keys to load the modality from annotation files
    """
    name = None
    token_format = '<|MODAL_{}|>'
    holder = None
    bos = None
    eos = None
    data_keys = None  # the keys to save the modality in DataSample
    load_keys = None  # the keys to load the modality from annotation files

    @classmethod
    def locatable(cls):
        if cls.bos is not None and cls.eos is not None and len(cls.bos) and len(cls.eos):
            return True
        return False

    @classmethod
    def index_to_string(cls, idx: List[int]):
        """ For motion and audio
        :param idx: index list.
        :return: for exp, <|MOTION_13|><|MOTION_150|><|MOTION_2|>
        """
        token_string = [cls.token_format.format(int(i)) for i in idx]
        return ''.join(token_string)

    @classmethod
    def string_to_index(cls, string: str, return_tensor=True) -> Union[List[int], Tensor, None]:
        if string is None:
            return None
        if cls.token_format is None:
            raise ValueError(f'Modality {type(cls)} doesnt support string to index, u can encode it with tokenizer')
        pattern = re.escape(cls.token_format).replace('\\{\\}', '(\d+)')
        ids = re.findall(pattern, string)
        ids = [int(i) for i in ids]
        if return_tensor:
            ids = torch.tensor(ids, dtype=torch.int64)
        return ids

    @classmethod
    def locate_modality(cls, text: str) -> List:
        """ Locate the substring of audio, motion, caption and script.
        :param text:
        :return:
        """
        # 4 modalities can be traced with bos and eos: motion, audio, caption and script
        pattern = re.escape(cls.bos) + '(.*?)' + re.escape(cls.eos)
        substrings = re.findall(pattern, text)
        substrings = sorted(substrings, key=len, reverse=True)
        return substrings


class Motion(Modality):
    name = 'motion'
    token_format = '<|MOTION_{}|>'
    holder = MOTION_HOLDER
    bos = '<|begin_of_motion|>'
    eos = '<|end_of_motion|>'
    data_keys = motion_keys
    load_keys = motion_keys


class PastMotion(Motion):
    name = 'past_motion'
    holder = PAST_MOTION_HOLDER
    bos = '<|begin_of_past_motion|>'
    data_keys = [f'past_{key}' for key in motion_keys]


class MiddleMotion(Motion):
    name = 'middle_motion'
    holder = MIDDLE_MOTION_HOLDER
    bos = '<|begin_of_middle_motion|>'
    data_keys = [f'middle_{key}' for key in motion_keys]


class FutureMotion(Motion):
    name = 'future_motion'
    holder = FUTURE_MOTION_HOLDER
    bos = '<|begin_of_future_motion|>'
    data_keys = [f'future_{key}' for key in motion_keys]


class InteractorMotion(Motion):
    name = 'interactor_motion'
    holder = INTERACTOR_MOTION_HOLDER
    data_keys = ['interactor_' + key for key in Motion.data_keys]
    load_keys = ['interactor_' + key for key in Motion.data_keys]
    bos = '<|begin_of_next_motion|>'


class Audio(Modality):
    name = 'audio'
    token_format = '<|AUDIO_{}|>'
    holder = AUDIO_HOLDER
    bos = '<|begin_of_audio|>'
    eos = '<|end_of_audio|>'
    data_keys = audio_keys
    load_keys = audio_keys


class Music(Audio):
    name = 'music'
    holder = MUSIC_HOLDER
    data_keys = music_keys
    load_keys = music_keys
    bos = '<|begin_of_music|>'
    eos = '<|end_of_music|>'


class Text(Modality):
    name = 'text'
    token_format = None
    bos = ''
    eos = ''


class Caption(Text):
    name = 'caption'
    holder = CAPTION_HOLDER
    bos = '<|begin_of_caption|>'
    eos = '<|end_of_caption|>'
    data_keys = caption_keys
    load_keys = caption_keys


class UnionCaption(Caption):
    name = 'union_caption'
    holder = UNION_CAPTION_HOLDER
    data_keys = union_caption_keys
    load_keys = union_caption_keys
    bos = '<|begin_of_union_caption|>'


class InteractorCaption(Caption):
    name = 'interactor_caption'
    holder = INTERACTOR_CAPTION_HOLDER
    data_keys = ['interactor_' + key for key in caption_keys]
    load_keys = ['interactor_' + key for key in caption_keys]
    bos = '<|begin_of_next_caption|>'


class Script(Text):
    name = 'script'
    holder = SCRIPT_HOLDER
    bos = '<|begin_of_script|>'
    eos = '<|end_of_script|>'
    data_keys = script_keys
    load_keys = script_keys


class Duration(Text):
    name = 'duration'
    holder = DURATION_HOLDER
    data_keys = duration_keys
    load_keys = duration_keys


class Genre(Text):
    name = 'genre'
    holder = GENRE_HOLDER
    data_keys = genre_keys
    load_keys = genre_keys


def is_modal(modal_A: Type[Modality], modal_B: Union[Type[Modality], Tuple[Type[Modality], ...]]) -> bool:
    return issubclass(modal_A, modal_B)
