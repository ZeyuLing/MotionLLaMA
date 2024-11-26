from functools import partial

from mmengine import FUNCTIONS
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch import nn

FUNCTIONS.register_module(module=size_based_auto_wrap_policy, force=True)
FUNCTIONS.register_module(name='llama_auto_wrap_policy',
                          module=partial(transformer_auto_wrap_policy,
                                         transformer_layer_cls=[LlamaDecoderLayer]),
                          force=True)

