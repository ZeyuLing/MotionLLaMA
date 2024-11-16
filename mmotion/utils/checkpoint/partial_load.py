"""
    The vocabulary size of motion_llama differs from in raw llama2,
    which causes size mismatch between embed_tokens and lm_head.'

    this function is developed for partially load the pretrained checkpoint to motion_llama

    for exp, the pretrained shape of embed_tokens is 32000,4096
"""
from typing import Dict, List

import torch


def partial_load(state_dict: Dict, model_state_dict: Dict, mismatch_keys: List):
    """
    :param state_dict: checkpoint state_dict
    :param model_state_dict: model state_dict
    :param mismatch_keys: keys with different shape
    :return: modified_state_dict
    """

    for key in mismatch_keys:
        weight = state_dict[key]
        model_weight = model_state_dict[key]
        min_size =cal_min_size(weight,model_weight)
        if min_size is None:
            continue
        

def cal_min_size(weight1:torch.Tensor,weight2:torch.Tensor):
    if len(weight1.shape)!=len(weight2.shape):
        #
        return None
