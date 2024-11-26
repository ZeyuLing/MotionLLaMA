"""
    The vocabulary size of motion_llama differs from in raw llama2,
    which causes size mismatch between embed_tokens and lm_head.'

    this function is developed for partially load the pretrained checkpoint to motion_llama

    for exp, the pretrained shape of embed_tokens is 32000,4096
"""
from typing import Dict, List, Tuple

import torch
from torch.nn.init import xavier_normal_, kaiming_normal_, trunc_normal_, normal_


def mismatch_load(state_dict: Dict,
                  model_state_dict: Dict,
                  mismatch_keys: List[Tuple],
                  print_loaded_keys=False,
                  init='zero'):
    """
    :param print_loaded_keys: print the loaded keys to terminal
    :param state_dict: checkpoint state_dict.
    :param model_state_dict: model state_dict.
    :param mismatch_keys: keys with different shape
    :return: modified_state_dict
    """
    partial_load_keys = []
    for key, weight_size, model_weight_size in mismatch_keys:
        if state_dict.get(key) is None:
            # in sharded files, each file contains only part of params
            continue
        weight = state_dict[key]
        model_weight = model_state_dict[key]
        if len(weight.shape) != len(model_weight.shape):
            continue
        partial_load_weight = weight_extend(weight, model_weight, init=init)
        state_dict[key] = partial_load_weight
        partial_load_keys.append(key)
    if print_loaded_keys:
        print('partial load keys:', partial_load_keys)
    return state_dict


def weight_extend(weight: torch.Tensor, model_weight: torch.Tensor, init='zero'):
    """ extend checkpoint weight to the same shape as model_weight
    :param weight: checkpoint weight
    :param model_weight: model weight
    :return: extended checkpoint weight
    """
    extended_weight = torch.zeros_like(model_weight).to(model_weight.device)
    if init == 'xavier':
        extended_weight = xavier_normal_(extended_weight)
    elif init == 'gaussian':
        extended_weight = normal_(extended_weight)
    elif init == 'kaiming':
        extended_weight = kaiming_normal_(extended_weight)
    elif init == 'trunc':
        extended_weight = trunc_normal_(extended_weight)
    else:
        assert init=='zero', f'{init} not implemented'
    min_sizes = [min(weight.shape[i], model_weight.shape[i]) for i in range(len(weight.shape))]
    slices = tuple(slice(0, dim) for dim in min_sizes)
    extended_weight[slices] = weight[slices]
    return extended_weight


if __name__ == '__main__':
    tensor_from = torch.rand([3, 45, 56])
    tensor_to = torch.zeros([4, 44, 58])
    extended = weight_extend(tensor_from, tensor_to)
    print(extended)
