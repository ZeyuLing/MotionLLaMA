"""
    generate multi-modal sequence to train motion_llm.

    4 tasks can be trained with motionx:
     motion in-between,  motion prediction, random sample motion
"""
import argparse
import glob
import os
import random
import sys
from collections import defaultdict
from typing import Union, List

import numpy as np
import torch
from mmengine import MODELS, Config
from tqdm import tqdm

from mmotion.utils import prompt_str2dict_finetune

sys.path.append(os.curdir)
from models.tokenizers import MultiModalTokenizer
from mmotion.utils.task.holders import MOTION_HOLDER, PAST_MOTION_HOLDER, FUTURE_MOTION_HOLDER, \
    INBETWEEN_MOTION_HOLDER
from mmotion.utils.task.prompt import make_prompt

from mmotion.utils import normalize
from mmotion.utils import load_pickle, save_pickle
from mmotion.utils.task.prompt import sample

from mmotion.utils.task.prompt.prompt_template.single_text_motion.n2m import COMPLETION_LIB as N2M_LIB

from mmotion.utils.task.prompt.prompt_template.completion.pred import COMPLETION_LIB as PRED_LIB
from mmotion.utils.task.prompt.prompt_template.completion.inbetween import COMPLETION_LIB as INBETWEEN_LIB


def process_item(vec_path: str,
                 save_path: str,
                 mean: Union[str, np.ndarray],
                 std: Union[str, np.ndarray],
                 override: bool = False,
                 pred: int = 5,  # 20 w
                 inbetween: int = 5,  # 20w
                 n2m: int = 3,  # 15w
                 ):
    """ 4w motions
    :param n2m: randomly sample motions
    :param inbetween: produce prompts for inbetween task
    :param pred: produce prompts for prediction task
    :param vec_path: motion vector path
    :param save_path: saving path of pkl.
    :param override: When save_path already exists, if True, the existed one will be covered,
                     otherwise, it will be updated.
    :return: None
    """
    result = defaultdict(list)
    if os.path.exists(save_path) and not override:
        result = load_pickle(save_path)

        # loading
    motion_vec = np.load(vec_path)
    motion_vec = normalize(motion_vec, mean, std).to(torch.float32)

    result['inbetween'] += prompt_part_motion(INBETWEEN_LIB, motion_vec, inbetween, is_pred=False, )
    result['pred'] += prompt_part_motion(PRED_LIB, motion_vec, pred, is_pred=True)

    result['n2m'] += prompt_motion(N2M_LIB, motion_vec, n2m)

    result['mean'] = mean
    result['std'] = std

    save_pickle(result, save_path)


def prompt_part_motion(
        PROMPT_LIB: List[str],
        motion: Union[np.ndarray, torch.Tensor],
        num_prompts: int = 1,
        is_pred: bool = True,
        min_len: int = 20) -> List[str]:
    """ For motion prediction and inbetween tasks
        As defined in motiongpt, predction use the first 20% tokens as past
        inbetween use the first and last 25% tokens
    :param min_len: min length for motion vqvae
    :param PROMPT_LIB: prompt lib for prediction or inbetween tasks
    :param motion: motion from motionx
    :param num_prompts: num prompts for each motion.
            Different segment will be implemented for each prompt.
    :param is_pred: true for prediction, False for inbetween
    :return:
    """
    results = []
    if num_prompts == 0:
        return results
    motion = motion.cuda()

    if len(motion) < min_len * 2 and is_pred:
        return []
    if len(motion) < min_len * 3 and not is_pred:
        return []
    prompts = sample(PROMPT_LIB, num_prompts)
    for prompt in prompts:
        if is_pred:
            mid_frame = random.randint(min_len, len(motion) - min_len)
            past_str = tokenizer.mmseq_to_str([motion[:mid_frame]], modal_list=['motion'])
            future_str = tokenizer.mmseq_to_str([motion[mid_frame:]], modal_list=['motion'])
            holder_replace_dict = {
                PAST_MOTION_HOLDER: past_str,
                FUTURE_MOTION_HOLDER: future_str
            }
        else:
            def generate_random_start_end(T, min_len):
                min_start = min_len
                max_start = T - min_len * 2
                if min_start > max_start:
                    raise ValueError("无法生成满足条件的随机数start和end。")
                start = random.randint(min_start, max_start)

                min_end = start + min_len
                max_end = T - min_len

                end = random.randint(min_end, max_end)

                return start, end

            start, end = generate_random_start_end(len(motion), min_len)
            if end - start < min_len:
                continue
            past_str = tokenizer.mmseq_to_str([motion[:start]], modal_list=['motion'])
            inbetween_str = tokenizer.mmseq_to_str([motion[start:end]], modal_list=['motion'])
            future_str = tokenizer.mmseq_to_str([motion[end:]], modal_list=['motion'])
            holder_replace_dict = {
                PAST_MOTION_HOLDER: past_str,
                INBETWEEN_MOTION_HOLDER: inbetween_str,
                FUTURE_MOTION_HOLDER: future_str
            }

        prompt = make_prompt(prompt, holder_replace_dict)
        try:
            result = prompt_str2dict_finetune(tokenizer, prompt)
            results.append(result)

        except Exception as e:
            print(e)
            break
    return results


def prompt_motion(PROMPT_LIB: List[str], motion: Union[np.ndarray, torch.Tensor], num_prompts=1):
    """ For motion only involved tasks.
    :param PROMPT_LIB: prompt lib for completion
    :param motion: a single motion from motionx
    :param num_prompts: how many prompts made of a single caption
    :return: each result is a dict. we return num_prompts results
    """
    results = []
    if num_prompts == 0:
        return []
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    motion = motion.cuda()
    m_str = tokenizer.mmseq_to_str([motion], modal_list=['motion'])

    prompts = sample(PROMPT_LIB, num_prompts)
    holder_replace_dict = {
        MOTION_HOLDER: m_str
    }
    for prompt in prompts:
        prompt = make_prompt(prompt, holder_replace_dict)
        try:
            result = prompt_str2dict_finetune(tokenizer, prompt)
            results.append(result)

        except Exception as e:
            print(e)
            print(motion.shape)
            break
        results.append(result)
    return results


if __name__ == '__main__':
    args = argparse.ArgumentParser('make prompt for motionx')
    args.add_argument('--data_root', default='data/motionx', help='root of motionx dataset')
    args.add_argument('--model_cfg', default='configs/llama2_7b_completion.py',
                      help='config file which includes tokenizer config')
    args.add_argument('--override', action='store_true', help='override existed prompts')
    args.add_argument('--skip_exist', action='store_true', help='skip existed files')
    args.add_argument('--mean', default='data/motionx/Mean.npy')
    args.add_argument('--std', default='data/motionx/Std.npy')
    args = args.parse_args()
    # make save dirs
    save_root = os.path.join(args.data_root, 'sequence', 'chat')
    os.makedirs(save_root, exist_ok=True)
    tokenizer_cfg = Config.fromfile(args.model_cfg)['tokenizer']
    tokenizer: MultiModalTokenizer = MODELS.build(tokenizer_cfg).cuda()

    for vec_path in tqdm(glob.glob(os.path.join(args.data_root, 'vecs_joints_52', '*.npy'))):
        index = os.path.basename(vec_path).split('.')[0]
        save_path = os.path.join(save_root, index + '.pkl')
        if args.skip_exist and os.path.exists(save_path):
            continue
        process_item(vec_path, save_path, override=args.override, mean=args.mean, std=args.std)
