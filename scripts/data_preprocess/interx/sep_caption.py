import os
import re
import sys
from os.path import join, exists, dirname
from typing import Dict

import fire
from openai import OpenAI
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.files_io.json import read_json
from mmotion.utils.files_io.txt import read_list_txt, write_list_txt

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-proj-rR5MCFuBCS4ewk5ysi3zT3BlbkFJHPok7yX8ThRbsqEdlaAJ",
)

prompt=("The following passage describes an action involving the interaction of two people."
        " One person is the initiator of the action, and the other is the recipient."
        " You need to distinguish between the two and describe each separately."
        " For example, when I input 'The first person grabs the second person's right elbow with both hands and vigorously drags him/her to the right while stepping back,'"
        " you should output: 'Interactor: A person is grabbing the other's right elbow with both hands and vigorously drags him/her to the right while stepping back."
        " Interactee: One person, while stepping back, is forcefully grabbed by the right hand and pulled to the right by someone.'"
        " Note that the subject for both the interactor and the interactee should be 'the person' or 'a person,'"
        " not 'another person' or 'the other person.' The interactor and interactee should both exist in your reply simultaneously."
        " Do not include any additional context before or after your output. My input is:{}")



def extract_content(input_string):
    pattern = r'Interactor: (.*?)\.\s*Interactee: (.*)\.'
    match = re.search(pattern, input_string)

    if match:
        person_a_content = match.group(1).strip()
        person_b_content = match.group(2).strip()
        return person_a_content, person_b_content
    else:
        return None, None
def sep_caption(union_caption:str, cur_try=0):
    if cur_try>10:
        return None, None
    input_str = prompt.format(union_caption)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": input_str}],
        model="gpt-4o-mini",
    )
    output_str = chat_completion.choices[0].message.content
    p1_caption, p2_caption = extract_content(output_str)
    if p1_caption is None or p2_caption is None:
        return sep_caption(union_caption, cur_try=cur_try+1)
    return p1_caption, p2_caption


def main(split_file:str='data/motionhub/interx/train.json',
         data_root:str='data/motionhub/'):
    data_list:Dict = read_json(split_file)['data_list']
    for item_key, data_info in tqdm(data_list.items()):

        if data_info.get('invalid'):
            continue

        union_caption_path = join(data_root, data_info.get('union_caption_path'))
        p1_actor = data_info.get('actor')
        p1_caption_path = join(data_root, data_info.get('caption_path'))
        p2_caption_path = p1_caption_path.replace('P1.txt', 'P2.txt').replace('p1.txt', 'p2.txt')
        is_p1 = 'p1.txt' in p1_caption_path.lower()
        if not is_p1:
            continue

        p1_caption_list = []
        p2_caption_list = []
        if exists(p1_caption_path) and exists(p2_caption_path):
            continue

        if union_caption_path:
            union_caption_list = read_list_txt(union_caption_path)
            if len(union_caption_list)==0:
                assert False, union_caption_path
            for caption in union_caption_list:
                er_caption, ee_caption = sep_caption(caption)
                if er_caption is None or ee_caption is None:
                    assert False, (caption, union_caption_path)
                if p1_actor:
                    p1_caption_list.append(er_caption)
                    p2_caption_list.append(ee_caption)
                else:
                    p1_caption_list.append(ee_caption)
                    p2_caption_list.append(er_caption)

            os.makedirs(dirname(p1_caption_path), exist_ok=True)
            try:
                write_list_txt(p1_caption_list, p1_caption_path)
                write_list_txt(p2_caption_list, p2_caption_path)
            except:
                assert False, union_caption_list










if __name__=='__main__':
    fire.Fire(main)