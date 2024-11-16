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

prompt=("The following are separate descriptions of each individual's specific actions in a partnered movement. You are tasked with merging these two descriptions into a single description of the overall partnered movement. For example, when I provide the following: "
        "Person A: A person walks up to another person and gives him/her a hug around the shoulders, while patting softly on his/her back with his/her right hand. "
        "Person B: One person is being hugged around the shoulders by someone and has his/her arms around the waist of that person."
        " You should combine the actions as follows: "
        "first person walks up to the second person and gives him/her a hug around the shoulders, "
        "while the second person puts his/her arms around the first person's waist. "
        "Then the first person pats the second person softly on his/her back with his/her right hand."
        "Your merging should be as concise as possible. For instance, when I provide: "
        "Person A: A person is dancing. Person B: A person is dancing."
        "You should merge the actions as:"
        "\"Two persons are dancing.\""
        " Do not include any additional context before or after your output. My input is: Person A: {}, Person B: {}")

def merge_caption(a_caption:str, b_caption):
    input_str = prompt.format(a_caption, b_caption)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": input_str}],
        model="gpt-4o-mini",
    )
    output_str = chat_completion.choices[0].message.content
    return output_str


def main(split_file:str='data/motionhub/hi4d/train.json',
         data_root:str='data/motionhub/'):
    data_list:Dict = read_json(split_file)['data_list']
    for item_key, data_info in tqdm(data_list.items()):

        interactor_key = data_info.get('interactor_key')
        interactor_info = data_list.get(interactor_key)
        if data_info.get('invalid') or interactor_info.get('invalid'):
            continue

        union_caption_path = (join(data_root, data_info.get('caption_path')).replace('caption', 'union_caption')
                              .replace('/P1.txt', '.txt').replace('/P2.txt', '.txt'))
        if os.path.exists(union_caption_path):
            continue
        p1_caption_path = join(data_root, data_info.get('caption_path'))
        p2_caption_path = join(data_root, interactor_info.get('caption_path'))
        if not exists(p1_caption_path) or not exists(p2_caption_path):
            continue
        p1_caption_list = read_list_txt(p1_caption_path)
        p2_caption_list = read_list_txt(p2_caption_path)

        os.makedirs(dirname(union_caption_path), exist_ok=True)

        union_caption_list = []
        for cap_1, cap_2 in zip(p1_caption_list, p2_caption_list):
            union_caption = merge_caption(cap_1, cap_2)
            union_caption_list.append(union_caption)
            print(union_caption)

        os.makedirs(dirname(union_caption_path), exist_ok=True)
        write_list_txt(union_caption_list, union_caption_path)










if __name__=='__main__':
    fire.Fire(main)