import sys
from glob import glob
from os.path import join, basename, dirname

import os
from openai import OpenAI

import fire
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.files_io.txt import write_txt

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-proj-rR5MCFuBCS4ewk5ysi3zT3BlbkFJHPok7yX8ThRbsqEdlaAJ",
)

prompt = ("I will provide you with basic information about an ongoing dance performance by a dancer."
          " Please help me organize this information into a complete sentence to describe the dance."
          " For example, when I input 'gender: male, dance form: cypher, genre: Break,'"
          " an appropriate output would be: 'A male dancer is performing a Break-style Cypher dance.'"
          " There should be no additional output, just the descriptive sentence."
          " My input is: gender: {}, dance form: {}, genre: {}")


def get_gender(dancer_id):
    dance_id = int(dancer_id.strip('d'))
    # female
    if dance_id in [1, 2, 3, 7, 8, 9, 12, 13, 15, 16, 17, 18, 19, 25, 26, 27]:
        return 'female'
    return 'male'


def get_genre(genre_id: str):
    genre_dict = {
        'BR': 'Break',
        'PO': 'Pop',
        'LO': 'Lock',
        'MH': 'Middle Hip-hop',
        'LH': 'LA style Hip-hop',
        'HO': 'House',
        'WA': 'Waack',
        'KR': 'Krump',
        'JS': 'Street Jazz',
        'JB': 'Ballet Jazz'
    }
    return genre_dict[genre_id[-2:]]


def get_dance_form(situation: str):
    dance_form_dict = {
        'BM': 'basic dance',
        'FM': 'advanced dance',
        'MM': 'moving camera',
        'GR': 'group dance',
        'SH': 'showcase',
        'CY': 'cypher',
        'BT': 'battle'
    }
    return dance_form_dict[situation[1:3]]


def get_dance_info(name: str):
    genre, situation, _, dancer_id = name.split('_')[:4]
    gender = get_gender(dancer_id)
    genre = get_genre(genre)
    dance_form = get_dance_form(situation)
    return gender, genre, dance_form


def main(
        data_root: str = 'data/motionhub/aist'
):
    motion_root = join(data_root, 'interhuman')
    motion_paths = glob(join(motion_root, '*.npy'))
    for motion_path in tqdm(motion_paths):
        filename = basename(motion_path).replace('.npy', '')
        gender, genre, dance_form = get_dance_info(filename)

        input_str = prompt.format(gender, dance_form, genre)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": input_str}],
            model="gpt-4o-mini",
        )
        output_str = chat_completion.choices[0].message.content
        save_path = motion_path.replace('interhuman', 'caption').replace('.npy', '.txt')
        os.makedirs(dirname(save_path), exist_ok=True)
        write_txt(save_path, output_str)


if __name__ == '__main__':
    fire.Fire(main)
