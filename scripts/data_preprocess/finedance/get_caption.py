import sys
from glob import glob
from os.path import join, basename, dirname

import os
from openai import OpenAI

import fire
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.json import read_json

from mmotion.utils.files_io.txt import write_txt

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-proj-rR5MCFuBCS4ewk5ysi3zT3BlbkFJHPok7yX8ThRbsqEdlaAJ",
)

prompt = ("A dancer is performing a hybrid-style dance,"
          " and I will provide you with the dance categories."
          " Please describe the dance. For example, when I input 'style: break, jazz,'"
          " you should output: 'A dancer is performing a fusion of break and jazz dance styles.'"
          " There should be no additional output. My input is: style: {}, {}")



def main(
        data_root: str = 'data/motionhub/finedance'
):
    label_root = join(data_root, 'label_json')
    label_paths = glob(join(label_root, '*.json'))
    for label_path in tqdm(label_paths):
        info = read_json(label_path)
        style_1, style_2 = info['style1'], info['style2']

        input_str = prompt.format(style_1, style_2)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": input_str}],
            model="gpt-4o-mini",
        )
        output_str = chat_completion.choices[0].message.content
        save_path = label_path.replace('label_json', 'caption').replace('.json', '.txt')
        os.makedirs(dirname(save_path), exist_ok=True)
        write_txt(save_path, output_str)


if __name__ == '__main__':
    fire.Fire(main)
