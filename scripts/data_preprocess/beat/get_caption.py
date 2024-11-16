import sys
from glob import glob
from os.path import join, basename, dirname, exists

import os

import fire
from tqdm import tqdm
from openai import OpenAI
sys.path.append(os.curdir)
from mmotion.utils.files_io.txt import read_txt, write_txt

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-proj-rR5MCFuBCS4ewk5ysi3zT3BlbkFJHPok7yX8ThRbsqEdlaAJ",
)
prompt = ("A speaker is delivering a short speech, and I will provide you with the script."
          " Please provide a one-sentence description of this speech."
          " In your description, you need to clearly identify the following:"
          "1. The language of the speech: Chinese, Japanese, English, Spanish."
          "2. The core content of the speech, summarized in a very brief sentence."
          "3. Analyze the mood of the speech based on its content, which can be calm, excited, sad, happy, etc."
          "For example, when I input:"
          " '当我有时间的时候,我喜欢在网上冲浪,密切关注新的时尚事件,如纽约时装周,巴黎时装周,伦敦时装周和米兰时装周'"
          " an appropriate description would be:"
          " 'A Chinese speaker is somewhat happily discussing their interest in fashion events.'"
          "There should be no additional output, just your description. My input is: {}")


def main(
        data_root: str = 'data/motionhub/beat_v2.0.0'
):
    content_root = join(data_root, '*/text')
    content_paths = glob(join(content_root, '*.txt'))
    for content_path in tqdm(content_paths):
        save_path = content_path.replace('text', 'caption')
        if exists(save_path):
            continue
        content = read_txt(content_path)
        input_str = prompt.format(content)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": input_str}],
            model="gpt-4o-mini",
        )
        output_str = chat_completion.choices[0].message.content

        os.makedirs(dirname(save_path), exist_ok=True)
        write_txt(save_path, output_str)


if __name__ == '__main__':
    fire.Fire(main)
