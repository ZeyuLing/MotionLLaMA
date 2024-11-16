import sys
from glob import glob
from os.path import join, basename, exists
from openai import OpenAI
import fire
import os
import pandas as pd
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.files_io.txt import write_txt

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-None-3KzBXayNvxW40M9eCY7qT3BlbkFJdliPQVMF2MUoxZQZtLZf",
)

prompt = ("The following are several actions performed continuously by a person."
          " I will place these actions in a list, and you should compose them into a complete sentence."
          " For example, when I input: ['Put down the phone', 'Stand up'],"
          " you should output 'A person puts down his phone then stands up.'"
          " There should be no additional output, and do not wrap the output in print."
          " My input is : {}")

def main(raw_caption_root: str = 'data/motionhub/trumans/slice_Actions',
         save_caption_root: str = 'data/motionhub/trumans/caption'):
    os.makedirs(save_caption_root, exist_ok=True)
    for raw_path in tqdm(glob(join(raw_caption_root, '*.txt'))):
        save_path = join(save_caption_root, basename(raw_path))
        if exists(save_path):
            continue
        captions_df = pd.read_csv(raw_path, sep='\t', header=None,
                                  names=['start_frame', 'end_frame', 'description'])

        descriptions = list(captions_df['description'])
        input_str = prompt.format(descriptions)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": input_str}],
            model="gpt-4o-mini",
        )
        output_str = chat_completion.choices[0].message.content
        write_txt(save_path, output_str)


if __name__ == '__main__':
    fire.Fire(main)
