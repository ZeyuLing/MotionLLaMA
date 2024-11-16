# some caption of humman have errors, modify them
import os
import sys
from glob import glob
from os.path import join, dirname, basename
from openai import OpenAI
import fire
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.txt import read_txt, write_txt

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-proj-rR5MCFuBCS4ewk5ysi3zT3BlbkFJHPok7yX8ThRbsqEdlaAJ",
)

prompt = ("This is a video file name that captures movements of a person."
          " Generally, its naming convention is [happening place]_[action performed by the person][unimportant suffix]."
          " Based on this file name, expand it into a complete sentence describing the person's actions."
          " For example, when I input: courtyard_basketball_00, a suitable output would be:"
          " A person is playing basketball in the courtyard. "
          "There should be no extra output. My input is: {}")


def main(data_root='data/motionhub/virtualhumans'):
    data_root = join(data_root, 'interhuman/')
    motion_paths = glob(join(data_root, '*.npy'))
    for motion_path in tqdm(motion_paths):
        filename = basename(motion_path).replace('.npy', '')
        input_str = prompt.format(filename)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": input_str}],
            model="gpt-4o-mini",
        )
        output_str = chat_completion.choices[0].message.content
        save_path = motion_path.replace('interhuman', 'caption')
        os.makedirs(dirname(save_path), exist_ok=True)
        write_txt(save_path, output_str)


if __name__ == '__main__':
    fire.Fire(main)
