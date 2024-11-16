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

prompt = ("This is a video file name that captures hand movements of a person."
          " Generally, its naming convention is [object being grasped]_[action performed after grasping][unimportant suffix]."
          " Based on this file name, expand it into a complete sentence describing the person's actions."
          " For example, when I input: airplane_fly_1,"
          " a suitable output would be: A person picks up a toy airplane and flies it in their hand."
          " There should be no extra output. My input is: {}")


def main(data_root='data/motionhub/motionx/'):
    egobody_caption_root = join(data_root, 'motionx_seq_text_v1.1/GRAB/')
    caption_paths = glob(join(egobody_caption_root, '*/*.txt'))
    for caption_path in tqdm(caption_paths):
        filename = basename(caption_path).replace('.txt', '')
        input_str = prompt.format(filename)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": input_str}],
            model="gpt-4o-mini",
        )
        output_str = chat_completion.choices[0].message.content
        save_path = caption_path.replace('motionx_seq_text_v1.1', 'caption')
        os.makedirs(dirname(save_path), exist_ok=True)
        write_txt(save_path, output_str)


if __name__ == '__main__':
    fire.Fire(main)
