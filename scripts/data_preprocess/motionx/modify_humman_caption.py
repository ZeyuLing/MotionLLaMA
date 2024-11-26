# some caption of humman have errors, modify them
import os
import sys
from glob import glob
from os.path import join

import fire
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.files_io.txt import read_txt, write_txt


def main(data_root='data/motionhub/motionx/'):

    humman_caption_root = join(data_root, 'motionx_seq_text_v1.1/humman')
    caption_paths = glob(join(humman_caption_root, '*/*.txt'))
    for caption_path in tqdm(caption_paths):
        caption = read_txt(caption_path)
        if caption.startswith('#'):
            caption = caption.strip('##Input: ##Ouput: ')
            caption = caption.strip('##Ouput: ')
            caption = caption.strip('##Input: ')
            write_txt(caption_path, caption)


if __name__ =='__main__':
    fire.Fire(main)