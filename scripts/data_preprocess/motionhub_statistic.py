import glob
import os
import sys
from collections import defaultdict
from os.path import join, exists

import fire
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.json import read_json
from mmotion.utils.files_io.txt import read_list_txt


def main(data_root: str = 'data/motionhub/'):
    statistic = defaultdict(lambda: 0)
    annotation_files = glob.glob(join(data_root, '*.json'))
    for annotation_file in annotation_files:
        data_list = read_json(annotation_file)['data_list']
        for key, value in tqdm(data_list.items()):
            duration = value['duration']
            statistic['num_clips'] += 1
            statistic['duration'] += duration
            caption_path = value.get('caption_path')
            union_caption_path = value.get('union_caption_path')
            caption_path = join(data_root, caption_path)

            if caption_path is not None and exists(caption_path):
                caption_list = read_list_txt(caption_path)
                statistic['num_captions'] += len(caption_list)

            if union_caption_path is not None:
                union_caption_path = join(data_root, union_caption_path)
                if exists(union_caption_path):
                    union_caption_list = read_list_txt(union_caption_path)
                    statistic['num_union_captions'] += len(union_caption_list)

            music_path = value.get('music_path')
            if music_path is not None:
                statistic['num_music'] += 1
                statistic['music_duration'] += duration

            audio_path = value.get('audio_path')
            if audio_path is not None:
                statistic['num_audio'] += 1
                statistic['audio_duration'] += duration

            interactor_key = value.get('interactor_key')
            if interactor_key is not None:
                statistic['interaction_motion_duration'] += duration
                statistic['num_interaction_motion'] += 1
    statistic['interaction_motion_duration'] /= 2
    statistic['num_interaction_motion'] /= 2
    statistic['num_union_captions'] /= 2
    print(statistic)


if __name__ == '__main__':
    fire.Fire(main)
