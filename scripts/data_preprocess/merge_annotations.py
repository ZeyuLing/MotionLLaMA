import os
import sys
from glob import glob
from os.path import join
from typing import List

import fire

sys.path.append(os.curdir)
from mmotion.utils.files_io.json import read_json, write_json


def merge_annos(file_list: List[str]):
    anno = {
        'meta_info': {
            "dataset": "motionhub",
            "version": "v1"
        },
        'data_list':
            {}
    }
    data_count = 0
    for file in file_list:
        data = read_json(file)
        data_count += len(data['data_list'])
        anno['data_list'].update(data['data_list'])
        assert data_count == len(anno['data_list']), (file, data_count, len(anno['data_list']))
    return anno


def main(data_root: str = 'data/motionhub'):
    train_annos = glob(join(data_root, '*/train.json'))
    val_annos = glob(join(data_root, '*/val.json'))
    test_annos = glob(join(data_root, '*/test.json'))

    # all_annos = merge_annos(train_annos+val_annos+test_annos)
    # write_json(join(data_root, 'all.json'), all_annos)
    # print(len(all_annos['data_list']), ' samples in all subset')

    train_annos = merge_annos(train_annos)
    write_json(join(data_root, 'train.json'), train_annos)
    print(len(train_annos['data_list']), ' samples in train subset')

    val_annos = merge_annos(val_annos)
    write_json(join(data_root, 'val.json'), val_annos)
    print(len(val_annos['data_list']), ' samples in val subset')

    test_annos = merge_annos(test_annos)
    write_json(join(data_root, 'test.json'), test_annos)
    print(len(test_annos['data_list']), ' samples in test subset')


if __name__ == '__main__':
    fire.Fire(main)
