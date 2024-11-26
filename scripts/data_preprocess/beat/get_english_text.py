import os
import sys
import string
from glob import glob
from os.path import join, dirname

import fire
from praatio.textgrid import openTextgrid
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.txt import write_txt


def extract_sentence(textgrid_path, tier_names=None):
    """
    从TextGrid文件中提取词语，去除标点，并拼接成句子。

    :param textgrid_path: TextGrid文件的路径
    :param tier_names: 包含词语的层名称列表。如果为None，则使用所有层
    :return: 拼接后的句子
    """
    try:
        tg = openTextgrid(textgrid_path, includeEmptyIntervals=True)
    except Exception as e:
        print(f"无法读取TextGrid文件: {e}")
        return ""

    words = []

    # 如果没有指定层名称，使用所有层
    if tier_names is None:
        tiers = tg.tierNames[0:1]
    else:
        tiers = tier_names

    for tier_name in tiers:
        if tier_name not in tg.tierNames:
            print(f"层 '{tier_name}' 不存在于TextGrid文件中。")
            continue

        tier = tg._tierDict[tier_name]
        for interval in tier.entries:
            word = interval[2].strip()  # interval 格式: (start, end, mark)
            if word:  # 排除空白
                # 去除标点
                word = word.translate(str.maketrans('', '', string.punctuation))
                words.append(word)

    sentence = ' '.join(words)
    return sentence

def main(text_grid_root: str = 'data/motionhub/beat_v2.0.0/beat_english_v2.0.0/textgrid'):
    for text_grid_path in tqdm(glob(join(text_grid_root, '*.TextGrid'))):
        sentence = extract_sentence(text_grid_path)
        save_path = text_grid_path.replace('textgrid', 'text').replace('.TextGrid', '.txt')
        os.makedirs(dirname(save_path), exist_ok=True)
        write_txt(save_path, sentence)

if __name__ == "__main__":
    fire.Fire(main)
    if len(sys.argv) != 2:
        print("用法: python script.py <TextGrid文件路径>")
        sys.exit(1)

    textgrid_path = sys.argv[1]
    sentence = extract_sentence(textgrid_path)
    print(sentence)
