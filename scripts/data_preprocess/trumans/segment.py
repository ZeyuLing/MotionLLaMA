import sys
from glob import glob
from os.path import join, exists

import os

import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.append(os.curdir)
from mmotion.utils.files_io.pickle import save_pickle


def slice_caption_and_smplx(caption_path,
                            smplx_path,
                            caption_output_dir,
                            smplx_output_dir,
                            slice_length=300):
    basename = os.path.basename(caption_path).replace('.txt', '')
    os.makedirs(caption_output_dir, exist_ok=True)
    os.makedirs(smplx_output_dir, exist_ok=True)
    captions_df = pd.read_csv(caption_path, sep='\t', header=None, names=['start_frame', 'end_frame', 'description'])
    smplx_data = np.load(smplx_path, allow_pickle=True)

    smplx_keys = ['betas', 'global_orient', 'body_pose', 'left_hand_pose',
                  'right_hand_pose', 'jaw_pose', 'expression', 'transl']

    # count frames
    total_frames = smplx_data['transl'].shape[0]

    slices = []
    current_slice = {
        'start_frame': None,
        'end_frame': None,
        'captions': []
    }

    for idx, row in captions_df.iterrows():
        cap_start = row['start_frame']
        cap_end = row['end_frame']
        description = row['description']

        # 初始化当前切片的起始帧
        if current_slice['start_frame'] is None:
            current_slice['start_frame'] = cap_start
            current_slice['end_frame'] = cap_end
            current_slice['captions'].append(row)
            continue

        # 计算加入当前 caption 后的时间跨度
        new_end = max(current_slice['end_frame'], cap_end)
        span = new_end - current_slice['start_frame']

        if span <= slice_length:
            # 可以加入当前切片
            current_slice['end_frame'] = new_end
            current_slice['captions'].append(row)
        else:
            # 当前切片时间跨度已达到或超过限制，保存当前切片
            slices.append(current_slice.copy())
            # 开始新的切片
            current_slice['start_frame'] = cap_start
            current_slice['end_frame'] = cap_end
            current_slice['captions'] = [row]

    if current_slice['captions']:
        slices.append(current_slice.copy())

    print(f"sliced into  {len(slices)} segments")

    for idx, slice_info in enumerate(slices):
        slice_start = slice_info['start_frame']
        slice_end = slice_info['end_frame']

        slice_start = max(slice_start, 0)
        slice_end = min(slice_end, total_frames)

        sliced_smplx = {}
        for key in smplx_keys:
            if key in smplx_data:
                sliced_smplx[key] = smplx_data[key][slice_start:slice_end]

        sliced_captions = pd.DataFrame(slice_info['captions'])
        sliced_captions['start_frame'] = sliced_captions['start_frame'] - slice_start
        sliced_captions['end_frame'] = sliced_captions['end_frame'] - slice_start

        sliced_smplx_path = os.path.join(smplx_output_dir, f'{basename}_slice_{idx + 1}.npz')
        save_pickle(sliced_smplx, sliced_smplx_path, )

        sliced_captions_path = os.path.join(caption_output_dir, f'{basename}_slice_{idx + 1}.txt')
        sliced_captions.to_csv(sliced_captions_path, sep='\t', index=False, header=False)


def main(
        caption_root: str = 'data/motionhub/trumans/Actions',
        slice_caption_root: str = 'data/motionhub/trumans/slice_Actions/',
        slice_smplx_root: str = 'data/motionhub/trumans/slice_smplx_result/',
        slice_length=300
):
    for caption_file in tqdm(glob(join(caption_root, '*.txt'))):
        smplx_file = (caption_file
                      .replace('/Actions/', '/smplx_result/')
                      .replace('.txt', '_smplx_results.pkl'))
        if not exists(smplx_file):
            print(f'{smplx_file} not exists, passed')
            continue
        slice_caption_and_smplx(caption_file, smplx_file, slice_caption_root, slice_smplx_root, slice_length)


if __name__ == '__main__':
    main()
