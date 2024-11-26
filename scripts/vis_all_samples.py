import numpy as np
import os
import sys
from os.path import dirname, exists, join, relpath

import fire
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.core.visualization.visualize_smpl import render_smpl
from mmotion.registry import MODELS
from mmotion.utils.files_io.json import read_json, write_json
from mmotion.utils.smpl_utils.merge import merge_pose, merge_persons

smplx_model = MODELS.build(dict(model_path='smpl_models/smplx/', type='SMPLX', gender='neutral', create_betas=False,
                                create_transl=False, create_expression=False, create_body_pose=False,
                                create_reye_pose=False, create_jaw_pose=False, create_leye_pose=False,
                                create_left_hand_pose=False, create_right_hand_pose=False,
                                create_global_orient=False,
                                use_pca=False, use_face_contour=False, keypoint_src='smplx_no_contour')).cuda()


def main(anno: str = 'data/motionhub/all.json'):
    data_root = dirname(anno)
    anno_data = read_json(anno)
    for key, sample in tqdm(anno_data['data_list'].items()):
        smplx_path = join(data_root, sample['smplx_path'])
        if not ('interhuman' in smplx_path):
            # no calibration needed
            continue
        vis_path = smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        os.makedirs(dirname(vis_path), exist_ok=True)
        if not exists(vis_path):
            smplx_dict_list = [merge_pose(np.load(smplx_path, allow_pickle=True))]
            palette = ['white']
            if 'interactor_key' in sample.keys():
                interactor_sample = join(data_root, anno_data['data_list'][sample['interactor_key']]['smplx_path'])
                interactor_sample = np.load(interactor_sample, allow_pickle=True)
                smplx_dict_list.append(merge_pose(interactor_sample))
                # yellow for P1, white for P2
                palette = ['Yellow', 'white']
            smplx_dict = merge_persons(smplx_dict_list)
            for k, value in smplx_dict.items():
                smplx_dict[k] = value.to('cuda')

            render_smpl(
                **smplx_dict,
                output_path=vis_path,
                body_model=smplx_model,
                resolution=(1024, 1024),
                R=np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ]),
                palette=palette,
                overwrite=True
            )

            anno_data['data_list'][key]['vis_path'] = relpath(vis_path, data_root)
            write_json(anno, anno_data)


if __name__ == '__main__':
    fire.Fire(main)
