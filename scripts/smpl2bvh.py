import numpy as np
import os
import sys
from os.path import normpath, basename, join

import fire
import torch
from os import makedirs

from einops import rearrange
from smplx import SMPLX

sys.path.append(os.curdir)
from mmotion.core.conventions.keypoints_mapping.smplx import SMPLX_KEYPOINTS_NO_CONTOUR, SMPLH_KEYPOINTS
from mmotion.registry import MODELS
from mmotion.utils.bvh.bvh import save_dict
from mmotion.utils.bvh.bvh_quaternions import from_axis_angle, to_euler
from mmotion.utils.smpl_utils.merge import merge_pose


def smpl2bvh(
        smplx_file: str,
        save_dir: str = 'vis_results',
        pretrained_smplx: str = 'checkpoints/smpl_models/smplx/',
        fps: int = 30,
        device='cpu'
):
    smplx_file = normpath(smplx_file)
    smplx_model: SMPLX = MODELS.build(
        dict(model_path=pretrained_smplx, type='SMPLX', gender='neutral', create_betas=False,
             create_transl=False, create_expression=False, create_body_pose=False,
             create_reye_pose=False, create_jaw_pose=False, create_leye_pose=False,
             create_left_hand_pose=False, create_right_hand_pose=False,
             create_global_orient=False,
             use_pca=False, use_face_contour=False, keypoint_src='smplx_no_contour')).to(device)

    names = SMPLH_KEYPOINTS
    parents = smplx_model.parents.detach().cpu().numpy()
    parents = np.concatenate([parents[:22], parents[25:55]])
    for i in range(len(parents)):
        if i >= 22 and parents[i] >= 25:
            parents[i] = parents[i] - 3
    print(parents)
    # You can define betas like this.(default betas are 0 at all.)
    rest = smplx_model(
        global_orient=torch.zeros([1, 3], dtype=torch.float32, device=device),
        body_pose=torch.zeros([1, 63], dtype=torch.float32, device=device),
        jaw_pose=torch.zeros([1, 3], dtype=torch.float32, device=device),
        leye_pose=torch.zeros([1, 3], dtype=torch.float32, device=device),
        reye_pose=torch.zeros([1, 3], dtype=torch.float32, device=device),
        left_hand_pose=torch.zeros([1, 45], dtype=torch.float32, device=device),
        right_hand_pose=torch.zeros([1, 45], dtype=torch.float32, device=device),
        betas=torch.zeros([1, 10], dtype=torch.float32, device=device),
        expression=torch.zeros([1, 10], dtype=torch.float32, device=device)

        # betas=torch.randn([1, 10], dtype=torch.float32)
    )
    rest_pose = np.concatenate([rest['joints'].detach().cpu().numpy().squeeze()[:22, :],
                                rest['joints'].detach().cpu().numpy().squeeze()[25:55, :]]
                               )

    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 1

    # Pose setting.
    smpl_dict = np.load(smplx_file, allow_pickle=True)
    smpl_dict = merge_pose(smpl_dict)
    transl = smpl_dict['transl'].numpy()
    rots = smpl_dict['poses'].numpy()
    rots = rearrange(rots, 'b (j c) -> b j c', c=3)
    rots = np.concatenate([rots[:, :22], rots[:, 25:55]], axis=1)
    # to quaternion
    rots = from_axis_angle(rots)
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    # positions[:,0] += trans * 10
    positions[:, 0] += transl
    rotations = np.degrees(to_euler(rots, order=order))
    bvh_data = {
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }
    makedirs(save_dir, exist_ok=True)
    save_path = join(save_dir, basename(smplx_file).split(".")[0] + ".bvh")
    save_dict(save_path, bvh_data)


if __name__ == "__main__":
    fire.Fire(smpl2bvh)
