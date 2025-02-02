import os
import sys

import pandas as pd
import numpy as np

from pytorch3d.transforms import axis_angle_to_matrix
from tqdm import tqdm
from os.path import join as pjoin
import torch
import copy
from smplx import SMPLX

from mmotion.utils.geometry.rotation_convert import matrix_to_axis_angle
from mmotion.motion_representation import face_joint_idx

sys.path.append(os.curdir)


orig_flip_pairs = \
    ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),  # body joints
     (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30, 45), (31, 46), (32, 47),
     (33, 48), (34, 49), (35, 50), (36, 51)  # hand joints
     )

left_chain = []
right_chain = []

for pair in orig_flip_pairs:
    left_chain.append(pair[0])
    right_chain.append(pair[1])

smplx_male_model_path = 'smpl_models/smplx/SMPLX_MALE.npz'
smplx_female_model_path = 'smpl_models/smplx/SMPLX_FEMALE.npz'


def swap_left_right(data):
    pose = data[..., : 3 + 51 * 3].reshape(data.shape[0], 52, 3)

    tmp = pose[:, right_chain, :]
    pose[:, right_chain, :] = pose[:, left_chain, :]
    pose[:, left_chain, :] = tmp

    pose[:, :, 1:3] *= -1
    # change translation
    trans = copy.deepcopy(data[..., 309:312])
    trans[..., 0] *= -1

    data[..., :3 + 51 * 3] = pose.reshape(data.shape[0], -1)
    data[..., 309:312] = trans

    return data


def rotate_motion(root_global_orient):
    trans_matrix = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    motion = np.dot(root_global_orient, trans_matrix)  # exchange the y and z axis

    return motion


def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient


def transform_translation(trans):
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    return trans


def get_smplx_322(data, ex_fps):
    fps = 0

    if 'mocap_frame_rate' in data:
        fps = data['mocap_frame_rate']
        print(fps)
        down_sample = int(fps / ex_fps)

    elif 'mocap_framerate' in data:
        fps = data['mocap_framerate']
        print(fps)
        down_sample = int(fps / ex_fps)
    else:
        # down_sample = 1
        return None

    frame_number = data['trans'].shape[0]

    fId = 0  # frame id of the mocap sequence
    pose_seq = []
    if 'pose_hand' in data.keys():
        for fId in range(0, frame_number, down_sample):
            pose_root = data['root_orient'][fId:fId + 1]
            pose_body = data['pose_body'][fId:fId + 1]
            pose_hand = data['pose_hand'][fId:fId + 1]
            pose_jaw = data['pose_jaw'][fId:fId + 1]
            pose_expression = np.zeros((1, 50))
            pose_face_shape = np.zeros((1, 100))
            pose_trans = data['trans'][fId:fId + 1]
            pose_body_shape = data['betas'][:10][None]
            pose = np.concatenate(
                (pose_root, pose_body, pose_hand, pose_jaw, pose_expression, pose_face_shape, pose_trans,
                 pose_body_shape),
                axis=1)
            pose_seq.append(pose)
    else:
        for fId in range(0, frame_number, down_sample):
            # if 'root_orient' in data.keys():
            #     pose_root = data['root_orient'][fId:fId + 1]
            # else:
            pose_root = data['poses'][fId:fId + 1, :3]
            pose_body = data['poses'][fId:fId + 1, 3:66]
            pose_hand = data['poses'][fId:fId + 1, 66:]
            pose_jaw = np.zeros((1, 3))
            pose_expression = np.zeros((1, 50))
            pose_face_shape = np.zeros((1, 100))
            pose_trans = data['trans'][fId:fId + 1]
            pose_body_shape = data['betas'][:10][None, :]
            assert pose_root.shape[0] > 0 and pose_body.shape[
                0] > 0, f"{pose_root.shape}, {pose_body.shape}, {data['poses'].shape}"
            pose = np.concatenate(
                (pose_root, pose_body, pose_hand, pose_jaw, pose_expression, pose_face_shape, pose_trans,
                 pose_body_shape),
                axis=1)
            pose_seq.append(pose)
    pose_seq = np.concatenate(pose_seq, axis=0)
    return pose_seq


def process_pose(pose):
    pose_root = pose[:, :3]
    pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
    pose[:, :3] = pose_root
    pose_trans = pose[:, 309:312]
    pose_trans = transform_translation(pose_trans)
    pose[:, 309:312] = pose_trans

    return pose


def face_z_align(pose, smplx_model):
    pose = torch.from_numpy(pose).float().cuda()

    param = {
        'root_orient': pose[:, :3],  # controls the global root orientation
        'pose_body': pose[:, 3:3 + 63],  # controls the body
        'pose_hand': pose[:, 66:66 + 90],  # controls the finger articulation
        'pose_jaw': pose[:, 66 + 90:66 + 93],  # controls the yaw pose
        'face_expr': pose[:, 159:159 + 50],  # controls the face expression
        'face_shape': pose[:, 209:209 + 100],  # controls the face shape
        'trans': pose[:, 309:309 + 3],  # controls the global body position
        'betas': pose[:, 312:],  # controls the body shape. Body shape is static
    }

    batch_size = param['face_expr'].shape[0]
    zero_pose = torch.zeros((batch_size, 3)).float().cuda()
    smplx_output = smplx_model(betas=param['betas'], body_pose=param['pose_body'],
                               global_orient=param['root_orient'], pose2rot=True, jaw_pose=zero_pose,
                               leye_pose=zero_pose, reye_pose=zero_pose,
                               left_hand_pose=param['pose_hand'][:, :45], right_hand_pose=param['pose_hand'][:, 45:],
                               expression=param['face_expr'][:, :10], transl=param['trans'])

    joints = smplx_output.joints
    joints = joints[:, face_joint_idx, :]
    param['root_orient'], param['trans'] = face_z_transform(joints.cpu().numpy(), param['root_orient'], param['trans'])

    pose_list = []
    for k in ['root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'face_expr', 'face_shape', 'trans', 'betas']:
        pose_list.append(param[k])
    pose_list = torch.cat(pose_list, dim=-1).cpu().numpy()

    return pose_list


if __name__ == '__main__':

    index_path = 'scripts/data_preprocess/motionx/humanml_index.csv'
    save_dir = 'data/motionhub/motionx/motion_data/smplx_322/humanml'
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    ex_fps = 30

    bad_list = []
    for i in tqdm(range(total_amount)):
        new_name = index_file.loc[i]['new_name']
        save_path = pjoin(save_dir, new_name)
        m_save_path = pjoin(save_dir, 'M' + new_name)
        if os.path.exists(save_path) and os.path.exists(m_save_path):
            continue
        # try:
        source_path = index_file.loc[i]['source_path']
        if 'humanact12' in source_path or '18_19_Justin' in source_path or '18_19_rory' in source_path or '20_21_Justin1' in source_path or '20_21_rory1' in source_path or '22_23_justin' in source_path or '22_23_Rory' in source_path:
            continue

        source_path = source_path.replace('./pose_data', 'data/motionhub/motionx/motion_data/amass')
        if 'BMLhandball' not in source_path:
            source_path = source_path.replace('_poses.npy', '_stageii.npz')
        else:
            source_path = source_path.replace('.npy', '.npz')
        source_path = source_path.replace(' ', '_')
        data = np.load(source_path)
        gender = data['gender'].item()

        if gender == 'male':
            smplx_model = SMPLX(smplx_male_model_path, num_betas=10, use_pca=False, use_face_contour=True,
                                batch_size=1).cuda()
        elif gender == 'female':
            smplx_model = SMPLX(smplx_female_model_path, num_betas=10, use_pca=False, use_face_contour=True,
                                batch_size=1).cuda()

        start_frame = index_file.loc[i]['start_frame']
        end_frame = index_file.loc[i]['end_frame']

        pose = get_smplx_322(data, ex_fps)

        if 'humanact12' not in source_path:
            if 'Eyes_Japan_Dataset' in source_path:
                pose = pose[int(3 * ex_fps):]
            if 'MPI_HDM05' in source_path:
                pose = pose[int(3 * ex_fps):]
            if 'TotalCapture' in source_path:
                pose = pose[int(1 * ex_fps):]
            if 'MPI_Limits' in source_path:
                pose = pose[int(1 * ex_fps):]
            if 'Transitions_mocap' in source_path:
                pose = pose[int(0.5 * ex_fps):]

            pose = pose[int(start_frame * 1.5):int(end_frame * 1.5)]
        if len(pose) < 1:
            bad_list.append(new_name)
            continue
        pose = process_pose(pose)

        pose = face_z_align(pose, smplx_model)

        if pose is None:
            bad_list.append(new_name)
            continue
        np.save(pjoin(save_dir, new_name), pose)
        pose_mirror = swap_left_right(pose)
        np.save(pjoin(save_dir, 'M' + new_name), pose_mirror)
        #
        # except Exception as e:
        #     bad_count += 1
        #     print(e)
        #     print(index_file.loc[i]['source_path'])

    print(f'total {len(bad_list)} bad samples: {bad_list}')
