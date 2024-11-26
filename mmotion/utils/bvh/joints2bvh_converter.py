import torch
from tqdm import tqdm

from mmotion.core.conventions.keypoints_mapping.smplx import SMPLH_KEYPOINTS
from mmotion.utils.bvh.bvh import load, save_anime
from mmotion.utils.bvh.bvh_animation import positions_global, Animation
from mmotion.utils.bvh.bvh_ik import InverseKinematics, BasicInverseKinematics
from mmotion.utils.bvh.bvh_quaternions import Quaternions
from mmotion.utils.bvh.bvh_skel_mapping import get_reorder, get_reorder_inv, get_end_points, get_parents
from mmotion.utils.bvh.remove_fs import remove_fs


class Joint2BVHConvertor:
    def __init__(self, template_file='data/motinohub/bvh_templates/smplh.bvh', data_source='smplh'):
        self.template: Animation = load(template_file, need_quater=True)

        self.re_order = get_reorder(data_source)

        self.re_order_inv = get_reorder_inv(data_source)
        self.end_points = get_end_points(data_source)

        self.template_offset = self.template.offsets.copy()
        self.parents = get_parents(data_source)

    def convert(self, positions, filename, iterations=10, foot_ik=True, fps=30):
        '''
        Convert the SMPL joint positions to Mocap BVH
        :param positions: (N, 22, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :param fps: fps
        :return:
        '''
        positions = positions[:, self.re_order]
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(positions.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=-0)
        new_anim.positions[:, 0] = positions[:, 0]

        if foot_ik:
            positions = remove_fs(positions, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=5,
                                  force_on_floor=True)
        ik_solver = BasicInverseKinematics(new_anim, positions, iterations=iterations, silent=True)
        new_anim = ik_solver()

        # BVH.save(filename, new_anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        glb = positions_global(new_anim)[:, self.re_order_inv]
        if filename is not None:
            save_anime(filename, new_anim, names=new_anim.names, frametime=1 / fps, order='zyx', quater=True)
        return new_anim, glb

    def convert_sgd(self, positions, filename, iterations=100, foot_ik=True, fps=30):
        '''
        Convert the SMPL joint positions to Mocap BVH

        :param positions: (N, 22, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        '''

        ## Positional Foot locking ##
        glb = positions[:, self.re_order]

        if foot_ik:
            glb = remove_fs(glb, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=2,
                            force_on_floor=True)

        ## Fit BVH ##
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(glb.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(glb.shape[0], axis=-0)
        new_anim.positions[:, 0] = glb[:, 0]
        anim = new_anim.copy()

        rot = torch.tensor(anim.rotations.qs, dtype=torch.float)
        pos = torch.tensor(anim.positions[:, 0, :], dtype=torch.float)
        offset = torch.tensor(anim.offsets, dtype=torch.float)

        glb = torch.tensor(glb, dtype=torch.float)
        ik_solver = InverseKinematics(rot, pos, offset, anim.parents, glb)
        print('Fixing foot contact using IK...')
        for i in tqdm(range(iterations)):
            mse = ik_solver.step()
            # print(i, mse)

        rotations = ik_solver.rotations.detach().cpu()
        norm = torch.norm(rotations, dim=-1, keepdim=True)
        rotations /= norm

        anim.rotations = Quaternions(rotations.numpy())
        anim.rotations[:, self.end_points] = Quaternions.id((anim.rotations.shape[0], len(self.end_points)))
        anim.positions[:, 0, :] = ik_solver.position.detach().cpu().numpy()
        if filename is not None:
            save_anime(filename, anim, names=new_anim.names, frametime=1 / fps, order='zyx', quater=True)
        # BVH.save(filename[:-3] + 'bvh', anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        glb = positions_global(anim)[:, self.re_order_inv]
        return anim, glb
