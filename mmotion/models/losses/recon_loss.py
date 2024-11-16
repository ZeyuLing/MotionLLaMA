from functools import partial
from typing import Dict, Union

import torch
from einops import rearrange

from torch import nn

from mmotion.registry import MODELS
import torch.nn.functional as F

from mmotion.structures import DataSample
from mmotion.utils.mask_utils import get_mask_rotation, batch_get_mask_rotation, batch_get_mask_joints


@MODELS.register_module(force=True)
class BaseLoss(nn.Module):
    def __init__(self, recons_type='l2', nb_joints: int = 52):
        super().__init__()

        self.nb_joints = nb_joints
        if recons_type == 'l1':
            self.recons_fn = partial(F.l1_loss, reduce='mean')
        elif recons_type == 'l2':
            self.recons_fn = partial(F.mse_loss, reduce='mean')
        elif recons_type == 'l1_smooth':
            self.recons_fn = partial(F.smooth_l1_loss, reduce='mean')
        else:
            raise NotImplementedError('{} is not implemented'.format(recons_type))

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        gt = inputs['motion']
        recons = outputs['pred_motion']
        loss_dict = {
            'recon_loss': self.recons_fn(gt, recons),
            'commit_loss': 0.02 * outputs['commit_loss']
        }
        return loss_dict


@MODELS.register_module(force=True)
class Hm3dWholeBodyLoss(BaseLoss):
    def __init__(self,
                 recons_type='l2',
                 nb_joints=52,
                 pos_body=1.5,
                 pos_hand=1.5,
                 root=1.,
                 feet=1.,
                 rot_body=1.,
                 rot_hand=1.,
                 vel_body=1.,
                 vel_hand=1.,
                 commit=0.02):
        """
        :param recons_type: reconstruction loss type
        :param nb_joints: number of joints
        :param recons: weight of reconstruction loss
        :param vel: weight of velocity loss
        :param commit: weight of commit loss
        """
        super().__init__(recons_type, nb_joints)

        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.pos_body = pos_body
        self.pos_hand = pos_hand
        self.rot_body = rot_body
        self.rot_hand = rot_hand
        self.root = root
        self.feet = feet
        self.vel_body = vel_body
        self.vel_hand = vel_hand
        self.commit = commit

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        gt = inputs['motion']
        recons = outputs['pred_motion']
        loss_dict = {
            'root_loss': self.root * self.recons_fn(gt[..., :4], recons[..., :4]),
            'pos_body_loss': self.pos_body * self.recons_fn(gt[..., 4: 67], recons[..., 4: 67]),
            'pos_hand_loss': self.pos_hand * self.recons_fn(gt[..., 67: 157], recons[..., 67:157]),
            'vel_body_loss': self.vel_body * self.recons_fn(gt[..., 157: 157 + 22 * 3], recons[..., 157: 157 + 22 * 3]),
            'vel_hand_loss': self.vel_hand * self.recons_fn(gt[..., 157 + 22 * 3: 157 + 52 * 3],
                                                            recons[..., 157 + 22 * 3: 157 + 52 * 3]),
            'rot_body_loss': self.rot_body * self.recons_fn(gt[..., 157 + 52 * 3: 157 + 52 * 3 + 21 * 6],
                                                            recons[..., 157 + 52 * 3: 157 + 52 * 3 + 21 * 6]),
            'rot_hand_loss': self.rot_hand * self.recons_fn(gt[..., 157 + 52 * 3 + 21 * 6:-4],
                                                            recons[..., 157 + 52 * 3 + 21 * 6:-4]),
            'feet_contact_loss': self.feet * self.recons_fn(gt[..., -4:], recons[..., -4:]),
            'commit_loss': self.commit * outputs['commit_loss']
        }
        return loss_dict


@MODELS.register_module(force=True)
class TomatoLoss(BaseLoss):
    def __init__(self,
                 recons_type='l2',
                 nb_joints=52,
                 pos=1.5,
                 root=1.,
                 vel=1.,
                 commit=0.02):
        """ VQVAE loss for humantomato motion representation
        :param recons_type: reconstruction loss type
        :param nb_joints: number of joints
        :param recons_type: weight of reconstruction loss
        :param vel: weight of velocity loss
        :param commit: weight of commit loss
        """
        super().__init__(recons_type, nb_joints)

        # 4 global motion associated to root
        # 3 local xyz
        # 3 global vel
        self.motion_dim = (nb_joints - 1) * 3 + nb_joints * 3 + 4
        self.pos = pos
        self.root = root
        self.vel = vel
        self.commit = commit

        self.pos_begin = 4
        self.pos_end = 4 + (self.nb_joints - 1) * 3
        self.vel_begin = 4 + (self.nb_joints - 1) * 3
        self.vel_end = 4 + (self.nb_joints - 1) * 3 + self.nb_joints * 3

    def forward(self, motion_pred, motion_gt, loss_commit=None, have_hand=None):
        loss_pos = self.pos * self.forward_pos(motion_pred, motion_gt)
        loss_root = self.root * self.forward_root(motion_pred, motion_gt)
        loss_vel = self.vel * self.forward_vel(motion_pred, motion_gt)
        loss_commit = self.commit * loss_commit

        loss_dict = {
            'root_loss': loss_root,
            'pos_loss': loss_pos,
            'vel_loss': loss_vel,
            'commit_loss': loss_commit}
        return loss_dict

    def forward_pos(self, motion_pred, motion_gt):
        loss = self.recons_fn(motion_pred[..., self.pos_begin:self.pos_end],
                              motion_gt[..., self.pos_begin:self.pos_end])
        return loss

    def forward_vel(self, motion_pred, motion_gt):
        loss = self.recons_fn(motion_pred[..., self.vel_begin:self.vel_end],
                              motion_gt[..., self.vel_begin:self.vel_end])
        return loss

    def forward_root(self, motion_pred, motion_gt):
        loss = self.recons_fn(motion_pred[..., :4], motion_gt[..., :4])
        return loss


@MODELS.register_module(force=True)
class SmplxLoss(BaseLoss):
    def __init__(self,
                 recons_type: str = 'l1_smooth',
                 transl=1.,
                 rot=1.,
                 joints=0.,
                 commit=0.02
                 ):
        super().__init__(recons_type)
        self.transl = transl
        self.rot = rot
        self.commit = commit
        self.joints = joints

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        has_hand = data_samples.has_hand
        rot_type = data_samples.rot_type

        commit_loss = outputs.get('commit_loss', None)
        pred_motion = outputs['pred_motion']
        gt_motion = inputs['motion']
        num_frames = pred_motion.shape[1]
        # b t j c
        recons_joints = outputs['recons_joints'] - pred_motion[..., :3].unsqueeze(-2)
        gt_joints = outputs['gt_joints'] - gt_motion[..., :3].unsqueeze(-2)

        loss_transl = self.recons_fn(pred_motion[..., :3], gt_motion[..., :3], reduction='mean')
        loss_rot = self.recons_fn(pred_motion[..., 3:], gt_motion[..., 3:], reduction='none')
        loss_joints = self.recons_fn(recons_joints, gt_joints, reduction='none')
        # assert not torch.any(torch.isnan(loss_rot)), (f'gt has nan:{torch.any(torch.isnan(gt_motion))},'
        #                                               f'pred has nan:{torch.any(torch.isnan(pred_motion))}')

        rot_mask = batch_get_mask_rotation(num_frames, has_hand, rot_type[0]).to(loss_rot)

        joints_mask = batch_get_mask_joints(num_frames, has_hand).to(loss_joints)
        loss_rot = (loss_rot * rot_mask).mean()
        loss_joints = (loss_joints * joints_mask).mean()

        loss_dict = dict(
            loss_transl=loss_transl * self.transl,
            loss_rot=loss_rot * self.rot,
            loss_joints=loss_joints * self.joints,
            loss_commit=commit_loss * self.commit
        )

        return loss_dict


@MODELS.register_module(force=True)
class InterHumanWholeBodyLoss(BaseLoss):
    def __init__(self,
                 recons_type='l2',
                 nb_joints=52,
                 pos_body=1.5,
                 pos_hand=1.5,
                 root=1.,
                 feet=1.,
                 rot_body=1.,
                 rot_hand=1.,
                 vel_body=1.,
                 vel_hand=1.,
                 commit=0.02):
        """
        :param recons_type: reconstruction loss type
        :param nb_joints: number of joints
        :param recons: weight of reconstruction loss
        :param vel: weight of velocity loss
        :param commit: weight of commit loss
        """
        super().__init__(recons_type)
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 4 foot contact
        self.nb_joints = nb_joints
        self.pos_body = pos_body
        self.pos_hand = pos_hand
        self.rot_body = rot_body
        self.rot_hand = rot_hand
        self.root = root
        self.feet = feet
        self.vel_body = vel_body
        self.vel_hand = vel_hand
        self.commit = commit

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        gt = inputs['motion']
        recons = outputs['pred_motion']

        loss_dict = {
            'transl_loss': self.root * self.recons_fn(gt[..., :3], recons[..., :3]),
            'pos_body_loss': self.pos_body * self.recons_fn(gt[..., 3: 66], recons[..., 3: 66]),
            'pos_hand_loss': self.pos_hand * self.recons_fn(gt[..., 66: 156], recons[..., 66:156]),
            'vel_body_loss': self.vel_body * self.recons_fn(gt[..., 156: 156 + 22 * 3], recons[..., 156: 156 + 22 * 3]),
            'vel_hand_loss': self.vel_hand * self.recons_fn(gt[..., 156 + 22 * 3: 156 + 52 * 3],
                                                            recons[..., 156 + 22 * 3: 156 + 52 * 3]),
            'rot_body_loss': self.rot_body * self.recons_fn(gt[..., 156 + 52 * 3: 156 + 52 * 3 + 22 * 6],
                                                            recons[..., 156 + 52 * 3: 156 + 52 * 3 + 22 * 6]),
            'rot_hand_loss': self.rot_hand * self.recons_fn(gt[..., 156 + 52 * 3 + 22 * 6:-4],
                                                            recons[..., 156 + 52 * 3 + 22 * 6:-4]),
            'feet_contact_loss': self.feet * self.recons_fn(gt[..., -4:], recons[..., -4:]),
            'commit_loss': self.commit * outputs['commit_loss']
        }
        return loss_dict


@MODELS.register_module()
class JointsWholeBodyLoss(BaseLoss):
    def __init__(self,
                 recons_type='l2',
                 nb_joints=52,
                 pos_body=1.5,
                 pos_hand=1.5,
                 root=1.,
                 commit=0.02,
                 rot_hand=0.,
                 rot_body=0.,
                 ):
        super().__init__(recons_type, nb_joints)
        self.pos_body = pos_body
        self.pos_hand = pos_hand
        self.rot_hand = rot_hand
        self.rot_body = rot_body
        self.root = root
        self.commit = commit

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        gt = inputs['motion']
        recons = outputs['pred_motion']

        if self.rot_body or self.rot_hand:
            gt_rotation = outputs['gt_rotation'].nan_to_num()
            # cont6d should be in range -1, 1. filter some nan
            recons_rotation = outputs['pred_rotation'].nan_to_num()

        loss_dict = {
            'transl_loss': self.root * self.recons_fn(gt[..., :3], recons[..., :3]),
            'pos_body_loss': self.pos_body * self.recons_fn(gt[..., 3: 66], recons[..., 3: 66]),
            'pos_hand_loss': self.pos_hand * self.recons_fn(gt[..., 66: 156], recons[..., 66: 156]),
            'commit_loss': self.commit * outputs['commit_loss']
        }
        if self.rot_body or self.rot_hand:
            loss_dict['rot_body_loss'] = self.rot_body * self.recons_fn(gt_rotation[..., :22, :],
                                                                        recons_rotation[..., :22, :])
            loss_dict['rot_hand_loss'] = self.rot_hand * self.recons_fn(gt_rotation[..., 22:, :],
                                                                        recons_rotation[..., 22:, :])
        return loss_dict

@MODELS.register_module()
class TranslCont6dLoss(BaseLoss):
    def __init__(self,
                 recons_type='l2',
                 nb_joints=52,
                 rot_body=1.2,
                 rot_hand=1.2,
                 joints_body=0.75,
                 joints_hand=0.75,
                 root=1.,
                 commit=0.02
                 ):
        super().__init__(recons_type, nb_joints)
        self.rot_body = rot_body
        self.rot_hand = rot_hand
        self.root = root
        self.joints_body = joints_body
        self.joints_hand = joints_hand
        self.commit = commit

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        has_hand = data_samples.get('has_hand')

        gt = inputs['motion']
        pred = outputs['pred_motion']

        gt_hands_cont6d = rearrange(gt[..., 135:], 'b t (j c) -> b j t c', c=6)
        pred_hands_cont6d = rearrange(pred[..., 135:], 'b t (j c) -> b j t c', c=6)

        gt_joints = outputs['gt_joints']
        pred_joints = outputs['pred_joints']

        gt_hand_joints = rearrange(gt_joints[:, :, 22:], 'b t j c -> b j t c')
        pred_hand_joints = rearrange(pred_joints[:, :, 22:], 'b t j c -> b j t c')

        loss_dict = {
            'transl_loss': self.root * self.recons_fn(gt[..., :3], pred[..., :3]),

            'rot_body_loss': self.rot_body * self.recons_fn(gt[..., 3: 135], pred[..., 3: 135]),
            'rot_hand_loss': self.rot_hand * self.recons_fn(gt_hands_cont6d,
                                                            pred_hands_cont6d),
            'joints_body_loss': self.joints_body * self.recons_fn(gt_joints[:, :, 1:22], pred_joints[:, :, 1:22]),
            'joints_hand_loss': self.joints_hand * self.recons_fn(gt_hand_joints, pred_hand_joints),

            'commit_loss': self.commit * outputs['commit_loss']
        }
        return loss_dict

@MODELS.register_module(force=True)
class GlobalTomatoLoss(BaseLoss):
    def __init__(self,
                 recons_type='l2',
                 nb_joints=52,
                 pos_body=1.5,
                 pos_hand=1.5,
                 root=1.,
                 vel_body=1.,
                 vel_hand=1.,
                 commit=0.02):
        """
        :param recons_type: reconstruction loss type
        :param nb_joints: number of joints
        :param commit: weight of commit loss
        """
        super().__init__(recons_type, nb_joints)

        # 6 * num joints motion (3 xyz, 3 vel)
        self.pos_body = pos_body
        self.pos_hand = pos_hand
        self.root = root
        self.vel_body = vel_body
        self.vel_hand = vel_hand
        self.commit = commit

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        gt = inputs['motion']
        recons = outputs['pred_motion']
        loss_dict = {
            'root_loss': self.root * self.recons_fn(gt[..., :3], recons[..., :3]),
            'pos_body_loss': self.pos_body * self.recons_fn(gt[..., 3: 66], recons[..., 3: 66]),
            'pos_hand_loss': self.pos_hand * self.recons_fn(gt[..., 66: 156], recons[..., 66:156]),
            'vel_body_loss': self.vel_body * self.recons_fn(gt[..., 156: 156 + 66], recons[..., 156: 156 + 66]),
            'vel_hand_loss': self.vel_hand * self.recons_fn(gt[..., 156 + 66:],
                                                            recons[..., 156 + 66:]),
            'commit_loss': self.commit * outputs['commit_loss']
        }
        return loss_dict
