from typing import Dict

import torch
from einops import rearrange

from mmotion.models.losses.recon_loss import BaseLoss
from mmotion.registry import MODELS
from mmotion.structures import DataSample

@MODELS.register_module()
class HoMiLossV1(BaseLoss):
    def __init__(self, recons_type='l2', nb_joints:int=52, pos:float=1.5, hand_rot:float= 0.1, commit:float=0.02):
        super().__init__(recons_type, nb_joints)
        self.hand_rot = hand_rot
        self.commit = commit
        self.pos = pos

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        has_hand = data_samples.get('has_hand')

        gt = inputs['motion']
        pred = outputs['pred_motion']

        gt_rotation = outputs['gt_rotation']
        pred_rotation= outputs['pred_rotation']

        gt_hand_rotation = rearrange(gt_rotation[:, :, 22:], 'b t j c -> b j t c')[has_hand]
        pred_hand_rotation = rearrange(pred_rotation[:, :, 22:], 'b t j c -> b j t c')[has_hand]

        nan_mask = ~torch.isnan(pred_hand_rotation) & ~torch.isnan(gt_hand_rotation)

        loss_dict = {
            'base_loss': self.recons_fn(gt, pred),

            'rot_hand_loss': self.hand_rot * self.recons_fn(gt_hand_rotation[nan_mask], pred_hand_rotation[nan_mask]),

            'commit_loss': self.commit * outputs['commit_loss']
        }
        return loss_dict