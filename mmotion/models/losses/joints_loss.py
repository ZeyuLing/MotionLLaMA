from typing import Dict

from mmotion.models.losses.recon_loss import BaseLoss
from mmotion.registry import MODELS

from mmotion.structures import DataSample
from mmotion.utils.mask_utils import get_mask_rotation, batch_get_mask_rotation, batch_get_mask_joints


@MODELS.register_module(force=True)
class JointsLoss(BaseLoss):
    def __init__(self,
                 recons_type: str = 'l1_smooth',
                 ):
        super().__init__(recons_type)

    def forward(self, inputs: Dict, outputs: Dict, data_samples: DataSample):
        has_hand = data_samples.has_hand
        commit_loss = outputs.get('commit_loss', None)
        recons_joints = outputs['recons_joints']
        gt_joints = outputs['gt_joints']
        num_frames = gt_joints.shape[1]

        loss_transl = self.recons_fn(recons_joints[:, :, :1], gt_joints[:, :, :1], reduction='mean')
        loss_body = self.recons_fn(recons_joints[:, :, 1:22], gt_joints[:, :, 1:22], reduction='mean')
        loss_hands = self.recons_fn(recons_joints[:, :, 22:], gt_joints[:, :, 22:], reduction='none')

        joints_mask = batch_get_mask_joints(num_frames, has_hand).to(loss_transl)
        loss_hands = (loss_hands*joints_mask[:, :, 22:]).mean()

        loss_dict = dict(
            loss_transl=loss_transl,
            loss_body=loss_body,
            loss_hands=loss_hands
        )

        if commit_loss is not None:
            loss_dict['loss_commit'] = commit_loss

        return loss_dict
