from torch import nn
from torch.nn import SmoothL1Loss

from mmotion.models.losses import KLLoss
from mmotion.registry import MODELS


@MODELS.register_module()
class MldVAELoss(nn.Module):
    def __init__(self, recons: float = 1., kl: float = 1e-4,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recons = recons
        self.kl = kl
        self.recons_fn = SmoothL1Loss(reduction='mean')
        self.kl_loss = KLLoss()

    def forward(self,
                pred_motion,
                gt_motion,
                dist_pred,
                dist_gt):
        loss_dict = {
            'recons_loss': self.recons * self.recons_fn(pred_motion, gt_motion),
            'kl_m2m_loss': self.kl * self.kl_loss(dist_pred, dist_gt),
        }

        return loss_dict
