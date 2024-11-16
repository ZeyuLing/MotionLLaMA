from torch import nn
from torch.nn import SmoothL1Loss

from mmotion.models.losses import KLLoss
from mmotion.models.losses.contrastive_loss import InfoNCE, ClipLoss
from mmotion.registry import MODELS


@MODELS.register_module()
class TMRLoss(nn.Module):
    def __init__(self, recons: float = 1., gen: float = 1., kl: float = 1e-5, latent: float = 1e-5,
                 info_nce: float = 0.1, info_nce_temp: float = 0.1, threshold_selfsim=0.85,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recons = recons
        self.gen = gen
        self.kl = kl
        self.latent = latent
        self.info_nce = info_nce
        self.info_nce_temp = info_nce_temp

        self.recons_fn = SmoothL1Loss(reduction='mean')
        self.latent_fn = SmoothL1Loss(reduction='mean')
        self.kl_loss = KLLoss()
        self.contrastive_loss = InfoNCE(info_nce_temp, threshold_selfsim)
        # self.contrastive_loss = ClipLoss(info_nce_temp)

    def forward(self,
                f_text,
                f_motion,
                f_ref,
                lat_text,
                lat_motion,
                dis_text,
                dis_motion,
                emb_dist,
                dis_ref=None):
        loss_dict = {
            'recons_loss': self.recons * self.recons_fn(f_motion, f_ref),
            'gen_loss': self.gen * self.recons_fn(f_text, f_ref),
            'kl_t2m_loss': self.kl * self.kl_loss(dis_text, dis_motion),
            'kl_m2t_loss': self.kl * self.kl_loss(dis_motion, dis_text),
            'kl_m2m_loss': self.kl*self.kl_loss(dis_motion, dis_ref),
            'kl_t2rm_loss': self.kl*self.kl_loss(dis_text, dis_ref),
            'latent_loss': self.latent * self.latent_fn(lat_text, lat_motion),
            'contrastive_loss': self.info_nce * self.contrastive_loss((dis_motion.loc, dis_text.loc), emb_dist),
        }

        return loss_dict
