import torch
from mmotion.models.generators.tma.tmr import TMR
from mmotion.registry import MODELS

import torch.nn.functional as F


@MODELS.register_module()
class InterTMR(TMR):

    def forward_tensor(self, inputs, data_samples=None):
        motion_a = inputs['motion']
        motion_b = inputs['interactor_motion']
        caption = data_samples.get('union_caption')
        motion = torch.cat([motion_a, motion_b], dim=-1)
        with torch.no_grad():
            text_embedding = self.filter_model.encode(caption)
            text_embedding = torch.tensor(text_embedding).to(motion)
            normalized = F.normalize(text_embedding, p=2, dim=1)
            emb_dist = normalized.matmul(normalized.T)

        ret = self.text_to_motion_forward(
            caption, inputs["num_frames"]
        )
        feat_from_text, latent_from_text, distribution_from_text = ret

        ret = self.motion_to_motion_forward(
            motion, inputs["num_frames"]
        )
        feat_from_motion, latent_from_motion, distribution_from_motion = ret

        # Assuming te ground truth is standard gaussian dist
        mu_ref = torch.zeros_like(distribution_from_text.loc)
        scale_ref = torch.ones_like(distribution_from_text.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        return dict(
            f_text=feat_from_text,
            f_motion=feat_from_motion,
            f_ref=motion,
            lat_text=latent_from_text,
            lat_motion=latent_from_motion,
            dis_text=distribution_from_text,
            dis_motion=distribution_from_motion,
            emb_dist=emb_dist,
            dis_ref=distribution_ref,
        )
