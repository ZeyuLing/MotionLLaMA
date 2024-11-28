from typing import Dict

import torch
import torch.nn.functional as F
from mmengine.model import BaseModel

from mmotion.models.generators.emage.vq import VQVAEConvZero
from mmotion.registry import MODELS
from mmotion.structures import DataSample


class Emage(BaseModel):
    def __init__(self,
                 hand_vqvae: Dict,
                 upper_vqvae: Dict,
                 lower_vqvae: Dict,
                 global_motion: Dict,
                 mage_transformer: Dict,
                 data_preprocessor: Dict = None,
                 init_cfg: Dict = None
                 ):
        super().__init__(data_preprocessor, init_cfg)
        self.hand_vqvae: VQVAEConvZero = MODELS.build(hand_vqvae).eval()
        self.upper_vqvae: VQVAEConvZero = MODELS.build(upper_vqvae).eval()
        self.lower_vqvae: VQVAEConvZero = MODELS.build(lower_vqvae).eval()
        self.global_motion = MODELS.build(global_motion).eval()

        self.mage_transformer = MODELS.build(mage_transformer)

    def forward_tensor(self, inputs, data_samples):
        motion = inputs['motion']
        upper = get_upper(motion)
        lower = get_lower(motion)
        hands = get_hands(motion)

        audio = inputs['audio']

        caption = data_samples.get('caption')

        speaker_id = data_samples.get('speaker_id')

        upper_idx = self.upper_vqvae.map2index(upper)
        lower_idx = self.lower_vqvae.map2index(upper)
        hands_idx = self.hands_vqvae.map2index(upper)

        latent_upper = self.upper_vqvae.map2latent(upper)  # bs*n/4
        latent_hands = self.hand_vqvae.map2latent(hands)  # bs*n/4
        latent_lower = self.lower_vqvaer.map2latent(lower)  # bs*n/4

        rec_dict: Dict = self.mage_transformer(
            audio=audio,
            text=caption,
            speaker_id=speaker_id,
            motion=motion,
        )

        rec_dict.update(
            {
                'latent_upper': latent_upper,
                'latent_lower': latent_lower,
                'latent_hands': latent_hands,

                'upper_idx': upper_idx,
                'lower_idx': lower_idx,
                'hands_idx': hands_idx
            }
        )

        return rec_dict

    def forward_loss(self, inputs, data_samples):
        rec_dict = self.forward_tensor(inputs, data_samples)

        loss_dict = {
            'upper_mse_loss': 3 * F.mse_loss(rec_dict['latent_upper'], rec_dict['rec_upper']),
            'lower_mse_loss': 3 * F.mse_loss(rec_dict['latent_lower'], rec_dict['rec_lower']),
            'hands_mse_loss': 3 * F.mse_loss(rec_dict['latent_hands'], rec_dict['rec_hands']),
            'hands_ce_loss': F.cross_entropy(rec_dict['logits_hands'], rec_dict['hands_idx']),
            'upper_ce_loss': F.cross_entropy(rec_dict['logits_upper'], rec_dict['upper_idx']),
            'lower_ce_loss': F.cross_entropy(rec_dict['logits_lower'], rec_dict['lower_idx']),
        }

        return loss_dict

    def forward_predict(self, inputs, data_samples: DataSample):
        rec_dict = self.forward_tensor(inputs, data_samples)
        pred_upper_idx = torch.argmax(rec_dict['logits_upper'], dim=-1)
        pred_lower_idx = torch.argmax(rec_dict['logits_lower'], dim=-1)
        pred_hands_idx = torch.argmax(rec_dict['logits_hands'], dim=-1)

        pred_lower = self.lower_vqvae.decode(pred_lower_idx)
        pred_upper = self.lower_vqvae.decode(pred_lower_idx)
        pred_hands = self.lower_vqvae.decode(pred_lower_idx)

        smplh_params = merge_smplh(pred_upper, pred_lower, pred_hands)
        data_samples.set_field(smplh_params, 'smplh_params')
        data_samples = self.post_process(inputs, data_samples)
        return data_samples

    def post_process(self, data_samples:DataSample):
        smplh_params = data_samples.get('smplh_params')
        joints = self.smpl_model(smplh_params)
        motion = joints2motion(joints)



