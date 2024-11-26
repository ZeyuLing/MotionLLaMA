from typing import Dict, Optional, Union, List

import torch
from mmengine.model import BaseModel
from torch import Tensor
from torch.distributions import Distribution, Normal

from mmotion.models.losses.mld_vae_loss import MldVAELoss
from mmotion.models.generators.tma.motionencoder import ActorAgnosticEncoder
from mmotion.models.generators.tma import ActorAgnosticDecoder

from mmotion.registry import MODELS
from mmotion.structures import DataSample


@MODELS.register_module()
class GestureVAE(BaseModel):
    def __init__(self, motionencoder: Dict,
                 motiondecoder: Dict, loss_cfg: Dict = dict(
                type='MldVAELoss'), init_cfg=None,
                 data_preprocessor=None):
        super().__init__(data_preprocessor, init_cfg)
        self.motionencoder: ActorAgnosticEncoder = MODELS.build(motionencoder)
        self.motiondecoder: ActorAgnosticDecoder = MODELS.build(motiondecoder)

        self.loss: MldVAELoss = MODELS.build(loss_cfg)

        self.fact = None

    def forward(self,
                inputs: torch.Tensor,
                data_samples: DataSample = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        if mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'loss':
            return self.forward_loss(inputs, data_samples)
        elif mode == 'tensor':
            return self.forward_tensor(inputs, data_samples)
        else:
            raise NotImplementedError(
                'Forward is not implemented now, please use infer.')

    def encode_motion(self, motion: Union[List[Tensor], Tensor],
                      lengths: Optional[List[int]] = None):
        """
        :param motion: A batch of motion vectors
        :param lengths: num of valid frames of each motion vector.
        :return:
        """
        if isinstance(motion, list):
            # if length is not unique, do padding
            lengths = [len(m) for m in motion]
            if any([l != lengths[0] for l in lengths]):
                max_length = max(lengths)
                motion = [torch.cat([m, torch.zeros([max_length - l, *m.shape[1:]]).to(m)])
                          for m, l in zip(motion, lengths)]

            motion = torch.stack(motion, dim=0)
        distribution = self.motionencoder(motion, lengths)
        latent_vector = self.sample_from_distribution(distribution)
        return distribution, latent_vector

    def motion_to_motion_forward(
            self,
            features,
            lengths: Optional[List[int]] = None,
    ):
        """
        This function encodes the given motion features into a latent space and then decodes them into a motion.
        If `return_latent` is True, it also returns the latent vector and the distribution.
        If `mask_ratio` is greater than 0, it masks a portion of the features before encoding.

        Args:
            features (Tensor): The motion features to encode.
            lengths (Optional[List[int]]): The lengths of the motion features. Default is None.
        Returns:
            features (Tensor): The decoded motion.
            latent_vector (Tensor): The latent vector. Only returned if `return_latent` is True.
            Distribution: The distribution. Only returned if `return_latent` is True.
        """

        # Encode the motion to the latent space
        # Behaves differently based on whether the class is set to use a VAE or not,
        # and whether a mask ratio is provided.

        # Decode the latent vector to a motion
        distribution, latent_vector = self.encode_motion(features, lengths)
        features = self.motiondecoder(latent_vector, lengths)

        return features, latent_vector, distribution

    def forward_tensor(self, inputs, data_samples=None):
        motion = inputs["motion"]
        lengths = data_samples.get('num_frames')
        ret = self.motion_to_motion_forward(
            motion, lengths
        )
        pred_motion, latent_from_motion, dist_pred = ret

        # Assuming te ground truth is standard gaussian dist
        mu_gt = torch.zeros_like(dist_pred.loc)
        scale_gt = torch.ones_like(dist_pred.scale)
        dist_gt = torch.distributions.Normal(mu_gt, scale_gt)

        return dict(
            pred_motion=pred_motion,
            gt_motion=motion,
            lat_motion=latent_from_motion,
            dist_gt=dist_gt,
            dist_pred=dist_pred,
        )

    def sample_from_distribution(
            self,
            distribution: Normal,
            *,
            fact: Optional[bool] = None,
            sample_mean: Optional[bool] = False,
    ):
        """
        This function samples from a given distribution. If `sample_mean` is True, it returns the mean of the distribution.
        If `fact` is provided, it rescales the sample using the reparameterization trick.

        Args:
            distribution (Distribution): The distribution to sample from.
            fact (Optional[bool]): A factor to rescale the sample. Default is None.
            sample_mean (Optional[bool]): Whether to return the mean of the distribution. Default is False.

        Returns:
            Tensor: The sampled tensor.
        """
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def forward_loss(self, inputs, data_samples: DataSample = None):
        feature_dict = self.forward_tensor(inputs, data_samples)
        feature_dict.pop('lat_motion', None)
        return self.loss(
            **feature_dict
        )

    @torch.no_grad()
    def forward_predict(self, inputs: Dict, data_samples=None):
        output_dict = self.forward_tensor(inputs, data_samples)
        output_dict = self.decompose_vector(output_dict, data_samples)
        data_samples.set_data(output_dict)

        data_sample_list = data_samples.split(allow_nonseq_value=True)

        return data_sample_list

    def decompose_vector(self, output_dict: Dict, data_sample: DataSample) -> Dict:
        """ Get joints, rotation, feet_contact, ... from predicted motion vectors.
        """
        normalized_gt_motion = output_dict["gt_motion"]
        normalized_pred_motion = output_dict["pred_motion"]

        gt_motion = self.data_preprocessor.destruct(normalized_gt_motion, data_sample)
        pred_motion = self.data_preprocessor.destruct(normalized_pred_motion, data_sample)

        gt_joints = self.data_preprocessor.vec2joints(gt_motion, data_sample)
        pred_joints = self.data_preprocessor.vec2joints(pred_motion, data_sample)

        output_dict.update(
            {
                'gt_joints': gt_joints,
                'pred_joints': pred_joints,
            }
        )
        return output_dict
