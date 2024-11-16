from typing import Dict, Optional, Union, List

import torch
from mmengine.model import BaseModel
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.distributions import Distribution, Normal

from mmotion.models.generators.tma.motionencoder import ActorAgnosticEncoder
from mmotion.models.generators.tma.motiondecoder import ActorAgnosticDecoder
from mmotion.models.generators.tma.textencoder import DistilbertActorAgnosticEncoder

from mmotion.models.losses.tmr_loss import TMRLoss
from mmotion.registry import MODELS
from mmotion.structures import DataSample

import torch.nn.functional as F


@MODELS.register_module()
class TMR(BaseModel):
    def __init__(self, textencoder: Dict, motionencoder: Dict, motiondecoder: Dict, loss_cfg: Dict = dict(
        type='TMRLoss'), filter_model_path: str = 'checkpoints/paraphrase-MiniLM-L6-v2', init_cfg=None,
                 data_preprocessor=None):
        super().__init__(data_preprocessor, init_cfg)
        self.textencoder: DistilbertActorAgnosticEncoder = MODELS.build(textencoder)
        self.motionencoder: ActorAgnosticEncoder = MODELS.build(motionencoder)
        self.motiondecoder: ActorAgnosticDecoder = MODELS.build(motiondecoder)
        self.filter_model = SentenceTransformer(
            filter_model_path
        ).eval()

        self.loss: TMRLoss = MODELS.build(loss_cfg)

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

    def text_to_motion_forward(self,
                               text_sentences: List[str],
                               num_frames: List[int],
                               ):
        distribution, latent_vector = self.encode_text(text_sentences)

        features = self.motiondecoder(latent_vector, num_frames)

        return features, latent_vector, distribution

    def encode_text(self,
                    text_sentences: List[str]
                    ):
        if isinstance(text_sentences, str):
            text_sentences = [text_sentences]
        distribution = self.textencoder(text_sentences)
        latent_vector = self.sample_from_distribution(distribution)
        return distribution, latent_vector

    def encode_motion(self, motion: Union[List[Tensor], Tensor], lengths: Optional[List[int]] = None):
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

        with torch.no_grad():
            text_embedding = self.filter_model.encode(inputs["caption"])
            text_embedding = torch.tensor(text_embedding).to(inputs["motion"][0])
            normalized = F.normalize(text_embedding, p=2, dim=1)
            emb_dist = normalized.matmul(normalized.T)

        ret = self.text_to_motion_forward(
            inputs["caption"], inputs["num_frames"]
        )
        feat_from_text, latent_from_text, distribution_from_text = ret

        ret = self.motion_to_motion_forward(
            inputs["motion"], inputs["num_frames"]
        )
        feat_from_motion, latent_from_motion, distribution_from_motion = ret

        # Assuming te ground truth is standard gaussian dist
        mu_ref = torch.zeros_like(distribution_from_text.loc)
        scale_ref = torch.ones_like(distribution_from_text.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        return dict(
            f_text=feat_from_text,
            f_motion=feat_from_motion,
            f_ref=inputs["motion"],
            lat_text=latent_from_text,
            lat_motion=latent_from_motion,
            dis_text=distribution_from_text,
            dis_motion=distribution_from_motion,
            emb_dist=emb_dist,
            dis_ref=distribution_ref,
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

        return self.loss(
            **feature_dict
        )

    @torch.no_grad()
    def forward_predict(self, inputs: Dict, data_samples=None):
        output_dict = self.forward_tensor(inputs, data_samples)
        out_data_sample = DataSample(
            **output_dict,
            **data_samples.to_dict()
        )

        data_sample_list = out_data_sample.split(allow_nonseq_value=True)
        return data_sample_list
