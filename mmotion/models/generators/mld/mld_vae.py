from typing import Optional, Union, Dict, List, Tuple

from mmengine.model import BaseModel
import torch
from torch import nn, Tensor
from torch.distributions import Normal
from torch.nn import TransformerEncoderLayer
from torch.nn.modules.transformer import _get_clones

from mmotion.models.losses.mld_vae_loss import MldVAELoss
from mmotion.models.generators.tma.utils import lengths_to_mask
from mmotion.registry import MODELS
from mmotion.structures import DataSample


class PositionEmbeddingLearned1D(nn.Module):

    def __init__(self, d_model, max_len=500):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        # self.pe = pe.unsqueeze(0).transpose(0, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pe)

    def forward(self, x):
        # not used in the final model

        pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        x = x + pos

        return x
        # return self.dropout(x)


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, latent_dim: int,
                 num_layers, norm=None):
        super().__init__()
        self.d_model = latent_dim

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_blocks = _get_clones(encoder_layer, 1)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        x = src

        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                       src_key_padding_mask=src_key_padding_mask)
            xs.append(x)
        for module in self.middle_blocks:
            x = module(x, src_mask=mask,
                       src_key_padding_mask=src_key_padding_mask)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                       src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


@MODELS.register_module()
class MldVAE(BaseModel):
    def __init__(self, nfeats: int = 156,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 loss_cfg: dict = None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(MldVAE, self).__init__(data_preprocessor, init_cfg)
        self.query_pos_encoder = PositionEmbeddingLearned1D(latent_dim)
        self.query_pos_decoder = PositionEmbeddingLearned1D(latent_dim)

        encoder_layer = TransformerEncoderLayer(
            latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
            batch_first=True
        )
        encoder_norm = nn.LayerNorm(latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, latent_dim, num_layers,
                                              encoder_norm)

        decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder = SkipTransformerEncoder(encoder_layer, latent_dim, num_layers,
                                              decoder_norm)

        self.global_motion_token = nn.Parameter(
            torch.randn(2, latent_dim))

        self.skel_embedding = nn.Linear(nfeats, latent_dim)
        self.final_layer = nn.Linear(latent_dim, nfeats)
        self.loss: MldVAELoss = MODELS.build(loss_cfg)

    def forward_tensor(self, inputs, data_samples):
        motion = inputs['motion']
        lengths = data_samples.get('num_frames')
        mask = lengths_to_mask(lengths, device=motion.device, max_len=max(lengths)).unsqueeze(-1)
        z, dist_pred = self.encode(motion, lengths)
        pred_motion = self.decode(z, lengths)
        motion = motion * mask
        pred_motion = pred_motion * mask
        # ground truth distribution
        mu_gt = torch.zeros_like(dist_pred.loc)
        scale_gt = torch.ones_like(dist_pred.scale)
        dist_gt = torch.distributions.Normal(mu_gt, scale_gt)
        return dict(
            dist_gt=dist_gt,
            dist_pred=dist_pred,
            pred_motion=pred_motion,
            gt_motion=motion
        )

    def forward_loss(self, inputs, data_samples):
        recons_dict = self.forward_tensor(inputs, data_samples)
        loss = self.loss(**recons_dict)

        return loss

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Tuple[Tensor, Normal]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        # b n
        mask = lengths_to_mask(lengths, device)
        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[None, :, :], (bs, 1, 1))
        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[1]),
                                dtype=torch.bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 1)

        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq,
                            src_key_padding_mask=~aug_mask)[:, :dist.shape[1]]

        mu = dist[:, 0:1]
        logvar = dist[:, 1:]

        # resampling
        std = logvar.exp().pow(0.5)
        dist = Normal(mu, std)
        latent = dist.rsample()
        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):
        """
        :param z: b n d
        :param lengths: expected length of decoded motion
        :return:
        """
        mask = lengths_to_mask(lengths, z.device)

        bs, nframes = mask.shape

        queries = torch.zeros(bs, nframes, z.shape[-1], device=z.device)
        xseq = torch.cat((z, queries), dim=1)
        z_mask = torch.ones((bs, z.shape[1]),
                            dtype=torch.bool,
                            device=z.device)
        augmask = torch.cat((z_mask, mask), dim=1)

        xseq = self.query_pos_decoder(xseq)
        output = self.decoder(
            xseq, src_key_padding_mask=~augmask)[:, z.shape[1]:]

        output = self.final_layer(output)
        return output

    def forward_predict(self, inputs, data_samples):
        output_dict = self.forward_tensor(inputs, data_samples)
        output_dict = self.decompose_vector(output_dict, data_samples)
        data_samples.set_data(output_dict)

        data_sample_list = data_samples.split(allow_nonseq_value=True)
        return data_sample_list

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
