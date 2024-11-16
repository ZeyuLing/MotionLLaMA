from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch
from mmotion.models.generators.mld.mld_vae import SkipTransformerEncoder, PositionEmbeddingLearned1D
from mmotion.registry import MODELS


@MODELS.register_module()
class MLDDenoiser(nn.Module):
    def __init__(self,
                 latent_dim=256,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos=True,
                 freq_shift=0,
                 text_encoded_dim: int = 256,
                 ):
        super().__init__()
        self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                   freq_shift)
        self.time_embedding = TimestepEmbedding(text_encoded_dim, latent_dim)

        self.emb_proj = nn.Sequential(nn.ReLU(), nn.Linear(text_encoded_dim, latent_dim)) \
            if text_encoded_dim != latent_dim else nn.Identity()

        encoder_layer = TransformerEncoderLayer(
            latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, latent_dim,
                                              num_layers, encoder_norm)

        self.query_pos = PositionEmbeddingLearned1D(latent_dim)

    def forward(self,
                sample,
                timestep,
                text_feature,
                ):
        """
        :param sample: b n c
        :param timestep: batch_size
        :param text_feature: b n_text c
        :return:
        """
        time_emb = self.time_proj(timestep)
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(1)

        text_emb_latent = self.emb_proj(text_feature)
        emb_latent = torch.cat((time_emb, text_emb_latent), dim=1)

        xseq = torch.cat((sample, emb_latent), dim=1)

        xseq = self.query_pos(xseq)
        tokens = self.encoder(xseq)

        sample = tokens[:, :sample.shape[1]]

        return sample
