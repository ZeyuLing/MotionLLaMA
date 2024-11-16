import torch
import pytorch_lightning as pl

from typing import List, Optional

from einops import repeat
from torch import nn, Tensor

from mmotion.models.generators.tma.operator.postional_encoding import PositionalEncoding
from mmotion.models.generators.tma.utils import lengths_to_mask
from mmotion.registry import MODELS


@MODELS.register_module()
class ActorAgnosticEncoder(pl.LightningModule):
    """
    This class is an actor-agnostic encoder for encoding input features.

    Attributes:
    - skel_embedding: a linear layer for embedding the input features.
    - mu_token, logvar_token: parameters for generating the mean and log variance of the latent distribution (only if VAE is used).
    - emb_token: parameter for generating the final output (only if VAE is not used).
    - sequence_pos_encoding: a positional encoding layer for adding positional information to the input features.
    - seqTransEncoder: a transformer encoder for encoding the input features.

    Methods:
    - __init__: initializes the ActorAgnosticEncoder object with the given parameters.
    - forward: encodes the input features and returns the encoded output.
    """

    def __init__(
            self,
            nfeats: int,
            vae: bool,
            latent_dim: int = 256,
            ff_size: int = 1024,
            num_layers: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            activation: str = "gelu",
            **kwargs
    ):
        """
        Initializes the ActorAgnosticEncoder object with the given parameters.

        Inputs:
        - nfeats: the number of input features.
        - vae: a flag indicating whether to use a Variational Autoencoder (VAE).
        - latent_dim: the dimension of the latent space.
        - ff_size: the size of the feedforward network in the transformer.
        - num_layers: the number of layers in the transformer.
        - num_heads: the number of attention heads in the transformer.
        - dropout: the dropout rate.
        - activation: the activation function to use in the transformer.

        Outputs: None
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        # Initialize the positional encoding layer
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        # Initialize the transformer encoder layer
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        # Initialize the transformer encoder
        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        """
        Encodes the input features and returns the encoded output.

        Inputs:
        - features: a tensor of input features.
        - lengths: a list of lengths of the input features.

        Outputs: the encoded output.
        """

        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)
        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        if self.hparams.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.hparams.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            dist = torch.distributions.Normal(mu, std)
            return dist

        else:
            return final[0]

@MODELS.register_module()
class ACTORStyleEncoder(nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
            self,
            nfeats: int,
            vae: bool,
            latent_dim: int = 256,
            ff_size: int = 1024,
            num_layers: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        x = features
        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]
