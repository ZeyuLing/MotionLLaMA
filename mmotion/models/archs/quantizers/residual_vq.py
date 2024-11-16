import numpy as np
from functools import partial
import torch
from einops import rearrange, pack
from einx import get_at
from torch import nn, cumsum
from torch.nn import ModuleList, Module

from mmotion.models.archs.quantizers.ema_reset import EMAResetQuantizer
from mmotion.models.archs.quantizers.utils import default
from mmotion.registry import MODELS


@MODELS.register_module()
class ResidualVQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
            self,
            dim: int,
            num_quantizers: int,
            layer_quantizer: dict,
            codebook_dim=None,
    ):
        super().__init__()
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.layers: ModuleList[EMAResetQuantizer] = ModuleList([MODELS.build(layer_quantizer) for _ in
                                                                 range(num_quantizers)])
        self.counter = [0] * self.codebook_size

    @property
    def codebook_size(self):
        return self.layers[0].codebook_size

    @property
    def codebook_dim(self):
        return self.layers[0].codebook_dim

    @property
    def codebooks(self):
        """
        :return: n_q, n_codes, dim
        """
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def quantize(self, x):
        """
        :param x: b n d
        :return: b n d q
        """
        x = self.project_in(x)

        quantized_out = 0.
        residual = x
        all_indices = []

        # go through the layers

        for quantizer_index, vq in enumerate(self.layers):
            # vector quantize forward
            quantized, embed_indices, commit_loss, _ = vq(
                rearrange(residual, 'b n d -> b d n')
            )
            quantized = rearrange(quantized, 'b d n -> b n d')
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(embed_indices)

        all_indices = torch.stack(all_indices, dim=0)
        if self.num_quantizers == 1:
            all_indices = all_indices.squeeze(0)
        return all_indices

    def dequantize(self, indices, start_layer=0):
        """
        :param indices: b n or b n n_q
        :return: b n d, final dequantized vector
        """
        if len(indices.shape) == 2:
            indices = indices[..., None]
        else:
            indices = rearrange(indices, 'q b n -> b n q')
        batch, num_layers = indices.shape[0], indices.shape[-1]

        indices, ps = pack([indices], 'b * q')
        all_codes = get_at('q [c] d, b n q -> q b n d',
                           self.codebooks[start_layer:start_layer + num_layers], indices)

        all_codes = cumsum(all_codes, dim=0)[-1]

        return all_codes

    def forward(
            self,
            x
    ):
        """
        :param x: b d n
        :return:
        """
        x = rearrange(x, 'b d n -> b n d')
        x = self.project_in(x)
        x = rearrange(x, 'b n d -> b d n')
        quantized_out = 0.
        residual = x
        layer_out = []
        all_losses = []
        all_indices = []

        # go through the layers

        for quantizer_index, vq in enumerate(self.layers):
            # vector quantize forward
            quantized, embed_indices, commit_loss, perplexity = vq(
                residual
            )

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            layer_out.append(quantized_out)
            all_indices.append(embed_indices)
            all_losses.append(commit_loss)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # stack all losses and indices

        all_losses, all_indices, layer_out = map(partial(torch.stack, dim=-1), (all_losses, all_indices, layer_out))
        return quantized_out, all_indices, all_losses, layer_out
