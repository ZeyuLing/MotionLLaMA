from typing import Tuple, List

import torch
from einops import rearrange
from torch import nn, Tensor
import torch.nn.functional as F

from mmotion.models.archs.quantizers.utils import round_ste
from mmotion.registry import MODELS


@MODELS.register_module()
class FSQuantizer(nn.Module):
    """
        smaller codebook size, faster inferring speed.
    """

    def __init__(self, code_dim, levels: List[int]):
        super().__init__()
        self.code_dim = code_dim if code_dim is not None else len(levels)
        self.e_dim = len(levels)  # effective dim

        has_projections = self.code_dim != self.e_dim

        self.project_in = nn.Linear(self.code_dim, self.e_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(self.e_dim, self.code_dim) if has_projections else nn.Identity()

        self.bn = nn.BatchNorm1d(self.e_dim)  # before project_in

        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64)
        self.register_buffer("_basis", _basis)

        self.n_e = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.n_e))
        self.register_buffer("implicit_codebook", implicit_codebook)
        dummy_loss = torch.tensor([0.0])
        self.register_buffer("dummy_loss", dummy_loss)

    def forward(self, z: Tensor) -> Tuple:
        """
        :param z: b c t
        :return: recon, commit_loss, perplexity
        """
        b, t, c = z.shape
        z_in = z
        # b t c
        z = self.preprocess(z)
        z = self.project_in(z)
        z = self.bn(z)
        assert z.shape[-1] == self.e_dim, "z shape{}".format(z.shape)
        z_flattened = z.contiguous().view(-1, self.e_dim)

        zhat = self._quantize(z_flattened).view(z.shape)
        min_encoding_indices = self.codes_to_indices(zhat)

        zhat = self.project_out(zhat)

        zhat = rearrange(zhat, '(b t) c -> b c t', b=b)

        commit_loss = F.mse_loss(z_in, zhat.detach())

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return zhat, commit_loss, perplexity

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def _quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.e_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int64)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        if project_out:
            codes = self.project_out(codes)
        return codes

    # interfaces
    def quantize(self, z):
        assert len(z.shape) == 2 and z.shape[-1] == self.e_dim
        return self.codes_to_indices(self._quantize(z))

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.implicit_codebook)
        return x

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = rearrange(x, 'n c t -> (n t) c')
        return x