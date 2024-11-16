import os
import sys
from contextlib import nullcontext
from functools import partial
from numpy import log2, ceil

from einops import rearrange, reduce, einsum
from mmengine.runner import autocast
from torch.nn import Module
from torch import nn
from torch.nn.functional import mse_loss

sys.path.append(os.curdir)
from mmotion.registry import MODELS
from mmotion.models.archs.quantizers.utils import exists, default, pack_one, maybe_distributed_mean, entropy, \
    unpack_one, l2norm, identity
import torch
import torch.nn.functional as F
class CosineSimLinear(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale

@MODELS.register_module(force=True)
class LFQ(Module):
    def __init__(
        self,
        *,
        dim = None, # dim in
        nb_code = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 1.,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.,               # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections = None,
        projection_has_bias = True,
        soft_clamp_input_value = None,
        cosine_sim_project_in = False,
        cosine_sim_project_in_scale = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # how much to shift the loss before softplus
        spherical = False,                          # from https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True               # will force the quantization step to be full precision
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(nb_code), 'either dim or nb_code must be specified for LFQ'
        assert not exists(nb_code) or log2(nb_code).is_integer(), (f'your codebook size must be a power of 2 for lookup free quantization'
                                                                               f' (suggested {2 ** ceil(log2(nb_code))})')

        nb_code = default(nb_code, lambda: 2 ** dim)
        self.nb_code = nb_code

        codebook_dim = int(log2(nb_code))   # code dim is always log2(codebook size)
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale = cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias = projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # whether to use BSQ (binary spherical quantization)

        self.spherical = spherical
        self.maybe_l2norm = (lambda t: l2norm(t) * self.codebook_scale) if spherical else identity

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

        # codes

        all_codes = torch.arange(nb_code)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer('codebook', codebook.float(), persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def dequantize(
        self,
        indices,
        project_out = True
    ):
        """
        :param indices: b n
        :param project_out: whether go through MLP
        :return:
        """

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = self.maybe_l2norm(codes)

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, x):

        x = rearrange(x, 'b d ... -> b ... d')
        x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # maybe l2norm

        x = self.maybe_l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, enabled = False) if force_f32 else nullcontext

        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # quantize by eq 3.

            original_input = x

            codebook_value = torch.ones_like(x) * self.codebook_scale
            quantized = torch.where(x > 0, codebook_value, -codebook_value)

            # calculate indices

            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

            if not self.keep_num_codebooks_dim:
                indices = rearrange(indices, '... 1 -> ...')
        return indices


    def forward(
        self,
        x,
        inv_temperature = 100.,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        :param x: b c t
        :param inv_temperature:
        :return:
        """

        # standardize image or video into (batch, seq, dimension)

        x = rearrange(x, 'b d ... -> b ... d')
        x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # split channels into codebooks
        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # maybe l2norm

        x = self.maybe_l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, enabled = False) if force_f32 else nullcontext

        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # quantize by eq 3.

            original_input = x

            codebook_value = torch.ones_like(x) * self.codebook_scale
            # positive: codebook_scale, negative: -codebook_scale
            # 1 or -1
            quantized = torch.where(x > 0, codebook_value, -codebook_value)

            # calculate indices
            # num of "1" represents the indices
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

            # maybe l2norm

            quantized = self.maybe_l2norm(quantized)

            # use straight-through gradients (optionally with custom activation fn) if training

            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized

            # entropy aux loss

            if self.training:

                if force_f32:
                    codebook = self.codebook.float()

                codebook = self.maybe_l2norm(codebook)

                # Euclidean distance to each code
                distance = -2 * einsum(original_input, codebook, '... i d, j d -> ... i j')

                prob = torch.softmax(-distance * inv_temperature, dim=-1)

                prob = rearrange(prob, 'b n ... -> (b n) ...')

                # whether to only use a fraction of probs, for reducing memory

                if self.frac_per_sample_entropy < 1.:
                    num_tokens = prob.shape[0]
                    num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                    rand_mask = torch.randn(num_tokens).argsort(dim = -1) < num_sampled_tokens
                    per_sample_probs = prob[rand_mask]
                else:
                    per_sample_probs = prob

                # calculate per sample entropy

                per_sample_entropy = entropy(per_sample_probs).mean()

                # distribution over all available tokens in the batch

                avg_prob = reduce(per_sample_probs, '... c d -> c d', 'mean')

                avg_prob = maybe_distributed_mean(avg_prob)

                codebook_entropy = entropy(avg_prob).mean()

                # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
                # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            else:
                # if not training, just return dummy 0
                entropy_aux_loss = self.zero

            # whether to make the entropy loss positive or not through a (shifted) softplus

            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(entropy_aux_loss + self.entropy_loss_offset)

            # commit loss

            if self.training and self.commitment_loss_weight > 0.:

                commit_loss = mse_loss(original_input, quantized.detach(), reduction = 'none')

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # input back to original dtype if needed

            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim

        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        x = unpack_one(x, ps, 'b * d')
        x = rearrange(x, 'b ... d -> b d ...')

        indices = unpack_one(indices, ps, 'b * c')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # complete aux loss

        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        return x, indices, aux_loss, None

    @property
    def codebook_size(self):
        return self.nb_code


if __name__=='__main__':
    motion = torch.rand([2, 512, 16])
    quantizer = LFQ(dim=512, nb_code=1024)
    indices = quantizer.quantize(motion)
    out = quantizer.dequantize(indices)
    print(out.shape)