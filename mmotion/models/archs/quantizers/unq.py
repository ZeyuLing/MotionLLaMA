
import numpy as np
import os
import sys
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, unpack, rearrange
from einx import get_at

sys.path.append(os.curdir)
from mmotion.models.archs.quantizers.utils import Lambda
from mmotion.registry import MODELS

class BatchNorm(nn.BatchNorm1d):
    """ Batch Normalization that always normalizes over last dim """
    def forward(self, x):
        return super().forward(x.view(-1, x.shape[-1])).view(*x.shape)

class Feedforward(nn.Sequential):
    def __init__(self, input_dim, hidden_dim=None, num_layers=2, output_dim=None,
                 BatchNorm=BatchNorm, Activation=nn.ReLU, bias=True, **kwargs):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or hidden_dim
        for i in range(num_layers):
            self.add_module('layer%i' % i, nn.Linear(
                input_dim if i == 0 else hidden_dim,
                output_dim if i == (num_layers - 1) else hidden_dim,
                bias=bias))
            self.add_module('layer%i_bn' % i, BatchNorm(hidden_dim))
            self.add_module('layer%i_activation' % i, Activation())

@MODELS.register_module()
class UNQ(nn.Module):
    def __init__(self, dim, num_codebooks=8, nb_code=256, hidden_dim=1024, bottleneck_dim=256,
                 encoder_layers=2, decoder_layers=2, initial_entropy=1.0, key_dim=None,
                 decouple_temperatures=True, init_codes_with_data=True, tau=1., hard=True):
        """
        Multi-codebook quantization that supports nonlinear encoder/decoder transformations
        :param dim: size of vectors to be quantized
        :param num_codebooks: the number of discrete spaces to quantize data vectors into
        :param nb_code: the number of vectors in each codebook
        :param initial_entropy: starting entropy value for data-aware initialization
        :param init_codes_with_data: if specified, codes will be initialized with random data vectors
        :param decouple_temperatures: if True, does not apply temperatures to logits when computing quantized distances
        """
        super().__init__()
        key_dim = key_dim or dim
        self.num_codebooks, self.nb_code = num_codebooks, nb_code
        self.decouple_temperatures = decouple_temperatures
        self.encoder = nn.Sequential(
            Feedforward(dim, hidden_dim, num_layers=encoder_layers),
            nn.Linear(hidden_dim, num_codebooks * bottleneck_dim),
            Lambda(lambda x: x.view(*x.shape[:-1], num_codebooks, bottleneck_dim))
        )

        self.codebook = nn.Parameter(torch.randn(num_codebooks, nb_code, key_dim))

        self.decoder = nn.Sequential()
        self.decoder.add_module('reshape', Lambda(lambda x: x.view(*x.shape[:-2], -1)))

        self.decoder.add_module('embed', Lambda(
                lambda x: x @ self.codebook.to(device=x.device).view(num_codebooks * nb_code, bottleneck_dim)))
        self.decoder.add_module('batchnorm', BatchNorm(bottleneck_dim))
        self.decoder.add_module('ffn', Feedforward(bottleneck_dim, hidden_dim, num_layers=decoder_layers))
        self.decoder.add_module('final', nn.Linear(hidden_dim, dim))

        self.log_temperatures = nn.Parameter(data=torch.zeros(num_codebooks) * float('nan'), requires_grad=True)
        self.initial_entropy, self.init_codes_with_data = initial_entropy, init_codes_with_data
        self.gumbel_softmax = partial(F.gumbel_softmax, tau=tau, hard=hard)

    def compute_logits(self, x, add_temperatures=True):
        """ Computes logits for code probabilities [batch_size, num_codebooks, nb_code] """
        assert len(x.shape) >= 2, "x should be of shape [..., vector_dim]"
        if len(x.shape) > 2:
            flat_logits = self.compute_logits(x.view(-1, x.shape[-1]), add_temperatures=add_temperatures)
            return flat_logits.view(*x.shape[:-1], self.num_codebooks, self.nb_code)

        # einsum: [b]atch_size, [n]um_codebooks, [c]odebook_size, [v]ector_dim
        logits = torch.einsum('bnd,ncd->bnc', self.encoder(x), self.codebook)

        if add_temperatures:
            if not self.is_initialized(): self.initialize(x)
            logits *= torch.exp(-self.log_temperatures[:, None])
        return logits

    def forward(self, x):
        """
        :param x: b c t
        :return:
        """
        """ quantizes x into one-hot codes and restores original data """
        x = rearrange(x, 'b c t -> b t c')
        if not self.is_initialized():
            self.initialize(x)
        logits_raw = self.compute_logits(x, add_temperatures=False)
        indices = logits_raw.argmax(dim=-1)
        logits = logits_raw * torch.exp(-self.log_temperatures[:, None])
        codes = self.gumbel_softmax(logits, dim=-1)  # [..., num_codebooks, nb_code]
        x_reco = self.decoder(codes)
        commit_loss = F.mse_loss(x_reco, x)

        distances_to_codes = - (logits_raw if self.decouple_temperatures else logits)
        return x_reco, indices, commit_loss, dict(x=x, logits=logits, codes=codes,
                                                  x_reco=x_reco,distances_to_codes=distances_to_codes)

    def quantize(self, x):
        x = rearrange(x, 'b c t -> b t c')
        """ encodes x into uint8 codes[b, n, num_codebooks] """
        assert False, self.compute_logits(x, add_temperatures=False).shape
        return self.compute_logits(x, add_temperatures=False).argmax(dim=-1)

    def dequantize(self, indices):
        if len(indices.shape)==2:
            indices = indices[..., None]
        batch, num_layers = indices.shape[0], indices.shape[-1]
        indices, ps = pack([indices], 'b * q')
        logits = get_at('q [c] d, b n q -> q b n d',
                           self.codebook[:num_layers], indices)
        logits, = unpack(logits, ps, 'q b * d')
        logits = logits * torch.exp(-self.log_temperatures[:, None])
        codes = self.gumbel_softmax(logits, dim=-1)  # [..., num_codebooks, nb_code]
        x_reco = self.decoder(codes)
        return x_reco


    def is_initialized(self):
        # note: can't set this as property because https://github.com/pytorch/pytorch/issues/13981
        return torch.all(torch.isfinite(self.log_temperatures.data))

    def initialize(self, x):
        """ Initialize codes and log_temperatures given data """
        with torch.no_grad():
            if self.init_codes_with_data:
                chosen_ix = torch.randint(0, x.shape[0], size=[self.nb_code * self.num_codebooks], device=x.device)
                chunk_ix = torch.arange(self.nb_code * self.num_codebooks, device=x.device) // self.nb_code
                # [b, n, n_q, d]
                x_encode = self.encoder(x)
                # nb_code * num_codebooks, num_codebooks, hidden_dim
                initial_keys = x_encode[chosen_ix, chunk_ix]
                assert False, initial_keys.shape
                # .view(*self.codebook.shape)
                self.codebook.data[:] = initial_keys

            base_logits = self.compute_logits(
                x, add_temperatures=False).view(-1, self.num_codebooks, self.nb_code)
            # ^-- [batch_size, num_codebooks, nb_code]

            log_temperatures = torch.tensor([
                fit_log_temperature(codebook_logits, target_entropy=self.initial_entropy, tolerance=1e-2)
                for codebook_logits in base_logits.transpose(1, 0, 2)
            ]).to(x)
            self.log_temperatures.data[:] = log_temperatures


def fit_log_temperature(logits, target_entropy=1.0, tolerance=1e-6, max_steps=100,
                        lower_bound=math.log(1e-9), upper_bound=math.log(1e9)):
    """
    Returns a temperature s.t. the average entropy equals mean_entropy (uses bin-search)
    :param logits: unnormalized log-probabilities, [batch_size, num_outcomes]
    :param target_entropy: target entropy to fit
    :returns: temperature (scalar) such that
        probs = exp(logits / temperature) / sum(exp(logits / temperature), axis=-1)
        - mean(sum(probs * log(probs), axis=-1)) \approx mean_entropy
    """
    assert isinstance(logits, np.ndarray)
    assert logits.ndim == 2
    assert 0 < target_entropy < np.log(logits.shape[-1])
    assert lower_bound < upper_bound
    assert np.isfinite(lower_bound) and np.isfinite(upper_bound)

    log_tau = (lower_bound + upper_bound) / 2.0

    for i in range(max_steps):
        # check temperature at the geometric mean between min and max values
        log_tau = (lower_bound + upper_bound) / 2.0
        tau_entropy = _entropy_with_logits(logits, log_tau)

        if abs(tau_entropy - target_entropy) < tolerance:
            break
        elif tau_entropy > target_entropy:
            upper_bound = log_tau
        else:
            lower_bound = log_tau
    return log_tau


def _entropy_with_logits(logits, log_tau=0.0, axis=-1):
    logits = np.copy(logits)
    logits -= np.max(logits, axis, keepdims=True)
    logits *= np.exp(-log_tau)
    exps = np.exp(logits)
    sum_exp = exps.sum(axis)
    entropy_values = np.log(sum_exp) - (logits * exps).sum(axis) / sum_exp
    return np.mean(entropy_values)


def compute_penalties(logits, individual_entropy_coeff=0.0, allowed_entropy=0.0, global_entropy_coeff=0.0,
                      cv_coeff=0.0, square_cv=True, eps=1e-9):
    """
    Computes typical regularizers for gumbel-softmax quantization
    Regularization is of slight help when performing hard quantization, but it isn't critical
    :param logits: tensor [batch_size, ..., nb_code]
    :param individual_entropy_coeff: penalizes mean individual entropy
    :param allowed_entropy: does not penalize individual_entropy if it is below this value
    :param cv_coeff: penalizes squared coefficient of variation
    :param global_entropy_coeff: coefficient for entropy of mean probabilities over batch
        this value should typically be negative (e.g. -1), works similar to cv_coeff
    """
    counters = dict(reg=torch.tensor(0.0, dtype=torch.float32, device=logits.device))
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    # [batch_size, ..., nb_code]

    if individual_entropy_coeff != 0:
        individual_entropy_values = - torch.sum(p * logp, dim=-1)
        clipped_entropy = F.relu(allowed_entropy - individual_entropy_values + eps).mean()
        individual_entropy = (individual_entropy_values.mean() - clipped_entropy).detach() + clipped_entropy

        counters['reg'] += individual_entropy_coeff * individual_entropy
        counters['individual_entropy'] = individual_entropy

    if global_entropy_coeff != 0:
        global_p = torch.mean(p, dim=0)  # [..., nb_code]
        global_logp = torch.logsumexp(logp, dim=0) - np.log(float(logp.shape[0]))  # [..., nb_code]
        global_entropy = - torch.sum(global_p * global_logp, dim=-1).mean()
        counters['reg'] += global_entropy_coeff * global_entropy
        counters['global_entropy'] = global_entropy

    if cv_coeff != 0:
        load = torch.mean(p, dim=0)  # [..., nb_code]
        mean = load.mean()
        variance = torch.mean((load - mean) ** 2)
        if square_cv:
            counters['cv_squared'] = variance / (mean ** 2 + eps)
            counters['reg'] += cv_coeff * counters['cv_squared']
        else:
            counters['cv'] = torch.sqrt(variance + eps) / (mean + eps)
            counters['reg'] += cv_coeff * counters['cv']

    return counters


if __name__=='__main__':
    motion = torch.rand([2, 512 , 16])
    quantizer = UNQ(dim=512, nb_code=1024)
    out = quantizer.quantize(motion)
    print(out)