import numpy as np
from typing import Any, List, Iterable, Dict

import torch
from einops import pack, unpack, repeat, rearrange
from mmengine.dist import is_distributed, dist
from torch import Tensor
import torch.nn.functional as F
from torch import nn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def l2norm(t):
    return F.normalize(t, dim=-1)


def identity(t):
    return t


class GumbelSoftmax(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.opts = kwargs

    def forward(self, logits, **kwargs):
        opts = dict(self.opts)
        if not self.training:
            opts['noise'] = 0.0
            opts['hard'] = True
        return gumbel_softmax(logits, **opts, **kwargs)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def gumbel_softmax(logits, dim=-1, tau=1.0, noise=1.0, hard=True, **kwargs):
    """
    Softmax with gumbel noise
    :param logits: inputs for softmax
    :param dim: normalize softmax along this dimension
    :param tau: gumbel softmax temperature
    :param hard: if True, works like onehot(sample) during forward pass,
        gumbel-softmax for backward pass
    :return: gumbel-softmax "probabilities", tensor of same shape as logits
    """
    if noise != 0:
        z = gumbel_noise(*logits.shape, device=logits.device, dtype=logits.dtype)
        logits = logits + noise * z
    if tau != 1.0:
        logits /= tau

    probs_gumbel = torch.softmax(logits, dim=dim)

    if hard:
        _, argmax_indices = torch.max(probs_gumbel, dim=dim)
        hard_argmax_onehot = to_one_hot(argmax_indices, depth=logits.shape[dim])
        if dim != -1 and dim != len(logits.shape) - 1:
            new_dim_order = list(range(len(logits.shape) - 1))
            new_dim_order.insert(dim, -1)
            hard_argmax_onehot = hard_argmax_onehot.permute(*new_dim_order)

        # forward pass: onehot sample, backward pass: gumbel softmax
        probs_gumbel = (hard_argmax_onehot - probs_gumbel).detach() + probs_gumbel

    return probs_gumbel


def to_one_hot(y, depth=None):
    """
    Takes integer with n dims and converts it to 1-hot representation with n + 1 dims.
    The n+1'st dimension will have zeros everywhere but at y'th index, where it will be equal to 1.
    Args:
        y: input integer (IntTensor, LongTensor or Variable) of any shape
        depth (int):  the size of the one hot dimension
    """
    y_flat = y.to(torch.int64).view(-1, 1)
    depth = depth if depth is not None else int(torch.max(y_flat)) + 1
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(1, y_flat, 1)
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    return y_one_hot


def gumbel_noise(*sizes, epsilon=1e-9, **kwargs):
    """ Sample noise from gumbel distribution """
    return -torch.log(-torch.log(torch.rand(*sizes, **kwargs) + epsilon) + epsilon)


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def default(val: Any, d: Any) -> Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    """ use kmeans to cluster the input tokens into codebook_size clusters.
    :param samples: num_codes code_dim
    :param num_clusters: codebook size as usual
    :param num_iters:
    :return:
    """
    dim, dtype = samples.shape[-1], samples.dtype
    # initial cluster center
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        samples_norm = (samples ** 2).sum(dim=-1, keepdim=True)  # (n, 1)

        means_norm = (means ** 2).sum(dim=-1, keepdim=True).T  # (1, c)

        dot_product = samples @ means.T  # (n, c)

        dists = - (samples_norm + means_norm - 2 * dot_product)
        # diffs = rearrange(samples, "n d -> n () d") - rearrange(
        #     means, "c d -> () c d"
        # )
        # dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_distributed():
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _check_number_of_params(params: List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                           "at least one worker has a different one.")


def broadcast_tensors(tensors: Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def sync_buffer(buffers, average=True):
    """
    Sync grad for buffers. If average is False, broadcast instead of averaging.
    """
    if not is_distributed():
        return
    handles = []
    for buffer in buffers:
        if torch.is_floating_point(buffer.data):
            if average:
                handle = torch.distributed.all_reduce(
                    buffer.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            else:
                handle = torch.distributed.broadcast(
                    buffer.data, src=0, async_op=True)
            handles.append((buffer, handle))
    for buffer, handle in handles:
        handle.wait()
        if average:
            buffer.data /= world_size


def sync_grad(params):
    """
    Simpler alternative to DistributedDataParallel, that doesn't rely
    on any black magic. For simple models it can also be as fast.
    Just call this on your model parameters after the call to backward!
    """
    if not is_distributed():
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handle = torch.distributed.all_reduce(
                p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            handles.append((p, handle))
    for p, handle in handles:
        handle.wait()
        p.grad.data /= world_size()


def average_metrics(metrics: Dict[str, float], count=1.):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))
