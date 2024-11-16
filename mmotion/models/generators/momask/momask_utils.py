import torch
import math
from einops import rearrange
import torch.nn.functional as F


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def cal_performance(pred_logits, labels, ignore_index=None):
    """
    :param pred_logits: predicted logits. b n c
    :param labels: real index. b n
    :param ignore_index: pad id
    :return:
    """
    loss = F.cross_entropy(pred_logits.permute(0, 2, 1), labels, ignore_index=ignore_index)
    pred_id = pred_logits.argmax(dim=-1)
    mask = labels.ne(ignore_index)
    n_correct = pred_id.eq(labels).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc


def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


def top_k(logits: torch.Tensor, thres=0.9, dim=1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim=dim)
    probs = torch.full_like(logits, float('-inf'), dtype=logits.dtype)
    probs.scatter_(dim, ind, val)
    return probs


def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise))


def gumbel_sample(t, temperature=1., dim=1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)
