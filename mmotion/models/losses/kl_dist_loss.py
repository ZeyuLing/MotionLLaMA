import torch

from mmotion.registry import MODELS


@MODELS.register_module()
class KLLoss:
    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"