from einops import rearrange
from kornia.filters import laplacian_1d
from torch import nn
import torch.nn.functional as F

from mmotion.registry import MODELS


@MODELS.register_module()
class LaplacianLoss(nn.Module):
    def __init__(self, kernel_size: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.laplacian_kernel = laplacian_1d(kernel_size)
        self.laplacian_kernel.requires_grad = False

    def forward(self, gt, pred):
        """
        :param gt: b t c
        :param pred: b t c
        :return:
        """
        gt = rearrange(gt, 'b t c -> b c t')
        gt_lap = F.conv1d(gt,self.laplacian_kernel.to(gt))
        pred_lap = F.conv1d(pred, self.laplacian_kernel.to(pred))

        return F.l1_loss(gt_lap, pred_lap, reduction='mean')



