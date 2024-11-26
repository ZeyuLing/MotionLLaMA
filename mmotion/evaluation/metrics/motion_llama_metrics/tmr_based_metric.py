import os
from typing import Dict, Optional

import torch
from mmengine import Config
from mmengine.evaluator import BaseMetric
from mmengine.model import is_model_wrapper
from torch.utils.data import DataLoader
from torch import nn

from mmotion.evaluation.functional.t2m.matching_score_precision import euclidean_distance_matrix, calculate_top_k
from mmotion.models import TMR
from mmotion.registry import MODELS

import torch.nn.functional as F


class BaseTMRMetric(BaseMetric):
    def __init__(self,
                 tmr_model: Dict,
                 dtype=torch.bfloat16,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super(BaseTMRMetric, self).__init__(collect_device, prefix)
        self.dtype = dtype
        self.tmr_model: TMR = self.build_tmr(tmr_model).to(dtype)

    def build_tmr(self, tmr_cfg) -> TMR:
        """
        :param vqvae_cfg: vqvae config
        :return: Vqvae module.
        """
        type = tmr_cfg['type']
        if os.path.isfile(type):
            # allow using config file path as type, simplify config writing.
            init_cfg = tmr_cfg.pop('init_cfg', None)
            tmr_cfg = Config.fromfile(type)['model']
            if init_cfg is not None:
                tmr_cfg['init_cfg'] = init_cfg

        tmr: TMR = MODELS.build(tmr_cfg).eval().cuda()
        if tmr_cfg.get('init_cfg', None) is not None:
            tmr.init_weights()
        self.data_preprocessor = tmr.data_preprocessor
        return tmr



    def evaluate(self, size=None) -> dict:
        return super().evaluate(size or self.size)

    def prepare(self, module: nn.Module, dataloader: DataLoader):
        self.size = len(dataloader.dataset)
        if is_model_wrapper(module):
            module = module.module
        self.data_preprocessor = module.data_preprocessor

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader
