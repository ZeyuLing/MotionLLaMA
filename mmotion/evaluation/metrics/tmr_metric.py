from typing import Dict, Optional, List

import torch
from mmengine.evaluator import BaseMetric
from mmengine.model import is_model_wrapper

from torch.utils.data import DataLoader

from mmotion.evaluation.functional.t2m.diversity import cal_diversity
from mmotion.evaluation.functional.t2m.matching_score_precision import euclidean_distance_matrix, calculate_top_k
from mmotion.registry import METRICS

import torch.nn.functional as F
from torch import nn

from mmotion.utils.typing import SampleList


@METRICS.register_module(force=True)
class TMRMetric(BaseMetric):
    """
        For TMR Model evaluation
        Including R-precision(Top-k)
        Multimodal distance
        FID
        diversity
    """

    def __init__(self,
                 text_key: str = 'lat_text',
                 motion_key: str = 'lat_motion',
                 top_k=3,
                 r_precision_batch: int = 256,
                 diversity_times: int = 300,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 ):
        super(TMRMetric, self).__init__(collect_device, prefix)
        self.text_key = text_key
        self.motion_key = motion_key
        self.top_k = top_k
        self.r_precision_batch = r_precision_batch
        self.diversity_times = diversity_times

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        mm_dist = 0
        top_k_mat = torch.zeros((self.top_k,))
        top_k_mat_m2t = torch.zeros((self.top_k,))

        all_text_embeddings = torch.cat([result['text_embedding'] for result in results], dim=0)
        all_motion_embeddings = torch.cat([result['motion_embedding'] for result in results], dim=0)
        num_samples = all_text_embeddings.shape[0]
        valid_num_samples = num_samples // self.r_precision_batch * self.r_precision_batch

        shuffle_idx = torch.randperm(num_samples)
        all_text_embeddings = all_text_embeddings[shuffle_idx]
        all_motion_embeddings = all_motion_embeddings[shuffle_idx]

        for i in range(num_samples // self.r_precision_batch):
            group_texts = F.normalize(all_text_embeddings[i * self.r_precision_batch: (i + 1) * self.r_precision_batch])
            group_motions = F.normalize(
                all_motion_embeddings[i * self.r_precision_batch: (i + 1) * self.r_precision_batch])

            dist_mat = euclidean_distance_matrix(
                group_texts, group_motions
            )

            mm_dist += dist_mat.trace()

            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

            argsmax_m2t = torch.argsort(dist_mat.T, dim=1)
            top_k_mat_m2t += calculate_top_k(argsmax_m2t, top_k=self.top_k).sum(axis=0)

        diversity = cal_diversity(all_motion_embeddings, self.diversity_times)
        diversity_text = cal_diversity(all_text_embeddings, self.diversity_times)
        res = {
            "mm_dist": mm_dist / valid_num_samples,
            "diversity": diversity,
            "diversity_text": diversity_text
        }
        for k in range(self.top_k):
            res[f'r_precision_top_{k + 1}'] = top_k_mat[k] / valid_num_samples
            res[f'm2t_r_precision_top_{k + 1}'] = top_k_mat_m2t[k] / valid_num_samples

        return res

    def process(self, data_batch, data_samples: SampleList):
        """
        :param data_batch:
        :param data_samples: output of model.forward_predict
        :return:
        """

        text_embedding = torch.stack([data_sample.get(self.text_key) for data_sample in data_samples], dim=0)
        motion_embedding = torch.stack([data_sample.get(self.motion_key) for data_sample in data_samples], dim=0)
        result = {
            'text_embedding': text_embedding,
            'motion_embedding': motion_embedding
        }

        self.results.append({**result})

    def evaluate(self, size=None) -> dict:
        # assert hasattr(self, 'size'), (
        #     'Cannot find \'size\', please make sure \'self.prepare\' is '
        #     'called correctly.')
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
