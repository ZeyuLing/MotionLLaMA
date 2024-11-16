from collections import defaultdict
from typing import Optional, List

from mmengine.evaluator import BaseMetric
from mmengine.model import is_model_wrapper
from torch.utils.data import DataLoader
from torch import nn

from mmotion.registry import METRICS
from mmotion.utils.typing import SampleList


@METRICS.register_module()
class ValidationMetric(BaseMetric):
    def __init__(self,
                 loss_key: str = 'validation_loss',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None, ):
        self.loss_key = loss_key
        super(ValidationMetric, self).__init__(prefix=prefix, collect_device=collect_device)

    def compute_metrics(self, results: List):
        all_metrics = defaultdict(list)
        for result in results:
            for key in result.keys():
                all_metrics[key].append(result[key])

        counter = {}
        for key, value in all_metrics.items():
            value = sum(value, [])
            task_samples = len(value)
            value = sum(value) / task_samples
            all_metrics[key] = value
            counter[f"{key}_num_samples"] = task_samples
        all_metrics.update(counter)
        return all_metrics

    def process(self, data_batch, data_samples: SampleList):
        val_loss = []
        result = defaultdict(list)
        for data_sample in data_samples:
            task = data_sample.get('task').abbr
            loss = data_sample.get(self.loss_key)
            result[task].append(loss)
            result['val_loss'].append(loss)
            val_loss.append(val_loss)

        self.results.append(result)

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
