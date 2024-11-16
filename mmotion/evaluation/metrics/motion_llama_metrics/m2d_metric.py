from typing import Optional, List

from mmengine.evaluator import BaseMetric
from mmengine.model import is_model_wrapper
from torch.utils.data import DataLoader
from torch import nn
import torch

from mmotion.evaluation.functional.m2d.beat_alignment import extract_music_beats, \
    extract_dance_beats, cal_beat_align, cal_hungarian_beat_align
from mmotion.evaluation.functional.m2d.diversity import cal_diversity_dance
from mmotion.evaluation.functional.m2d.kinetic_feature import extract_kinetic_features
from mmotion.evaluation.functional.t2m.fid import cal_fid
from mmotion.registry import METRICS
from mmotion.utils.task.task_lib import Music2Dance, CaptionMusic2Dance
from mmotion.utils.typing import SampleList


def normalize(feat, feat2):
    mean = feat.mean(dim=0)
    std = feat.std(dim=0)

    return (feat - mean) / (std + 1e-6), (feat2 - mean) / (std + 1e-6)


@METRICS.register_module()
class M2DMetric(BaseMetric):
    def __init__(self,
                 music_key: str = 'music',
                 motion_key: str = 'joints',
                 pred_motion_key: str = 'pred_joints',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 full_body=True):
        super().__init__(collect_device, prefix)
        self.pred_motion_key = pred_motion_key
        self.motion_key = motion_key
        self.music_key = music_key
        self.full_body = full_body

    def compute_metrics(self, results: List):
        all_motion_kinetic_features = torch.cat([result['motion_kinetic_feature'] for result in results], dim=0)
        all_pred_motion_kinetic_features = torch.cat([result['pred_motion_kinetic_feature'] for result in results],
                                                     dim=0)
        all_motion_kinetic_features, all_pred_motion_kinetic_features = normalize(
            all_motion_kinetic_features, all_pred_motion_kinetic_features
        )
        all_motion_beats = sum([result['motion_beat'] for result in results], [])
        all_pred_motion_beats = sum([result['pred_motion_beat'] for result in results], [])
        all_music_beats = sum([result['music_beat'] for result in results], [])

        num_samples = all_motion_kinetic_features.shape[0]
        fid_k = cal_fid(all_pred_motion_kinetic_features,
                        all_motion_kinetic_features)
        ba_gt = [cal_beat_align(motion_beat, music_beat) for motion_beat, music_beat in
                 zip(all_motion_beats, all_music_beats)]
        ba = [cal_beat_align(motion_beat, music_beat) for motion_beat, music_beat in
              zip(all_pred_motion_beats, all_music_beats)]
        diversity = cal_diversity_dance(all_pred_motion_kinetic_features)
        diversity_gt = cal_diversity_dance(all_motion_kinetic_features)

        res = {
            'fid_k': fid_k,
            'beat_alignment': sum(ba) / num_samples,
            'beat_alignment_gt': sum(ba_gt) / num_samples,
            'num_samples': num_samples,
            'diversity': diversity,
            'diversity_gt': diversity_gt
        }

        return res

    def process(self, data_batch, data_samples: SampleList):
        fps = [data_sample.get('fps') for data_sample in data_samples][0]
        self.fps = fps
        tasks = [data_sample.get('task') for data_sample in data_samples]
        assert all([task == tasks[0] for task in tasks])
        if tasks[0] not in [Music2Dance, CaptionMusic2Dance]:
            return

        music_beat = [extract_music_beats(data_sample.get(self.music_key),
                                          sr=data_sample.get('sr')) for data_sample in data_samples]
        motion = [data_sample.get(self.motion_key) for data_sample in data_samples]
        pred_motion = [data_sample.get(self.pred_motion_key) for data_sample in data_samples]
        if not self.full_body:
            motion = [m[:, :22] for m in motion]
            pred_motion = [m[:, :22] for m in pred_motion]
        motion_beat = [extract_dance_beats(m)[0] for m in motion]
        pred_motion_beat = [extract_dance_beats(m)[0] for m in pred_motion]

        motion_kinetic_feature = torch.stack([extract_kinetic_features(m) for m in motion], dim=0)
        pred_motion_kinetic_feature = torch.stack([extract_kinetic_features(m) for m in pred_motion])

        result = {
            'music_beat': music_beat,
            'motion': motion,
            'pred_motion': pred_motion,
            'motion_beat': motion_beat,
            'pred_motion_beat': pred_motion_beat,
            'motion_kinetic_feature': motion_kinetic_feature,
            'pred_motion_kinetic_feature': pred_motion_kinetic_feature,
        }

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
