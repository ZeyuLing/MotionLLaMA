from typing import Optional, Any, Sequence

from mmengine.evaluator import BaseMetric

from mmotion.evaluation.functional.d2m.beat_cover_hit_score import batch_beat_cover_hit_f1_score
from mmotion.registry import METRICS
from mmotion.utils.task.task_lib import Dance2Music


@METRICS.register_module()
class D2MMetric(BaseMetric):
    def __init__(self, pred_key='pred_music',
                 gt_key='music',
                 sr=24000,
                 collect_devices: str = 'cpu',
                 prefix: Optional[str] = None,
                 ):
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.sr = sr
        super().__init__(collect_devices, prefix)

    def compute_metrics(self, results: list) -> dict:
        pred_music = sum([result['pred_music'] for result in results], [])
        gt_music = sum([result['gt_music'] for result in results], [])
        num_not_match = sum(result['not_match'] for result in results)
        bcs, bhs, f1 = batch_beat_cover_hit_f1_score(gt_music, pred_music, self.sr)
        return {
            'beat_cover_score': bcs,
            'beat_hit_score': bhs,
            'beat_f1_score': f1,
            'd2m_not_matched': num_not_match
        }

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        not_match = 0
        batch_gt_music = []
        batch_pred_music = []
        tasks = [data_sample.get('task') for data_sample in data_samples]
        if tasks[0] != Dance2Music:
            return
        for sample in data_samples:
            gt_music = sample.get(self.gt_key)
            pred_music = sample.get(self.pred_key)
            if pred_music is None:
                not_match += 1
            else:
                batch_gt_music.append(gt_music)
                batch_pred_music.append(pred_music)
        result = {
            'gt_music': batch_gt_music,
            'pred_music': batch_pred_music,
            'not_match': not_match
        }
        self.results.append(result)
