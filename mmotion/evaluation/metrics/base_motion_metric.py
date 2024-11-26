from typing import Optional

from mmotion.evaluation.metrics.base_sample_wise_metric import BaseSampleWiseMetric
from mmotion.evaluation.metrics.metrics_utils import obtain_data
from mmotion.utils.typing import SampleList


class BaseMotionMetric(BaseSampleWiseMetric):
    def __init__(self,
                 gt_key: str = 'gt_motion',
                 pred_key: str = 'pred_motion',
                 scaling=1,
                 device='cpu',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 has_hand_key: str = 'has_hand',
                 rot_type: str = 'quaternion') -> None:
        assert self.metric is not None, (
            '\'metric\' must be defined for \'BaseSampleWiseMetric\'.')
        super().__init__(gt_key, pred_key, scaling, device, collect_device, prefix)

        self.has_hand_key = has_hand_key
        self.rot_type = rot_type

    def process(self, data_batch, data_samples: SampleList):

        for data in data_samples:
            has_hand = data.get(self.has_hand_key, True)
            gt = obtain_data(data, self.gt_key, self.device)
            if self.pred_key in data.keys():
                pred = obtain_data(data, self.pred_key, self.device)
            else:
                pred = obtain_data(data['output'], self.pred_key, self.device)

            result = self.process_result(gt, pred, has_hand)
            if isinstance(result, dict):
                self.results.append({**result})
            else:
                self.results.append({self.metric: result})
