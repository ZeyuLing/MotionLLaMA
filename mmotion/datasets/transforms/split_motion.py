from typing import Union, List

from mmcv import BaseTransform

from mmotion.registry import TRANSFORMS
from mmotion.utils.task.task_lib import MotionInbetween, MotionPrediction


@TRANSFORMS.register_module()
class SplitPrediction(BaseTransform):
    def __init__(self, keys: Union[str, List[str]] = 'motion', past_ratio: float = 0.4):
        assert keys, 'Keys should not be empty.'
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys
        self.past_ratio = past_ratio

    def split_past_future(self, motion):
        num_frames = len(motion)
        past_frames = int(num_frames * self.past_ratio)
        return motion[:past_frames], motion[past_frames:]

    def transform(self, results):
        if results['task'] not in [MotionPrediction]:
            return results
        for key in self.keys:
            if key in results:
                results[f'past_{key}'], results[f'future_{key}'] = self.split_past_future(results[key])
                assert len(results[f'past_{key}']) > 0 and len(results[f'future_{key}']) > 0
                results[f'past_{key}_num_frames'] = len(results[f'past_{key}'])
                results[f'future_{key}_num_frames'] = len(results[f'future_{key}'])
        results['past_ratio'] = self.past_ratio

        return results


@TRANSFORMS.register_module()
class SplitInbetween(BaseTransform):
    def __init__(self, keys: Union[str, List[str]] = 'motion',
                 past_ratio: float = 0.2,
                 future_ratio: float = 0.2):
        assert keys, 'Keys should not be empty.'
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys
        self.past_ratio = past_ratio
        self.future_ratio = future_ratio

    def split_past_middle_future(self, motion):
        num_frames = len(motion)
        past_frames = int(num_frames * self.past_ratio)
        future_frames = int(num_frames * self.future_ratio)
        return motion[:past_frames], motion[past_frames:-future_frames], motion[-future_frames:]

    def transform(self, results):
        if results['task'] not in [MotionInbetween]:
            return results
        for key in self.keys:
            if key in results:
                results[f'past_{key}'], results[f'middle_{key}'], results[f'future_{key}'] \
                    = self.split_past_middle_future(results[key])
                assert len(results[f'past_{key}']) > 0 and len(results[f'middle_{key}']) > 0 and len(
                    results[f'future_{key}']) > 0
        results['past_ratio'] = self.past_ratio
        results['future_ratio'] = self.future_ratio
        return results
