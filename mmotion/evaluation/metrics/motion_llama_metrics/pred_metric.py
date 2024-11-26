from typing import Dict, Optional, List

import torch
from einops import repeat
from torch.nn.functional import normalize

from mmotion.evaluation.functional.keypoint_eval import average_distance_error, final_distance_error
from mmotion.evaluation.functional.t2m.diversity import cal_diversity
from mmotion.evaluation.functional.t2m.fid import cal_fid
from mmotion.evaluation.metrics.motion_llama_metrics.tmr_based_metric import BaseTMRMetric
from mmotion.registry import METRICS
from mmotion.utils.task.task_lib import MotionPrediction
from mmotion.utils.typing import SampleList


@METRICS.register_module()
class PredMetric(BaseTMRMetric):
    def __init__(self,
                 tmr_model: Dict,
                 motion_key='motion',
                 joints_key='joints',
                 pred_motion_key='pred_motion',
                 pred_joints_key='future_joints',
                 dtype=torch.bfloat16,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        self.motion_key = motion_key
        self.joints_key = joints_key
        self.pred_motion_key = pred_motion_key
        self.pred_joints_key = pred_joints_key
        super().__init__(tmr_model, dtype, collect_device, prefix)

    def compute_metrics(self, results: list) -> Dict:
        all_not_match = sum(result['pred_not_match'] for result in results)
        all_joints = sum([result['joints'] for result in self.results], [])
        all_pred_joints = sum([result['pred_joints'] for result in self.results], [])
        all_motion_embeddings = torch.cat([result['motion_embedding'] for result in results],
                                          dim=0)
        all_pred_motion_embeddings = torch.cat([result['pred_motion_embedding'] for result in results], dim=0)

        num_pred_samples = all_pred_motion_embeddings.shape[0]

        fid = cal_fid(normalize(all_pred_motion_embeddings),
                      normalize(all_motion_embeddings))

        diversity = cal_diversity(all_pred_motion_embeddings)
        diversity_gt = cal_diversity(all_motion_embeddings)
        ade, fde = self.cal_ade_fde(all_joints, all_pred_joints)
        res = {
            "pred_fid": fid,
            "pred_diversity": diversity,
            "pred_diversity_gt": diversity_gt,
            "num_pred_samples": num_pred_samples,
            "not_matched_pred_sample": all_not_match,
            'pred_ade': ade,
            'pred_fde': fde
        }
        return res

    def cal_ade_fde(self, joints: List[torch.Tensor], pred_joints: List[torch.Tensor]):
        ade = 0.
        fde = 0.
        for gt, pred in zip(joints, pred_joints):
            if pred.shape[0] < gt.shape[0]:
                pad_frames = gt.shape[0] - pred.shape[0]
                pad = repeat(pred[-1], 'j c -> t j c', t=pad_frames)
                pred = torch.cat([pred, pad], dim=0)
            else:
                pred = pred[:gt.shape[0]]
            ade += average_distance_error(pred, gt, reduction='mean')
            fde += final_distance_error(pred, gt, reduction='mean')
        return ade / len(joints), fde / len(joints)

    def process(self, data_batch, data_samples: SampleList):
        not_match = 0
        batch_motion = []
        batch_pred_motion = []
        batch_joints = []
        batch_pred_joints = []
        for data_sample in data_samples:
            task = data_sample.get('task')
            if task not in [MotionPrediction]:
                continue
            motion = data_sample.get(self.motion_key, None)
            pred_motion = data_sample.get(self.pred_motion_key, None)
            joints = data_sample.get(self.joints_key)
            pred_joints = data_sample.get(self.pred_joints_key)

            if motion is not None:
                batch_motion.append(motion.to(self.dtype).cuda())
                batch_joints.append(joints)
            if pred_motion is None:
                not_match += 1
            if pred_motion is not None:
                batch_pred_motion.append(pred_motion.to(self.dtype).cuda())
                batch_pred_joints.append(pred_joints)

        result = {}
        if len(batch_motion) == 0:
            return
        batch_motion, _ = self.data_preprocessor._do_norm(batch_motion)

        result['motion_embedding'] = self.tmr_model.encode_motion(batch_motion)[1]
        result['joints'] = batch_joints

        if len(batch_pred_motion):
            batch_pred_motion, _ = self.data_preprocessor._do_norm(batch_pred_motion)
            result['pred_motion_embedding'] = self.tmr_model.encode_motion(batch_pred_motion)[1]
            result['pred_joints'] = batch_pred_joints
        result['pred_not_match'] = not_match
        self.results.append(result)
