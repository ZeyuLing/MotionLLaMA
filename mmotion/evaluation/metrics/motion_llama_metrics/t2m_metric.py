from typing import Dict, List, Optional

import torch.nn.functional as F
from mmotion.evaluation.functional.t2m.diversity import cal_diversity
from mmotion.evaluation.functional.t2m.fid import cal_fid
from mmotion.evaluation.functional.t2m.matching_score_precision import cal_mmdist_rprecision
from mmotion.evaluation.metrics.motion_llama_metrics.tmr_based_metric import BaseTMRMetric
from mmotion.registry import METRICS
import torch
from mmotion.utils.task.task_lib import Caption2Motion
from mmotion.utils.typing import SampleList


@METRICS.register_module()
class T2MMetric(BaseTMRMetric):
    def __init__(self,
                 tmr_model: Dict,
                 dtype=torch.bfloat16,
                 text_key: str = 'caption',
                 motion_key: str = 'motion',
                 pred_motion_key: str = 'pred_motion',
                 top_k=3,
                 r_precision_batch: int = 256,
                 diversity_times: int = 300,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None, ):
        """ For single-person t2m and m2t evaluation.
        If caption is not provided, fid and diversity of pred motions will be evaluated
        including following metrics:
        FID,
        Matching Score, R-precision,
        Diversity,

        :param tmr_model: TMR
        :param pred_motion_key: pred motion key
        :param motion_key: gt motion key
        :param text_key: gt caption key
        """
        super(T2MMetric, self).__init__(tmr_model, dtype, collect_device, prefix)
        self.pred_motion_key = pred_motion_key
        self.motion_key = motion_key
        self.text_key = text_key
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
        all_not_matched_sample = sum(result['not_matched_sample'] for result in results)
        all_text_embeddings = torch.cat([result['text_embedding'] for result in results], dim=0)
        all_motion_embeddings = torch.cat([result['motion_embedding'] for result in results], dim=0)
        all_pred_motion_embeddings = torch.cat([result['pred_motion_embedding'] for result in results], dim=0)

        num_samples = all_text_embeddings.shape[0]
        valid_num_samples = num_samples // self.r_precision_batch * self.r_precision_batch

        shuffle_idx = torch.randperm(num_samples)
        all_text_embeddings = all_text_embeddings[shuffle_idx]
        all_motion_embeddings = all_motion_embeddings[shuffle_idx]
        all_pred_motion_embeddings = all_pred_motion_embeddings[shuffle_idx]

        # matching score and r-precision calculation
        mm_dist, top_k_mat = cal_mmdist_rprecision(
            all_pred_motion_embeddings, all_text_embeddings,
            self.r_precision_batch, self.top_k, reduction=True)

        fid = cal_fid(F.normalize(all_pred_motion_embeddings),
                      F.normalize(all_motion_embeddings))
        diversity = cal_diversity(all_pred_motion_embeddings, self.diversity_times)
        res = {
            "t2m_mm_dist": mm_dist,
            "t2m_diversity": diversity,
            "fid": fid,
            "num_t2m_samples": valid_num_samples,
            "not_matched_t2m_sample": all_not_matched_sample
        }
        for k in range(self.top_k):
            res[f't2m_r_precision_top_{k + 1}'] = top_k_mat[k]

        return res

    def process(self, data_batch, data_samples: SampleList):
        """ Get text and motion embeddings for metrics computation
        :param data_batch:
        :param data_samples: output of model.forward_predict
        :return:
        """
        tasks = [data_sample.get('task') for data_sample in data_samples]
        assert all([task == tasks[0] for task in tasks])
        if tasks[0] != Caption2Motion:
            return

        caption = [data_sample.get(self.text_key) for data_sample in data_samples]
        motion = []
        pred_motion = []
        not_matched_sample = 0
        for data_sample in data_samples:
            gt = data_sample.get(self.motion_key)
            pred = data_sample.get(self.pred_motion_key)
            if pred is None:
                pred = torch.zeros_like(gt, device=gt.device, dtype=self.dtype)
                not_matched_sample += 1
            motion.append(gt.to(self.dtype))
            pred_motion.append(pred.to(self.dtype))

        motion, _ = self.data_preprocessor.do_norm(motion)
        pred_motion, _ = self.data_preprocessor.do_norm(pred_motion)
        result = {
            'text_embedding': self.tmr_model.encode_text(caption)[1],
            'motion_embedding': self.tmr_model.encode_motion(motion)[1],
            'pred_motion_embedding': self.tmr_model.encode_motion(pred_motion)[1],
            'not_matched_sample': not_matched_sample
        }
        self.results.append({**result})
