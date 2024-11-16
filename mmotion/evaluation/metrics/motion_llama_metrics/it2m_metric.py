from typing import Dict, List, Optional

from mmotion.evaluation.metrics.motion_llama_metrics.t2m_metric import T2MMetric
from mmotion.evaluation.functional.t2m.diversity import cal_diversity
from mmotion.evaluation.functional.t2m.fid import cal_fid
from mmotion.evaluation.functional.t2m.matching_score_precision import euclidean_distance_matrix, calculate_top_k, \
    cal_mmdist_rprecision
from mmotion.registry import METRICS
import torch
import torch.nn.functional as F
from mmotion.utils.task.task_lib import Caption2Motion, InterUnionCaption2Motion
from mmotion.utils.typing import SampleList


@METRICS.register_module()
class IT2MMetric(T2MMetric):
    def __init__(self,
                 tmr_model: Dict,
                 dtype=torch.bfloat16,
                 text_key: str = 'union_caption',
                 motion_key: str = 'motion',
                 interactor_motion_key: str = 'interactor_motion',
                 pred_motion_key: str = 'pred_motion',
                 pred_interactor_motion_key='pred_interactor_motion',
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
        self.pred_interactor_motion_key = pred_interactor_motion_key
        self.interactor_motion_key = interactor_motion_key
        super(IT2MMetric, self).__init__(tmr_model=tmr_model,
                                         motion_key=motion_key,
                                         pred_motion_key=pred_motion_key,
                                         top_k=top_k,
                                         text_key=text_key,
                                         diversity_times=diversity_times,
                                         r_precision_batch=r_precision_batch,
                                         dtype=dtype,
                                         collect_device=collect_device,
                                         prefix=prefix)

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
            "it2m_mm_dist": mm_dist,
            "it2m_diversity": diversity,
            "it2m_fid": fid,
            "num_it2m_samples": valid_num_samples,
            "not_matched_it2m_sample": all_not_matched_sample
        }
        for k in range(self.top_k):
            res[f'it2m_r_precision_top_{k + 1}'] = top_k_mat[k]

        return res

    def process(self, data_batch, data_samples: SampleList):
        """ Get text and motion embeddings for metrics computation
        :param data_batch:
        :param data_samples: output of model.forward_predict
        :return:
        """
        tasks = [data_sample.get('task') for data_sample in data_samples]
        assert all([task == tasks[0] for task in tasks])
        if tasks[0] != InterUnionCaption2Motion:
            return

        caption = [data_sample.get(self.text_key) for data_sample in data_samples]
        motion = []
        pred_motion = []
        not_matched_sample = 0
        for data_sample in data_samples:
            gt_a = data_sample.get(self.motion_key)
            gt_b = data_sample.get(self.interactor_motion_key)
            gt = torch.cat([self.data_preprocessor._do_norm(gt_a)[0],
                            self.data_preprocessor._do_norm(gt_b)[0]], dim=-1)
            pred_a = data_sample.get(self.pred_motion_key)
            pred_b = data_sample.get(self.pred_interactor_motion_key)

            if pred_a is None or pred_b is None:
                pred = torch.zeros_like(gt, device=gt.device)
                not_matched_sample += 1
            else:
                pred_num_frames = min(len(pred_a), len(pred_b))
                pred = torch.cat([self.data_preprocessor._do_norm(pred_a[:pred_num_frames])[0],
                                  self.data_preprocessor._do_norm(pred_b[:pred_num_frames])[0]], dim=-1)
            motion.append(gt.to(self.dtype))
            pred_motion.append(pred.to(self.dtype))

        result = {
            'text_embedding': self.tmr_model.encode_text(caption)[1],
            'motion_embedding': self.tmr_model.encode_motion(motion)[1],
            'pred_motion_embedding': self.tmr_model.encode_motion(pred_motion)[1],
            'not_matched_sample': not_matched_sample
        }

        self.results.append(result)
