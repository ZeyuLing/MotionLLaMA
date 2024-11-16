from typing import Dict, List, Optional

from nlgeval import NLGEval

from mmotion.evaluation.functional.t2m.matching_score_precision import euclidean_distance_matrix, calculate_top_k, \
    cal_mmdist_rprecision
from mmotion.evaluation.metrics.motion_llama_metrics.tmr_based_metric import BaseTMRMetric
from mmotion.registry import METRICS
import torch
from bert_score import score as compute_bert_score
from mmotion.utils.task.task_lib import Motion2Text
from mmotion.utils.typing import SampleList


@METRICS.register_module()
class M2TMetric(BaseTMRMetric):
    def __init__(self,
                 tmr_model: Dict,
                 bert_path='checkpoints/roberta-large',
                 dtype=torch.bfloat16,
                 text_key: str = 'caption',
                 text_list_key: str = 'caption_list',
                 pred_text_key: str = 'pred_caption',
                 motion_key: str = 'motion',
                 top_k=3,
                 r_precision_batch: int = 256,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None, ):
        """ For single-person t2m and m2t evaluation.
        If caption is not provided, fid and diversity of pred motions will be evaluated
        including following metrics:
        FID,
        Matching Score, R-precision,
        Diversity,

        :param tmr_model: TMR
        :param motion_key: gt motion key
        :param text_key: gt caption key
        """
        super(M2TMetric, self).__init__(tmr_model, dtype, collect_device, prefix)
        self.nlg_evaluator = NLGEval(
            metrics_to_omit=['METEOR',
                             'EmbeddingAverageCosineSimilarity',
                             'SkipThoughtCS',
                             'SPICE',
                             'VectorExtremaCosineSimilarity',
                             'GreedyMatchingScore'],
            no_glove=True
        )
        self.motion_key = motion_key
        self.text_key = text_key
        self.text_list_key = text_list_key
        self.pred_text_key = pred_text_key
        self.top_k = top_k
        self.r_precision_batch = r_precision_batch
        self.bert_path = bert_path

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        all_gt_captions = sum([result['text'] for result in results], [])
        all_not_matched_sample = sum(result['not_matched_sample'] for result in results)
        all_caption_candidates = sum([result['text_candidates'] for result in results], [])
        all_pred_captions = sum([result['pred_text'] for result in results], [])
        all_pred_text_embeddings = torch.cat([result['pred_text_embedding']
                                              for result in results], dim=0)
        all_motion_embeddings = torch.cat([result['motion_embedding'] for result in results],
                                          dim=0)

        num_samples = all_motion_embeddings.shape[0]
        valid_num_samples = num_samples // self.r_precision_batch * self.r_precision_batch

        shuffle_idx = torch.randperm(num_samples)
        all_motion_embeddings = all_motion_embeddings[shuffle_idx]
        all_pred_text_embeddings = all_pred_text_embeddings[shuffle_idx]

        mm_dist, top_k_mat = cal_mmdist_rprecision(
            all_pred_text_embeddings, all_motion_embeddings,
            self.r_precision_batch, self.top_k, reduction=True)

        bert_precision, bert_recall, bert_f1 = compute_bert_score(
            all_pred_captions,
            all_caption_candidates,
            lang='en',
            rescale_with_baseline=True,
            idf=True,
            device=self.collect_device,
            verbose=True,
            model_type=self.bert_path,
            num_layers=17
        )
        all_caption_candidates = [list(refs) for refs in zip(*all_caption_candidates)]
        scores = self.nlg_evaluator.compute_metrics(ref_list=all_caption_candidates, hyp_list=all_pred_captions)
        res = {
            'm2t_mm_dist': mm_dist,
            'bert_f1': bert_f1.mean(),
            'bert_precision': bert_precision.mean(),
            'bert_recall': bert_recall.mean(),
            'bleu_1': scores['Bleu_1'],
            'bleu_4': scores['Bleu_4'],
            'rouge_l': scores['ROUGE_L'],
            'CIDEr': scores['CIDEr'],
            'num_m2t_samples': valid_num_samples,
            "not_matched_m2t_sample": all_not_matched_sample
        }
        for k in range(self.top_k):
            res[f'm2t_r_precision_top_{k + 1}'] = top_k_mat[k]

        return res

    def process(self, data_batch, data_samples: SampleList):
        """ Get text and motion embeddings for metrics computation
        :param data_batch:
        :param data_samples: output of model.forward_predict
        :return:
        """
        tasks = [data_sample.get('task') for data_sample in data_samples]
        if tasks[0] != Motion2Text:
            return
        batch_caption = []
        batch_caption_list = []
        batch_motion = []
        batch_pred_caption = []
        not_matched_sample = 0
        for data_sample in data_samples:
            caption = data_sample.get(self.text_key)
            caption_list = data_sample.get(self.text_list_key)
            if len(caption_list) < 3:
                caption_list = caption_list + caption_list[:3 - len(caption_list)]
            pred_caption = data_sample.get(self.pred_text_key)
            if pred_caption is None:
                not_matched_sample += 1
                pred_caption = ''
            motion = data_sample.get(self.motion_key).to(self.dtype)
            batch_pred_caption.append(pred_caption)
            batch_caption.append(caption)
            batch_caption_list.append(caption_list)
            batch_motion.append(motion)
        batch_motion, _ = self.data_preprocessor._do_norm(batch_motion)
        result = {
            'text': batch_caption,
            'pred_text': batch_pred_caption,
            'text_candidates': batch_caption_list,
            'motion_embedding': self.tmr_model.encode_motion(batch_motion)[1],
            'pred_text_embedding': self.tmr_model.encode_text(batch_pred_caption)[1],
            'not_matched_sample': not_matched_sample
        }

        self.results.append({**result})
