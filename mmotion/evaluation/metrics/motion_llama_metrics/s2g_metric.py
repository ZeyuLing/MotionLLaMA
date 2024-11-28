from typing import Dict, Optional, List
import torch
from mmengine import Config
import os
from mmengine.evaluator import BaseMetric
from mmengine.model import is_model_wrapper
from torch.utils.data import DataLoader
from torch import nn

from mmotion.evaluation.functional.m2d.beat_alignment import extract_dance_beats, extract_music_beats, \
    cal_beat_align, cal_hungarian_beat_align
from mmotion.evaluation.functional.s2g.l1div import cal_l1div
from mmotion.evaluation.functional.t2m.fid import cal_fid
from mmotion.models import GestureVAE
from mmotion.registry import MODELS, METRICS
from mmotion.utils.task.task_lib import Audio2Gesture, CaptionAudio2Gesture
from mmotion.utils.typing import SampleList


@METRICS.register_module()
class S2GMetric(BaseMetric):
    def __init__(self,
                 gesture_vae: Dict,
                 motion_key: str = 'motion',
                 pred_motion_key: str = 'pred_motion',
                 joints_key: str = 'joints',
                 pred_joints_key: str = 'pred_joints',
                 audio_key: str = 'audio',
                 dtype=torch.bfloat16,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None
                 ):
        self.motion_key = motion_key
        self.audio_key = audio_key
        self.pred_motion_key = pred_motion_key
        self.joints_key = joints_key
        self.pred_joints_key = pred_joints_key
        self.dtype = dtype
        self.gesture_vae: GestureVAE = self.build_vae(gesture_vae)

        super(S2GMetric, self).__init__(collect_device, prefix)

    def build_vae(self, gesture_vae: Dict):
        """
        :param gesture_vae: vae config
        :return: Vqvae module.
        """
        type = gesture_vae['type']
        if os.path.isfile(type):
            # allow using config file path as type, simplify config writing.
            init_cfg = gesture_vae.pop('init_cfg', None)
            gesture_vae = Config.fromfile(type)['model']
            if init_cfg is not None:
                gesture_vae['init_cfg'] = init_cfg

        vae = MODELS.build(gesture_vae).eval().cuda()
        if gesture_vae.get('init_cfg', None) is not None:
            vae.init_weights()
        self.data_preprocessor = vae.data_preprocessor
        return vae.eval().to(dtype=self.dtype, device='cuda')

    def compute_metrics(self, results: List):
        all_motion = sum([result['motion'] for result in results], [])
        all_pred_motion = sum([result['pred_motion'] for result in results], [])
        all_motion_embeddings = torch.cat([result['motion_embedding'] for result in results], dim=0).to(self.dtype)
        all_pred_motion_embeddings = torch.cat([result['pred_motion_embedding'] for result in results], dim=0).to(
            self.dtype)

        num_test_motion = all_motion_embeddings.shape[0]
        num_pred_samples = all_pred_motion_embeddings.shape[0]

        all_motion_beats = sum([result['motion_beat'] for result in results], [])
        all_pred_motion_beats = sum([result['pred_motion_beat'] for result in results], [])
        all_audio_beats = sum([result['audio_beat'] for result in results], [])

        ba_gt = [cal_beat_align(motion_beat, music_beat) for motion_beat, music_beat in
                 zip(all_motion_beats, all_audio_beats)]
        ba = [cal_beat_align(motion_beat, music_beat) for motion_beat, music_beat in
              zip(all_pred_motion_beats, all_audio_beats)]

        hba_gt = [cal_hungarian_beat_align(motion_beat, music_beat, fps=self.fps) for motion_beat, music_beat in
                  zip(all_motion_beats, all_audio_beats)]
        hba = [cal_hungarian_beat_align(motion_beat, music_beat, fps=self.fps) for motion_beat, music_beat in
               zip(all_pred_motion_beats, all_audio_beats)]

        fid = cal_fid(all_pred_motion_embeddings,
                      all_motion_embeddings)
        l1_div = cal_l1div(all_pred_motion)
        l1_div_gt = cal_l1div(all_motion)
        res = {
            'fid': fid,

            'l1_div': l1_div,
            'l1_div_gt': l1_div_gt,

            'num_test_motion': num_test_motion,
            'num_pred_samples': num_pred_samples,

            'beat_alignment': sum(ba) / num_test_motion,
            'beat_alignment_gt': sum(ba_gt) / num_test_motion,

            'hungarian_ba': sum(hba) / num_test_motion,
            'hungarian_ba_gt': sum(hba_gt) / num_test_motion
        }
        return res

    def process(self, data_batch, data_samples: SampleList):
        self.fps = [data_sample.get('fps') for data_sample in data_samples][0]
        batch_motion = []
        batch_pred_motion = []
        batch_joints = []
        batch_pred_joints = []
        tasks = [data_sample.get('task') for data_sample in data_samples]
        if tasks[0] not in [Audio2Gesture, CaptionAudio2Gesture]:
            return
        for data_sample in data_samples:
            motion = data_sample.get(self.motion_key)
            pred_motion = data_sample.get(self.pred_motion_key)
            joints = data_sample.get(self.joints_key)
            pred_joints = data_sample.get(self.pred_joints_key)
            batch_joints.append(joints)
            batch_pred_joints.append(pred_joints)
            batch_motion.append(motion.to(self.dtype))
            batch_pred_motion.append(pred_motion.to(self.dtype))

        audio_beat = [extract_music_beats(data_sample.get(self.audio_key),
                                          data_sample.get('sr'),
                                          data_sample.get('fps')) for data_sample in data_samples]
        motion_beat = [extract_dance_beats(m)[0] for m in batch_joints]
        pred_motion_beat = [extract_dance_beats(m)[0] for m in batch_pred_joints]

        result = {}
        if len(batch_motion):
            batch_motion, _ = self.data_preprocessor.do_norm(batch_motion)
            result['motion_embedding'] = self.gesture_vae.encode_motion(batch_motion)[1]
            result['motion_beat'] = motion_beat
            result['motion'] = batch_motion

        if len(batch_pred_motion):
            batch_pred_motion, _ = self.data_preprocessor.do_norm(batch_pred_motion)
            result['pred_motion_embedding'] = self.gesture_vae.encode_motion(batch_pred_motion)[1]
            result['pred_motion_beat'] = pred_motion_beat
            result['pred_motion'] = batch_pred_motion

        result['audio_beat'] = audio_beat

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
