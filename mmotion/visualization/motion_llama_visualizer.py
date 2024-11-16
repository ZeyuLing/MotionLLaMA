import re
from typing import Dict, Union, List

from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from mmotion.registry import VISUALIZERS
from mmotion.structures import DataSample
from mmotion.utils.task import Task


@VISUALIZERS.register_module(force=True)
class MotionLLaMAVisualizer(Visualizer):
    """
        Visualizer for Motion VQ-VAE evaluation
    """

    def __init__(self,
                 audio_key='audio',
                 motion_key='joints',
                 text_key='text',
                 fn_key: Union[str, List[str]] = ['caption_path', 'music_path',
                                                  'hm3d_path', 'audio_path', 'motion_path'],
                 name: str = 'visualizer',
                 *args,
                 **kwargs):
        super().__init__(name, *args, **kwargs)
        self.audio_key = audio_key
        self.motion_key = motion_key
        self.text_key = text_key
        self.fn_key = fn_key
        if not isinstance(self.fn_key, list):
            self.fn_key = [self.fn_key]

    @master_only
    def add_datasample(self, data_sample: DataSample, step=0) -> None:
        merged_dict = {
            **data_sample.to_dict()
        }
        if 'output' in merged_dict.keys() and isinstance(merged_dict['output'], dict):
            merged_dict.update(**merged_dict['output'])

        task: Task = merged_dict.get('task')

        fn = None
        for key in self.fn_key:
            if key in merged_dict.keys():
                fn = merged_dict[key]
                break
        if fn is None:
            raise AttributeError(f'{self.fn_key} does not exist in data sample keys: {merged_dict.keys()}')
        if isinstance(fn, list):
            fn = fn[0]
        fn = re.split(r' |/|\\', fn)[-1]
        fn = fn.split('.')[0]
        fn = f'{task.abbr}_{fn}'
        self.visualize(fn, merged_dict, step)

    def visualize(self, fn: str, merged_dict: Dict, step: int = 0):
        """ For most visualization situations, just:
        1. Save the text(predicted and ground truth).
        2. Save the audio(predicted and ground truth).
        3. Render all people motions(predicted and ground truth)

        For motion prediction, motion in-between and interaction motion A to B,
        we implement the visualization individually.
        :param fn: The output file name
        :param merged_dict: including the outputs and data_samples.
        :param step: current training step
        :return:
        """

        self.output_text(fn, merged_dict, step)
        pred_audio_path, gt_audio_path = self.output_audio(fn, merged_dict, step)
        pred_motion_path, gt_motion_path = self.output_motion(fn, merged_dict, step)
        self.output_audio_motion(fn, step, pred_audio_path, gt_audio_path,
                                 pred_motion_path, gt_motion_path)

    def output_audio_motion(self, fn, step, pred_audio_path=None, gt_audio_path=None,
                            pred_motion_path=None, gt_motion_path=None, ):
        backend = self._vis_backends.get('merge_audio_video')
        if backend is None:
            return None, None
        if pred_motion_path is None and gt_motion_path is None:
            return
        if pred_audio_path is None and gt_audio_path is None:
            return
        if pred_motion_path is not None:
            audio_path = pred_audio_path or gt_audio_path
            self._vis_backends.get('merge_audio_video').add_image(
                name=fn, audio_path=audio_path, video_path=pred_motion_path, key='pred_audio_motion', step=step)
        if gt_motion_path is not None:
            if gt_audio_path is not None:
                self._vis_backends.get('merge_audio_video').add_image(
                    name=fn, audio_path=gt_audio_path, video_path=gt_motion_path, key='gt_audio_motion', step=step)
            if pred_audio_path is not None:
                self._vis_backends.get('merge_audio_video').add_image(
                    name=fn, audio_path=pred_audio_path, video_path=gt_motion_path, key='pred_audio_motion', step=step)

    def output_motion(self, fn, merged_dict, step: int = 0):
        """ Confirm that u have pred_joints and gt_joints(optional) in merged dict.
        If the task involves interaction motion and completion motion,
        refer to MotionDatapreprocessor.merge_completion_interaction() to merge persons and clips.
        """

        pred_motion_path = None
        gt_motion_path = None
        gt_motion = merged_dict.get(self.motion_key, None)
        if gt_motion is not None:
            gt_motion_path = self._vis_backends.get('motion').add_image(name=fn, step=step, motion=gt_motion, key='gt')

        pred_motion = merged_dict.get(f'pred_{self.motion_key}', None)
        if pred_motion is not None:
            pred_motion_path = self._vis_backends.get('motion').add_image(name=fn, step=step, motion=pred_motion,
                                                                          key='pred')

        recons_motion = merged_dict.get(f'recons_{self.motion_key}', None)
        if recons_motion is not None:
            self._vis_backends.get('motion').add_image(name=fn, step=step, motion=recons_motion,
                                                       key='recons')
        return pred_motion_path, gt_motion_path

    def output_audio(self, fn, merged_dict, step: int = 0):
        backend = self._vis_backends.get('audio')
        if backend is None:
            return None, None
        pred_audio_path = None
        gt_audio_path = None
        pred_audio = merged_dict.get(f'pred_{self.audio_key}', None)
        gt_audio = merged_dict.get(self.audio_key, None)
        if pred_audio is not None:
            pred_audio_path = self._vis_backends.get('audio').add_image(name=fn, step=step, audio=pred_audio,
                                                                        key='pred')
        if gt_audio is not None:
            gt_audio_path = self._vis_backends.get('audio').add_image(name=fn, step=step, audio=gt_audio, key='gt')
        return pred_audio_path, gt_audio_path

    def output_text(self, fn, merged_dict, step: int = 0):

        pred_text = merged_dict.get(f'pred_{self.text_key}')
        if pred_text:
            self._vis_backends.get('text').add_image(name=fn, step=step, text=pred_text, key='pred')

        gt_text = merged_dict.get(self.text_key)
        # save_text
        if gt_text:
            self._vis_backends.get('text').add_image(name=fn, step=step, text=gt_text, key='gt')
