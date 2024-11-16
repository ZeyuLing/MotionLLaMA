import os.path
from typing import Optional, Sequence, Dict

from mmengine import HOOKS
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from tqdm import tqdm


@HOOKS.register_module(force=True)
class MotionVisualizationHook(Hook):
    """
        Write the predicted results during the process of testing.
    """

    def __init__(self,
                 interval: int = 1,
                 sample_interval: int = 1,
                 draw_gt: bool = True,
                 draw_pred: bool = True,
                 show_bar: bool = False):
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.interval = interval
        self.sample_interval = sample_interval
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.show_bar = show_bar

    def after_test_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: Dict = None,
                        outputs: Optional[Sequence] = None) -> None:
        """
        :param runner: The runner of the training process.
        :param batch_idx: The index of the current batch in the test loop.
        :param data_batch: Data from dataloader of MotionVQVAEDataset
        :param outputs: (Sequence, optional): Outputs from model.
        :return: None
        """
        self.save_dir = os.path.join(runner.work_dir, runner.timestamp, 'vis_data')
        if self.every_n_inner_iters(batch_idx, self.interval):
            progress = tqdm(enumerate(outputs)) if self.show_bar else enumerate(outputs)
            for sample_idx, output in progress:  # type: ignore
                data_sample = output
                # visualizer should be MotionVisualizer
                if sample_idx % self.sample_interval == 0:
                    self._visualizer.add_datasample(data_sample,
                                                     save_dir=self.save_dir,
                                                     draw_gt=self.draw_gt,
                                                     draw_pred=self.draw_pred)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: Dict = None,
                       outputs: Optional[Sequence] = None) -> None:
        return self.after_test_iter(runner, batch_idx, data_batch, outputs)
