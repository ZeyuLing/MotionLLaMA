from typing import Union, Dict, List, Sequence

import torch
from mmengine import LOOPS
from mmengine.runner import ValLoop, autocast, TestLoop
from torch.utils.data import DataLoader

from mmotion.evaluation import Evaluator


@LOOPS.register_module(force=True)
class ValLoop(ValLoop):
    """
    The original ValLoop implementation doesn't support dtype setting
    """
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 dtype='fp16') -> None:
        if isinstance(dtype, str):
            if dtype in ['half', 'fp16', 'float16']:
                dtype=torch.float16
            elif dtype in ['float32', 'fp32', 'float']:
                dtype = torch.float32
            elif dtype in ['bf16', 'bfloat16']:
                dtype = torch.bfloat16
            else:
                raise NotImplementedError(f'Unsupported dtype {dtype}')
        self.dtype = dtype
        super().__init__(runner, dataloader, evaluator, fp16)
    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16, dtype=self.dtype):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module(force=True)
class TestLoop(TestLoop):
    """
    The original ValLoop implementation doesn't support dtype setting
    """
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 dtype='fp16') -> None:
        if isinstance(dtype, str):
            if dtype in ['half', 'fp16', 'float16']:
                dtype=torch.float16
            elif dtype in ['float32', 'fp32', 'float']:
                dtype = torch.float32
            elif dtype in ['bf16', 'bfloat16']:
                dtype = torch.bfloat16
            else:
                raise NotImplementedError(f'Unsupported dtype {dtype}')
        self.dtype = dtype
        super().__init__(runner, dataloader, evaluator, fp16)
    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16, dtype=self.dtype):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)