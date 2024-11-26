import os
import sys
from numbers import Number
from typing import Optional, List, Union

import numpy as np
import torch
from torch.nn.functional import pad

sys.path.append(os.curdir)
from mmotion.registry import MODELS


@MODELS.register_module()
class Pad1D:
    def __init__(self,
                 size: Optional[int] = None,
                 size_divisor: Optional[int] = None,
                 pad_to_max: bool = False
                 ):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_to_max = pad_to_max
        if not self.pad_to_max:
            assert size is not None or size_divisor is not None, \
                'size or size_divisor should be valid'
        assert size is None or size_divisor is None


    def __call__(self, signal: Union[List[torch.Tensor], torch.Tensor], is_batch: bool = True):
        """
        :param signal: signal need to be padded, motion or audio
        :param is_batch: signal is a batch
        :return:
        """
        size = None
        if not is_batch:
            signal = [signal]

        is_audio = len(signal[0].shape) == 1
        if self.pad_to_max:
            assert is_batch, ('if you want to pad the samples to the max size in the batch,'
                              'you should input a batch of tensors')
            max_size = max([s.shape[0] for s in signal])
            size = max_size
        if self.size_divisor is not None:
            if size is None:
                size = signal[0].shape[0]
            pad_length = int(np.ceil(size / self.size_divisor)) * self.size_divisor
            size = pad_length
        elif self.size is not None:
            size = self.size
        pad_val = 0

        pad_infos = []
        padded_signals = []
        for s in signal:
            if is_audio:
                s = s.unsqueeze(-1)
            ps = pad(
                s.transpose(0, 1),
                (0, max(0, size - s.shape[0])),
                mode='constant',
                value=pad_val
            ).transpose(0, 1)
            pad_info = {
                'padding_size': ps.shape[0] - s.shape[0],
                'pad_size_divisor': self.size_divisor,
                'num_frames': s.shape[0]
            }
            pad_infos.append(pad_info)
            padded_signals.append(ps)
        padded_signals = torch.stack(padded_signals, dim=0)
        if is_audio:
            padded_signals = padded_signals.squeeze(-1)
        if not is_batch:
            padded_signals = padded_signals[0]
            pad_infos = pad_infos[0]

        return padded_signals, pad_infos


if __name__ == '__main__':
    pad_1d = Pad1D(pad_to_max=True)

    batch_motion = [
        torch.rand([1956012]),
        torch.rand([3111321]),
        torch.rand([2566666])
    ]

    padded, infos = pad_1d(batch_motion)
    print(padded.shape)
    print(infos)
