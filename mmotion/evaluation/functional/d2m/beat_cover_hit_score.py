from typing import Union, Sequence

import librosa
import numpy as np
import torch


def beat_detect(x: Union[torch.Tensor, np.ndarray], sr=24000) -> np.ndarray:
    """
    :param x: music, shape in T
    :param sr: 24000 as default
    :return:
    """
    if isinstance(x, torch.Tensor):
        x = x.float().detach().cpu().numpy()
    onsets = librosa.onset.onset_detect(y=x, sr=sr, wait=1, delta=0.2, pre_avg=3, post_avg=3, pre_max=3, post_max=3,
                                        units='time')
    num_seconds = int(np.ceil(len(x) / sr))
    beats = np.zeros([num_seconds])
    for time in onsets:
        beats[int(np.trunc(time))] = 1
    return beats


def beat_scores(gt: Union[torch.Tensor, np.ndarray], syn: Union[torch.Tensor, np.ndarray], sr: int = 24000):
    """
    :param gt: gt music
    :param syn: synthesised music
    :return:
    """
    gt = gt[:len(syn)]
    assert len(gt) == len(syn)
    gt_beats = beat_detect(gt, sr)
    sync_beats = beat_detect(syn, sr)
    num_gt_beats = sum(gt_beats)
    num_sync_beats = sum(sync_beats)
    hit_beats = np.sum(gt_beats * sync_beats)
    hit_rate = hit_beats / num_gt_beats if num_sync_beats else 0
    cover_rate = hit_beats / num_sync_beats if num_sync_beats else 0
    return cover_rate, hit_rate


def batch_beat_cover_hit_f1_score(batch_gt: Sequence[Union[torch.Tensor, np.ndarray]],
                                  batch_sync: Sequence[Union[torch.Tensor, np.ndarray]], sr: int = 24000):
    batch_size = len(batch_gt)
    batch_cover_score = []
    batch_hit_score = []
    for gt, sync in zip(batch_gt, batch_sync):
        cover_rate, hit_rate = beat_scores(gt, sync, sr)
        batch_cover_score.append(cover_rate)
        batch_hit_score.append(hit_rate)
    bcs, bhs = sum(batch_cover_score) / batch_size, sum(batch_hit_score) / batch_size
    f1 = (2 * bcs * bhs) / (bcs + bhs)
    return bcs, bhs, f1
