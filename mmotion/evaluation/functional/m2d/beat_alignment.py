from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from typing import Union

import librosa
import torch
import numpy as np

from scipy.optimize import linear_sum_assignment


def extract_music_beats(music: Union[torch.Tensor, np.ndarray], sr: int, motion_fps=30):
    """
    Extract beats from a music file.

    :param music: [T]
    :param sr: sampling rate
    :return: Beats (indices) in the audio signal.
    """
    if isinstance(music, torch.Tensor):
        music = music.float().detach().cpu().numpy()
    # Compute the onset envelope (energy of the audio signal)

    onset_env = librosa.onset.onset_strength(y=music, sr=sr)
    # Detect beats based on the onset envelope
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beats = librosa.frames_to_time(beats)
    beats = beats * motion_fps

    return beats


def cal_beat_align(dance_beat: np.ndarray,
                   music_beat: np.ndarray,
                   sigma=3):
    """
    :param dance_beat: N1, each item means a beat in the dance motion
    :param music_beat: N2, each item means a beat in the music
    :return:
    """

    ba = 0.
    for bb in music_beat:
        ba += np.exp(-np.min((dance_beat - bb) ** 2) / 2 / sigma ** 2)
    return ba / len(music_beat)


def cal_hungarian_beat_align(dance_beat: np.ndarray, music_beat: np.ndarray, fps=30.):
    dance_beat = dance_beat * 1. / fps
    music_beat = music_beat * 1. / fps

    n_dance = len(dance_beat)
    n_music = len(music_beat)

    # 如果dance_beat的长度小于music_beat，扩展dance_beat
    if n_dance < n_music:
        repeats = (n_music + n_dance - 1) // n_dance  # 向上取整
        dance_extended = np.tile(dance_beat, repeats)[:n_music]
    else:
        dance_extended = dance_beat[:n_music]

    # 构建成本矩阵（使用绝对差作为成本）
    cost_matrix = np.abs(dance_extended[:, np.newaxis] - music_beat[np.newaxis, :])

    # 执行匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 计算总匹配距离
    total_distance = cost_matrix[row_ind, col_ind].sum()

    matching_pairs = []
    for i, j in zip(row_ind, col_ind):
        if i < n_dance and j < n_music:
            distance = cost_matrix[i, j]
            matching_pairs.append((i % n_dance, j, dance_extended[i], music_beat[j], distance))
        # 忽略超出原始dance_beat范围的匹配（由于扩展可能导致索引超出）

    # 计算匹配分数，分数越高表示匹配越好
    score = 1 / (1 + total_distance)  # 可根据需求调整评分方式
    return score


def extract_dance_beats(pred_motion: Union[torch.Tensor, np.ndarray]):
    """
    :param pred_motion: t j c
    :return:
    """
    # upsample to 60fps to make a fair comparison
    if isinstance(pred_motion, torch.Tensor):
        pred_motion = pred_motion.float().detach().cpu().numpy()
    kinetic_vel = np.mean(np.sqrt(np.sum((pred_motion[1:] - pred_motion[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = gaussian_filter(kinetic_vel, 5)
    # find the minimum point of velocity return (array,)
    motion_beats = argrelextrema(kinetic_vel, np.less)[0]
    return motion_beats, len(kinetic_vel)
