from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat
from mmengine.model import BaseModel
import torch.nn.functional as F

from mmotion.registry import MODELS


def create_src_key_padding_mask(lengths):
    batch_size = len(lengths)
    max_len = max(lengths)
    seq_range = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len)
    lengths_tensor = torch.tensor(lengths).unsqueeze(1)
    mask = seq_range >= lengths_tensor
    return mask.to(torch.bool)  # dtype: torch.bool


class BasicBlock(nn.Module):
    """ based on timm: https://github.com/rwightman/pytorch-image-models """

    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, dilation=1,
                 first_dilation=None, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm1d, drop_block=None,
                 drop_path=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size // 2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, stride=stride, kernel_size=ker_size, padding=first_dilation,
                          dilation=dilation, bias=True),
                norm_layer(planes),
            )
        else:
            self.downsample = None
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x


class WavEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential(
            BasicBlock(1, 32, 15, 5, first_dilation=1600, downsample=True),
            BasicBlock(32, 32, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(32, 32, 15, 1, first_dilation=7, ),
            BasicBlock(32, 64, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(64, 64, 15, 1, first_dilation=7),
            BasicBlock(64, 128, 15, 6, first_dilation=0, downsample=True),
        )

    def forward(self, wav_data, motion_data):
        target_T = motion_data.shape[1]
        wav_data = wav_data.unsqueeze(1)
        wav_data = self.feat_extractor(wav_data)
        if wav_data.shape[-1] != target_T:
            wav_data = F.interpolate(wav_data, size=target_T, mode='linear', align_corners=True)
        wav_data = rearrange(wav_data, 'b c t -> b t c')
        return wav_data


@MODELS.register_module()
class CaMN(BaseModel):
    def __init__(self, pose_dims=156, num_speakers=30, body_dims=66, hand_dims=90,
                 speaker_f=8, audio_f=128, hidden_size=256, n_layer=4, dropout=0.3,
                 data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)

        in_channels = pose_dims + speaker_f + audio_f
        self.hidden_size = hidden_size
        self.audio_encoder = WavEncoder(audio_f)

        self.speaker_embedding = nn.Sequential(
            nn.Embedding(num_speakers, speaker_f),
            nn.Linear(speaker_f, speaker_f),
            nn.LeakyReLU(0.1, True)
        )

        self.LSTM = nn.LSTM(in_channels, hidden_size=hidden_size, num_layers=n_layer,
                            batch_first=True, bidirectional=True, dropout=dropout)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_size // 2, body_dims)
        )

        self.LSTM_hands = nn.LSTM(in_channels + body_dims, hidden_size=hidden_size, num_layers=n_layer,
                                  batch_first=True, bidirectional=True, dropout=dropout)
        self.out_hands = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_size // 2, hand_dims)
        )
        self.audio_fusion_dim = audio_f + speaker_f
        self.audio_fusion = nn.Sequential(
            nn.Linear(self.audio_fusion_dim, hidden_size // 2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_size // 2, audio_f),
            nn.LeakyReLU(0.1, True),
        )

    def forward_tensor(self, inputs, data_samples=None):
        num_frames = data_samples.get('num_frames')
        motion = inputs.get('motion')
        audio = inputs.get('audio')
        speaker = torch.tensor(data_samples.get('speaker_id')).to(motion.device)
        motion_seed = torch.zeros_like(motion)
        B, T = motion.shape[:2]
        speaker_feat_seq = self.speaker_embedding(speaker)
        speaker_feat_seq = repeat(speaker_feat_seq, 'b d -> b t d', t=T)

        audio_feat_seq = self.audio_encoder(audio, motion)

        audio_fusion_seq = self.audio_fusion(
            torch.cat((audio_feat_seq, speaker_feat_seq), dim=2)
            .reshape(-1, self.audio_fusion_dim))
        audio_feat_seq = audio_fusion_seq.reshape(*audio_feat_seq.shape)

        in_data = torch.cat((motion_seed, audio_feat_seq, speaker_feat_seq), dim=2)
        output, _ = self.LSTM(in_data)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        # b t body_dim
        decoder_outputs = self.out(output)
        # b t c + body_dim
        in_data = torch.cat((in_data, decoder_outputs), dim=2)
        output_hands, _ = self.LSTM_hands(in_data)
        output_hands = output_hands[:, :, :self.hidden_size] + output_hands[:, :, self.hidden_size:]
        output_hands = self.out_hands(output_hands.reshape(-1, output_hands.shape[2]))
        decoder_outputs_hands = output_hands.reshape(in_data.shape[0], in_data.shape[1], -1)
        holistic_output = torch.cat((decoder_outputs, decoder_outputs_hands), dim=-1)
        return holistic_output

    def forward_loss(self, inputs, data_samples):
        motion = inputs.get('motion')
        num_frames = data_samples.get('num_frames')
        pred_motion = self.forward_tensor(inputs, data_samples)
        mask = create_src_key_padding_mask(num_frames).unsqueeze(-1).to(motion.device)
        pred_motion = pred_motion * ~mask
        motion = motion * ~mask

        return {
            'loss': F.smooth_l1_loss(pred_motion, motion, reduction='mean', beta=0.1)
        }

    def forward_predict(self, inputs, data_samples):
        pred_motion = self.forward_tensor(inputs, data_samples)

        data_samples.set_field(pred_motion, 'pred_motion')
        data_samples.set_data(inputs)
        # calculate joints positions from motion vectors.

        for key, value in data_samples.to_dict().items():
            if key.endswith('motion') and value is not None:
                value = self.data_preprocessor.destruct(value, data_samples)
                data_samples.set_field(value, key)
                joints_key = key.replace('motion', 'joints')
                joints = self.data_preprocessor.vec2joints(value, data_samples)
                data_samples.set_field(joints, joints_key)
            elif key.endswith('music') or key.endswith('audio'):
                value = self.data_preprocessor._undo_pad_audio(value, data_samples)
                data_samples.set_field(value, 'key')
        data_samples = data_samples.split(allow_nonseq_value=True)
        return data_samples

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        if mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'loss':
            return self.forward_loss(inputs, data_samples)
        else:
            raise NotImplementedError(
                'Forward is not implemented now, please use infer.')
