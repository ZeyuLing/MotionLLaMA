import torch.nn as nn
import torch
from diffusers.models.resnet import Downsample1D, Upsample1D

from mmotion.models.generators.motion_tokenizer.encoder_decoders.homi.modules.activation import get_activation
from mmotion.models.generators.motion_tokenizer.encoder_decoders.homi.modules.freq_gate import Gate
from mmotion.models.generators.motion_tokenizer.encoder_decoders.homi.modules.norm import get_norm


class HoMiResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, norm_type: str = 'group',
                 activation_type: str = 'relu', dropout=0., zq_channels=512, dilation: int = 1, use_gate: bool = False):
        super().__init__()
        self.use_spatial_norm = norm_type == 'spatial'
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = get_norm(norm_type, in_channels, zq_channels)
        self.act1 = get_activation(activation_type, in_channels)
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=dilation,
                               dilation=dilation)
        self.use_gate = use_gate
        if use_gate:
            self.gate = Gate(out_channels)
        self.norm2 = get_norm(norm_type, out_channels, zq_channels)
        self.act2 = get_activation(activation_type, in_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if self.in_channels == self.out_channels:
            self.conv_skip = nn.Identity()
        elif self.use_conv_shortcut:
            self.conv_skip = nn.Conv1d(in_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        else:
            self.conv_skip = nn.Conv1d(in_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x, zq=None):
        h = x
        h = self.norm1(h, zq) if self.use_spatial_norm else self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h, zq) if self.use_spatial_norm else self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.use_gate:
            h = h.mul_(self.gate(h))
        x = self.conv_skip(x)

        return x + h


class HoMiBlock1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            zq_channels: int = None,
            dropout=0.,
            num_layers: int = 1,
            add_downsample: bool = False,
            add_upsample: bool = False,
            norm_type: str = None,
            dilation=1,
            activation_type: str = 'relu',
            use_gate: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_downsample = add_downsample

        # there will always be at least one resnet
        resnets = [HoMiResConvBlock(in_channels, out_channels, conv_shortcut=False, activation_type=activation_type,
                                    norm_type=norm_type, zq_channels=zq_channels, dropout=dropout, dilation=dilation,
                                    use_gate=use_gate)
                   for _ in range(num_layers)]

        self.resnets = nn.ModuleList(resnets)

        self.upsample = Upsample1D(out_channels, use_conv_transpose=True) if add_upsample else nn.Identity()

        self.downsample = Downsample1D(out_channels, use_conv=True) if add_downsample else nn.Identity()

    def forward(self, hidden_states: torch.FloatTensor, zq=None) -> torch.FloatTensor:
        """
        :param hidden_states:
        :param zq: for decoder, input the quantized output zq to normalize the hidden states
        :return:
        """
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, zq)

        hidden_states = self.upsample(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states
