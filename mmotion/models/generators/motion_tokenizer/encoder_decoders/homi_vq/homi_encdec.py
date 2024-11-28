import os
import sys
from typing import Tuple
import torch
from torch import nn
import torch.autograd.graph
from mmotion.models.generators.motion_tokenizer.encoder_decoders.homi_vq.homi_blocks import HoMiBlock1D
from mmotion.models.generators.motion_tokenizer.encoder_decoders.homi_vq.modules import get_activation
from mmotion.models.generators.motion_tokenizer.encoder_decoders.homi_vq.modules import get_norm

sys.path.append(os.curdir)

from mmotion.registry import MODELS


@MODELS.register_module(force=True)
class HoMiEncoder(nn.Module):
    def __init__(self, in_channels: int = 156, body_in_channels: int = 66, out_channels: int = 512,
                 block_out_channels: Tuple[int, ...] = (64,),
                 layers_per_block: int = 2, layers_mid_block: int = 0, norm_type: str = None, dilation_growth_rate=1,
                 activation_type: str = 'relu', use_gate: bool = False, out_gate: bool = True):
        super().__init__()
        hand_in_channels = in_channels - body_in_channels
        self.body_dim = body_in_channels
        self.body_encoder = BaseEncoder(
            body_in_channels, out_channels, block_out_channels, layers_per_block, layers_mid_block, norm_type,
            dilation_growth_rate, activation_type, use_gate
        )
        self.hand_encoder = BaseEncoder(
            hand_in_channels, out_channels, block_out_channels, layers_per_block, layers_mid_block, norm_type,
            dilation_growth_rate, activation_type, use_gate
        )

        self.out = HoMiBlock1D(
            in_channels=out_channels * 2,
            out_channels=out_channels * 2,
            num_layers=layers_mid_block,
            norm_type=norm_type,
            activation_type=activation_type,
            dilation=dilation_growth_rate,
            add_upsample=False,
            add_downsample=False,
            use_gate=out_gate
        )

    def forward(self, x):
        body = x[:, :self.body_dim]
        hand = x[:, self.body_dim:]
        body = self.body_encoder(body)
        hand = self.hand_encoder(hand)
        return self.out(torch.cat([body, hand], dim=1))

    @property
    def downsample_rate(self):
        return self.body_encoder.downsample_rate

    @property
    def motion_dim(self):
        return self.body_encoder.motion_dim + self.hand_encoder.motion_dim


@MODELS.register_module(force=True)
class BaseEncoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
    """

    def __init__(
            self,
            in_channels: int = 156,
            out_channels: int = 156,
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: int = 2,
            layers_mid_block: int = 0,
            norm_type: str = None,
            dilation_growth_rate=1,
            activation_type: str = 'relu',
            use_gate: bool = False
    ):
        super().__init__()

        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.conv_in = nn.Conv1d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        assert norm_type != 'spatial', 'SpatialNorm should be used in Decoder'
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = HoMiBlock1D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                add_upsample=False,
                norm_type=norm_type,
                activation_type=activation_type,
                dilation=dilation_growth_rate,
                use_gate=use_gate
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = HoMiBlock1D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            num_layers=layers_mid_block,
            norm_type=norm_type,
            activation_type=activation_type,
            dilation=dilation_growth_rate,
            add_upsample=False,
            add_downsample=False,
            use_gate=use_gate

        ) if layers_mid_block else nn.Identity()

        #
        self.conv_norm_out = get_norm(norm_type, block_out_channels[-1])
        self.conv_act = get_activation(activation_type or 'silu')

        conv_out_channels = out_channels
        self.conv_out = nn.Conv1d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    @property
    def downsample_rate(self):
        return 2 ** (len(self.down_blocks) - 1)

    @property
    def motion_dim(self):
        return self.in_channels


@MODELS.register_module(force=True)
class HoMiDecoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: int = 3,
            layers_mid_block: int = 0,
            dilation_growth_rate=1,
            norm_type: str = None,  # group, spatial
            activation_type: str = 'relu',
            use_gate: bool = False,
    ):
        super().__init__()
        self.use_spatial_norm = norm_type == 'spatial'
        self.conv_in = nn.Conv1d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        zq_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.have_mid = layers_mid_block > 0
        self.mid_block = HoMiBlock1D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            num_layers=layers_mid_block,
            zq_channels=zq_channels,
            norm_type=norm_type,
            activation_type=activation_type,
            dilation=dilation_growth_rate,
            use_gate=use_gate
        ) if layers_mid_block else nn.Identity()

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = HoMiBlock1D(
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                add_downsample=False,
                zq_channels=zq_channels,
                norm_type=norm_type,
                activation_type=activation_type,
                dilation=dilation_growth_rate,
                use_gate=use_gate
            )

            self.up_blocks.append(up_block)

        # out
        self.conv_norm_out = get_norm(norm_type, block_out_channels[0], zq_channels)

        self.conv_act = get_activation(activation_type or 'silu')
        self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(
            self,
            sample: torch.Tensor
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""
        zq = sample if self.use_spatial_norm else None
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, zq) if self.have_mid else self.mid_block(sample)
        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, zq)

        # post-process
        sample = self.conv_norm_out(sample, zq) if self.use_spatial_norm else self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


if __name__ == '__main__':
    encoder = HoMiEncoder(
        block_out_channels=(512, 512, 512),
        num_joints=52,
        joint_channels=32,
        out_channels=512
    )
    decoder = HoMiDecoder(
        block_out_channels=(512, 512, 512),
        num_joints=52,
        joint_channels=32,
        out_channels=156
    )
    motion = torch.rand([2, 156, 64])
    out = encoder(motion)
    print(out.shape)
    out = decoder(out)
    print(out.shape)
