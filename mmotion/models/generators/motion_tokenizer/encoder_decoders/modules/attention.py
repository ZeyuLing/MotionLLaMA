from einops import rearrange
from torch import nn
import torch
from torch.nn.functional import softmax

from mmotion.models.generators.motion_tokenizer.encoder_decoders.modules.activation import get_activation

from mmotion.models.generators.motion_tokenizer.encoder_decoders.modules.norm import get_norm
from mmotion.motion_representation.param_utils import t2m_body_hand_adjacent_matrix, t2m_adjacent_matrix


def get_attn(attn_type: str, in_channels: int, hidden_channels=None, zq_channels=None, norm_type=None, num_heads=1,
             num_joints=None, use_pe=False):
    if attn_type == 'time':
        return TimeAttnBlock(in_channels, hidden_channels, norm_type=norm_type, zq_channels=zq_channels)
    if attn_type == 'joint':
        return JointAttnBlock(in_channels, hidden_channels, num_joints, norm_type=norm_type, zq_channels=zq_channels,
                              use_pe=use_pe)
    if attn_type == 'channel':
        return ChannelAttnBlock(in_channels, hidden_channels, num_heads, norm_type, zq_channels)

    raise NotImplementedError(f'{attn_type} is not implemented')


class TimeAttnBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, norm_type: str = None, zq_channels=None, num_heads=None):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.use_spatial_norm = norm_type == 'spatial'
        hidden_channels = hidden_channels or in_channels
        self.norm = get_norm(norm_type, in_channels, zq_channels)
        self.q = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.k = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.v = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.proj_out = nn.Conv1d(hidden_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x, zq=None):
        h_ = x
        h_ = self.norm(h_, zq) if self.use_spatial_norm else self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t = q.shape
        q = q.permute(0, 2, 1)  # b,t,c
        k = k.reshape(b, c, t)  # b,c,t
        w_ = torch.bmm(q, k)  # b,t,t    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, t)
        w_ = w_.permute(0, 2, 1)  # b,t,t (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c, hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, t)

        h_ = self.proj_out(h_)

        return x + h_


class JointAttnBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_joints: int = 52, norm_type: str = None,
                 zq_channels=None, use_pe=False):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints
        assert hidden_channels % num_joints == 0, f"channels {hidden_channels} must be divisible by num_joints {num_joints}"
        self.use_pe = use_pe
        if use_pe:
            A = t2m_body_hand_adjacent_matrix if num_joints == 52 else t2m_adjacent_matrix
            self.pe = LaplacianPositionalEncoding(hidden_channels // num_joints, A)
        self.use_spatial_norm = norm_type == 'spatial'
        self.norm = get_norm(norm_type, in_channels, zq_channels)
        self.q = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.k = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.v = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.proj_out = nn.Conv1d(hidden_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x, zq=None):
        b = x.shape[0]
        h_ = x
        h_ = self.norm(h_, zq) if self.use_spatial_norm else self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        if self.use_pe:
            q = self.pe(rearrange(q, 'b (j c) t -> b t j c', j=self.num_joints))
            k = self.pe(rearrange(k, 'b (j c) t -> b t j c', j=self.num_joints))
            q = rearrange(q, 'b t j c -> (b t) j c')
            k = rearrange(k, 'b t j c -> (b t) c j')
        else:
            q = rearrange(q, 'b (j c) t -> (b t) j c', j=self.num_joints)
            k = rearrange(k, 'b (j c) t -> (b t) c j', j=self.num_joints)

        c = q.shape[-1]
        w_ = torch.bmm(q, k)  # (b,t),j,j    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=-1)

        # attend to values
        v = rearrange(v, 'b (j c) t -> (b t) c j', j=self.num_joints)
        h_ = torch.bmm(v, w_)  # b, t c j(hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = rearrange(h_, '(b t) c j -> b (j c) t', b=b)

        h_ = self.proj_out(h_)

        return x + h_


class TransformerBlock(nn.Module):
    def __init__(self, in_channels: int, mult: float = 4, attn_type: str = 'joint', activation_type='silu',
                 hidden_channels=None, dropout: float = 0, num_joints: int = 52, norm_type: str = None,
                 zq_channels=None, use_pe=False):
        super().__init__()
        self.attn = get_attn(attn_type, in_channels, hidden_channels, zq_channels, norm_type, num_joints, use_pe)
        self.use_spatial_norm = norm_type == 'spatial'
        inner_dim = int(in_channels * mult)
        self.norm = get_norm(norm_type, in_channels, zq_channels)
        self.ff = nn.Sequential(
            get_activation(activation_type),
            nn.Conv1d(in_channels, inner_dim, 1),
            nn.Dropout(dropout),
            nn.Conv1d(inner_dim, in_channels, 1)
        )

    def forward(self, hidden_states, zq=None):
        """
        :param hidden_states: b t (j c)
        :return: b t (j c)
        """

        hidden_states = self.attn(hidden_states, zq)
        norm_hidden_states = self.norm(hidden_states, zq) if self.use_spatial_norm else self.norm(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states

        return hidden_states


class ChannelAttnBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=None, norm_type: str = None,
                 zq_channels=None):
        super().__init__()
        self.num_heads = num_heads or hidden_channels // 32
        self.use_spatial_norm = norm_type == 'spatial'
        self.norm = get_norm(norm_type, in_channels, zq_channels)
        self.q = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.k = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.v = nn.Conv1d(in_channels,
                           hidden_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.proj_out = nn.Conv1d(hidden_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x, zq=None):
        """
        x: b c t
        """

        h_ = x
        h_ = self.norm(h_, zq) if self.use_spatial_norm else self.norm(h_)
        q = softmax(rearrange(self.q(h_), 'b (n c) t -> (b n) c t', n=self.num_heads), dim=1)
        k = softmax(rearrange(self.k(h_), 'b (n c) t -> (b n) t c', n=self.num_heads), dim=-1)
        v = rearrange(self.v(h_), 'b (n c) t -> (b n) c t', n=self.num_heads)
        # (b n) c c
        w_ = torch.bmm(q, k)
        h_ = torch.bmm(w_, v)
        h_ = rearrange(h_, '(b n) c t -> b (n c) t', n=self.num_heads)
        return x + self.proj_out(h_)
