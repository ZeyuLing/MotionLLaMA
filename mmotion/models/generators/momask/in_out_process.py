from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch


class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)  # Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        return output


class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)  # Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output
