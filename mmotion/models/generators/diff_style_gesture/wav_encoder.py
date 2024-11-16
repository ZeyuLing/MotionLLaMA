from torch import nn


class WavEncoder(nn.Module):
    def __init__(self, dim_in=768, dim_out=64):
        super().__init__()
        self.audio_feature_map = nn.Linear(dim_in, dim_out)

    def forward(self, rep):
        rep = self.audio_feature_map(rep)
        return rep
