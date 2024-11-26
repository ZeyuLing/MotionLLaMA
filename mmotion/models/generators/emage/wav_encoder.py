from mmotion.models.generators.emage.basic_block import BasicBlock


class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1):
        super().__init__()
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential(
            BasicBlock(audio_in, out_dim // 4, 15, 5, first_dilation=1600, downsample=True),
            BasicBlock(out_dim // 4, out_dim // 4, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(out_dim // 4, out_dim // 4, 15, 1, first_dilation=7, ),
            BasicBlock(out_dim // 4, out_dim // 2, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(out_dim // 2, out_dim // 2, 15, 1, first_dilation=7),
            BasicBlock(out_dim // 2, out_dim, 15, 3, first_dilation=0, downsample=True),
        )

    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1)
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)
