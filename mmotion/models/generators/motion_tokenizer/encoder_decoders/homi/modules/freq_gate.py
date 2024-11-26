import torch
from einops import rearrange
from torch import nn
from torch.fft import fft


class Gate(nn.Module):
    def __init__(self, in_channels, squeeze_rate: int = 2):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels // squeeze_rate, kernel_size=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels // squeeze_rate, out_channels=in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # fft doesnt support bfloat16
        ori_dtype = x.dtype
        x = x.float()

        x = torch.real(fft(rearrange(fft(x), 'b c t -> b t c')))
        x = rearrange(x, 'b t c -> b c t').to(ori_dtype)
        x = self.se(x)

        return x


if __name__ == "__main__":
    model = Gate(512)
    x = torch.rand([2, 512, 64])
    print(model(x).shape)
