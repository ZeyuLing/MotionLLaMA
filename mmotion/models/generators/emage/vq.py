from torch import nn

from mmotion.models.generators.emage.quantizer import Quantizer
from mmotion.models.generators.emage.res_block import ResBlock
from mmotion.registry import MODELS


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class VQEncoder(nn.Module):
    def __init__(self,
                 input_feats: int,
                 num_layers=4,
                 hidden_size: int = 256):
        super(VQEncoder, self).__init__()
        n_down = num_layers
        channels = [hidden_size]
        for i in range(n_down - 1):
            channels.append(hidden_size)

        input_size = input_feats
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i - 1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs


class VQDecoder(nn.Module):
    def __init__(self, input_feats, num_layers=4, hidden_size: int = 256):
        super(VQDecoder, self).__init__()
        n_up = num_layers
        channels = []
        for i in range(n_up - 1):
            channels.append(hidden_size)
        channels.append(hidden_size)
        channels.append(input_feats)
        input_size = hidden_size
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            up_factor = 2 if i < n_up - 1 else 1
            layers += [
                # nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs


@MODELS.register_module()
class VQVAEConvZero(nn.Module):
    def __init__(self, input_feats,
                 num_layers=4,
                 codebook_size=256,
                 code_dim=256, ):
        super(VQVAEConvZero, self).__init__()
        self.encoder = VQEncoder(input_feats, num_layers, code_dim)
        self.quantizer = Quantizer(codebook_size, code_dim, 1.)
        self.decoder = VQDecoder(input_feats, num_layers, code_dim)

    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat": vq_latent,
            "embedding_loss": embedding_loss,
            "perplexity": perplexity,
            "rec_pose": rec_pose
        }

    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index

    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q

    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose


class MotionEncoder(nn.Module):
    def __init__(self, input_feats, num_layers: int = 4, hidden_size=256):
        super(MotionEncoder, self).__init__()
        n_down = num_layers
        channels = [hidden_size]
        for i in range(n_down - 1):
            channels.append(hidden_size)

        input_size = input_feats
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i - 1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs
