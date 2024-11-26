from torch import nn

from mmotion.models.generators.mcm.attention import TimeWiseSelfAttention, CrossAttention, ChannelWiseSelfAttention
from mmotion.models.generators.mcm.ffn import FFN


class MWNetBlock(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1,
                 chan_first=False):
        super().__init__()
        self.sa_block = TimeWiseSelfAttention(latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = CrossAttention(latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn_1 = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.cwa_block = ChannelWiseSelfAttention(latent_dim, num_head, dropout, time_embed_dim)
        self.ffn_2 = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.chan_first = chan_first

    def forward(self, x, xf, emb, src_mask):
        x = self.cwa_block(x, emb, src_mask)
        x = self.ffn_1(x, emb)
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn_2(x, emb)
        return x
