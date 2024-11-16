import torch
from torch import nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach()) ** 2) + self.beta * \
               torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        # print(z.shape)
        z_flattened = z.contiguous().view(-1, self.e_dim)
        # print(z_flattened.shape)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices.reshape(z.shape[0], -1)

    def get_codebook_entry(self, indices):
        """

        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim,)).contiguous()
        return z_q
