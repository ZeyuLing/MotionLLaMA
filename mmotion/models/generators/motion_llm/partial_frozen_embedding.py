import torch.nn.functional as F
import torch.nn as nn
import torch

class PartialFrozenEmbedding(nn.Module):
    def __init__(self, ori_emb: nn.Embedding, num_frozen):
        super(PartialFrozenEmbedding, self).__init__()
        self.ori_emb = ori_emb
        self.num_frozen = num_frozen

        # 冻结前 num_frozen 个参数
        self.frozen_weight = nn.Parameter(self.ori_emb.weight.data[:num_frozen].clone(), requires_grad=False)

        # 其余参数
        self.weight = nn.Parameter(self.ori_emb.weight.data[num_frozen:].clone())

    def forward(self, input):
        weight = torch.cat([self.frozen_weight, self.weight], dim=0)
        return F.embedding(
            input, weight, self.ori_emb.padding_idx, self.ori_emb.max_norm,
            self.ori_emb.norm_type, self.ori_emb.scale_grad_by_freq, self.ori_emb.sparse)

class PartialFrozenLinear(nn.Module):
    def __init__(self, ori_linear: nn.Linear, num_frozen: int, bias=True):
        super(PartialFrozenLinear, self).__init__()
        self.ori_linear = ori_linear
        self.num_frozen = num_frozen

        # 冻结前 num_frozen 个参数
        self.frozen_weight = nn.Parameter(self.ori_linear.weight.data[:num_frozen].clone(), requires_grad=False)
        self.weight = nn.Parameter(self.ori_linear.weight.data[num_frozen:].clone())

        # 处理偏置
        if bias and self.ori_linear.bias is not None:
            self.frozen_bias = nn.Parameter(self.ori_linear.bias.data[:num_frozen].clone(), requires_grad=False)
            self.bias = nn.Parameter(self.ori_linear.bias.data[num_frozen:].clone())
        else:
            self.frozen_bias = None
            self.bias = None

    def forward(self, input):

        weight_combined = torch.cat([self.frozen_weight, self.weight], dim=0)
        bias_combined = torch.cat([self.frozen_bias, self.bias], dim=0) if self.frozen_bias is not None else None

        return F.linear(input, weight_combined, bias_combined)

