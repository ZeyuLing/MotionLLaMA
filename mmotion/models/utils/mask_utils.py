import torch


def create_src_key_padding_mask(lengths, tgt_length=None):
    """ Creates a mask for padded sequence. 0 for valid, 1 for padded.
    :param tgt_length: target sequence length, if None, tgt_length will be set to max(lengths)
    :param lengths: valid length of input sequence. [B,]
    :return:
    """
    batch_size = len(lengths)
    if tgt_length is None:
        tgt_length = max(lengths)
    seq_range = torch.arange(tgt_length).unsqueeze(0).expand(batch_size, tgt_length)
    lengths_tensor = torch.tensor(lengths).unsqueeze(1)
    mask = seq_range >= lengths_tensor
    return mask.to(torch.bool)  # dtype: torch.bool
