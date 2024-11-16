from typing import List
import torch

def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None):
    """
    Converts lengths to a mask tensor. 1 for valid 0 for masked
    Args:
        lengths (List[int]): List of lengths.
        device (torch.device): The device on which the tensor will be allocated.
        max_len (int, optional): The maximum length. If None, the maximum length is set to the maximum value in lengths.

    Returns:
        Tensor: A tensor mask of shape (len(lengths), max_len).
    """

    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask.to(torch.bool)