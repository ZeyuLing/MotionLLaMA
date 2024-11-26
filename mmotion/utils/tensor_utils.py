from mmcv import to_tensor


def convert_to_tensor(data):
    """ Convert an arbitrary type data into torch.Tensor or its iter element to torch.Tensor
    :param data:
    :return:
    """
    if isinstance(data, dict):
        return {key: convert_to_tensor(value) for key, value in data.items()}
    else:
        try:
            return to_tensor(data)
        except TypeError:
            return data