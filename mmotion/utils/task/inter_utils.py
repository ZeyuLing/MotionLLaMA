# utils for interaction motion tasks
import torch
from einops import repeat

from mmotion.structures import DataSample


def uniform_motion_length(data_sample: DataSample) -> DataSample:
    """ Ensure the pred motions of all persons are uniform
    :param data_sample: output data sample
    :return:
    """
    pred_interactor_motion = data_sample.get('pred_interactor_motion', None)
    if pred_interactor_motion is None:
        # non-inter tasks
        return data_sample
    motion = data_sample.get('pred_motion', data_sample.get('motion'))
    if len(pred_interactor_motion) > len(motion):
        pred_interactor_motion = pred_interactor_motion[:len(motion)]
    elif len(pred_interactor_motion) < len(motion):
        pred_interactor_motion = torch.cat([pred_interactor_motion,
                                            repeat(pred_interactor_motion[-1:], '1 c -> t c',
                                                   t=len(motion)-len(pred_interactor_motion))]
                                           )
    data_sample.pred_interactor_motion = pred_interactor_motion
    return data_sample


