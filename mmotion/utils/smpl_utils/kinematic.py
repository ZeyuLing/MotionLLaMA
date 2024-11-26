from typing import Union, List, Dict

import torch
from smplx import SMPLX
from smplx.utils import SMPLXOutput


def forward_kinematic(smplx_model: SMPLX, smplx_dict: Union[List[Dict], Dict]):
    batch = True
    batch_joints = []
    if not isinstance(smplx_dict, List):
        batch = False
        smplx_dict = [smplx_dict]
    for sd in smplx_dict:
        smplx_output: SMPLXOutput = smplx_model.forward(**sd)
        joints = torch.cat([smplx_output.joints.data[:, :22],
                            smplx_output.joints.data[:, 25:25+30]], dim=1)
        batch_joints.append(joints.detach().cpu().numpy())
    if not batch:
        batch_joints = batch_joints[0]
    return batch_joints
