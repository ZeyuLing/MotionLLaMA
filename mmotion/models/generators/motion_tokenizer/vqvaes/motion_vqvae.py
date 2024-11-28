# Partially from https://github.com/Mael-zys/T2M-GPT
import sys
from os.path import join, basename
from typing import Dict
import numpy as np

import fire
import torch
import os

from einops import rearrange, repeat

sys.path.append(os.curdir)

from mmotion.registry import MODELS
from mmotion.models.generators.motion_tokenizer.vqvaes.base_vqvae import BaseVQVAE
from mmotion.structures import DataSample
from mmotion.utils.typing import SampleList

from mmengine import Config, init_default_scope

init_default_scope('mmotion')


@MODELS.register_module(force=True)
class MotionVQVAE(BaseVQVAE):
    """
        Vqvae for humanml3d representation
    """

    def forward_tensor(self, inputs: Dict, data_samples: DataSample) -> SampleList:
        output_dict = self.forward_encode_decode(inputs, data_samples)

        return self.decompose_vector(output_dict, data_samples)

    @torch.no_grad()
    def forward_predict(self, inputs: Dict, data_samples: DataSample) -> SampleList:
        output_dict = self.forward_tensor(inputs, data_samples)

        output_dict.pop('perplexity', None)
        output_dict.pop('commit_loss', None)

        data_samples.set_data(output_dict)

        data_sample_list = data_samples.split(allow_nonseq_value=True)
        # batch smpl dict will be split to a batch lis+t of smpl dicts
        return data_sample_list


def main(cfg: str = 'configs/vqvae/homi_vqvae/homi_vq_64_2048code_1536dim_3depth.py',
         checkpoint: str = '/data/lzy/projects/motion_llama/work_dirs/homi_gate_encoder_out_gate_64_2048code_1536dim_3depth/best_MPJPE_epoch_5000.pth',
         vis_dir='tmp',
         sample: str = 'data/motionhub/motionx/motion_data/interhuman/aist/subset_0001/Dance_Ballet_Jazz_Chaines.npy'):
    os.makedirs(vis_dir, exist_ok=True)
    save_path = join(vis_dir, basename(sample).replace('.npy', '.mp4'))
    cfg = Config.fromfile(cfg)['model']
    cfg['init_cfg'] = dict(
        type='Pretrained',
        checkpoint=checkpoint
    )
    motion_vqvae: MotionVQVAE = MODELS.build(cfg).to(torch.float).cuda().eval()
    motion_vqvae = motion_vqvae.bfloat16()
    motion_vqvae.init_weights()

    motion = torch.from_numpy(np.load(sample)[..., :156])[None].cuda().bfloat16()
    motion, _ = motion_vqvae.data_preprocessor._preprocess_motion_batch(motion, None)
    B, T, C = motion.shape
    padding = torch.zeros_like(motion[:, -1:])
    padding = repeat(padding, '1 1 c -> 1 t c', t=200)
    motion = torch.cat([motion, padding], dim=1)
    index = motion_vqvae.encode(motion)
    index[:, -50:] = index[:, -52:-51]

    motion = motion_vqvae.decode(index, is_idx=True)[0]
    motion = motion_vqvae.data_preprocessor.destruct(motion, None)
    motion = rearrange(motion, 't (j c) -> t j c', c=3)
    motion = motion.detach().float().cpu().numpy()[:T]

    from mmotion.core.visualization import visualize_kp3d
    visualize_kp3d(
        motion,
        frame_names='',
        convention='blender',
        output_path=save_path,
        resolution=(1024, 1024),
        data_source='smplh')


if __name__ == '__main__':
    fire.Fire(main)
