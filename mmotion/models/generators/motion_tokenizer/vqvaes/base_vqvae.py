import numpy as np
from typing import Dict, Union, List

import torch
from einops import rearrange
from mmengine.model import BaseModel, BaseDataPreprocessor
from torch import Tensor

from mmotion.models.archs.quantizers import EMAResetQuantizer
from mmotion.registry import MODELS
from mmotion.structures import DataSample
from mmotion.utils.typing import SampleList


@MODELS.register_module()
class BaseVQVAE(BaseModel):

    def __init__(self,
                 quantizer: Dict = None,
                 encoder: Dict = None,
                 decoder: Dict = None,
                 loss_cfg: Dict = None,
                 data_preprocessor: BaseDataPreprocessor = dict(
                     type='BaseDataPreprocessor'),
                 init_cfg=None) -> None:
        """
        :param nfeats: input feature dimension
        :param quantizer: quantizer config.
        :param loss_cfg: loss function config
        """
        super().__init__(data_preprocessor, init_cfg)
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)

        if loss_cfg is not None:
            self.loss_fn = MODELS.build(loss_cfg)

        self.quantizer: EMAResetQuantizer = MODELS.build(quantizer) if quantizer else None

    def forward(self,
                inputs: torch.Tensor,
                data_samples: DataSample = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        if mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'loss':
            return self.forward_loss(inputs, data_samples)
        elif mode == 'tensor':
            return self.forward_tensor(inputs, data_samples)
        else:
            raise NotImplementedError(
                'Forward is not implemented now, please use infer.')

    def preprocess(self, x):
        # (bs, T, C) -> (bs, C, T)
        x = rearrange(x, 'b t c -> b c t')
        return x

    def postprocess(self, x):
        # (bs, C, T) ->  (bs, T, C)
        x = x.permute(0, 2, 1)
        return x

    def forward_encode_decode(self, inputs, data_samples):
        x = inputs['motion']
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        x_quantized, _, commit_loss, perplexity = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return {
            'gt_motion': x,
            'pred_motion': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity
        }

    def forward_tensor(self, inputs, data_samples) -> Dict:
        return self.forward_encode_decode(inputs, data_samples)

    def forward_loss(self, inputs, data_samples) -> Dict:
        output_dict = self.forward_tensor(inputs, data_samples)
        loss_dict = self.loss_fn(inputs, output_dict, data_samples)
        return loss_dict

    def encode(self, x: Tensor) -> Tensor:
        """ encode the input features to latent features and code idx
        :param x: raw motion rep in [b, t, c]
        :return: code index in shape [n,c] or [q,n,c] q represents num of codebooks
        """
        x = rearrange(x, 'b t c -> b c t')
        x_encoder = self.encoder(x)
        x_encoder = rearrange(x_encoder, 'b c t -> b t c')
        code_idx = self.quantizer.quantize(x_encoder)   # n c or q n c
        # latent, dist
        return code_idx

    def decode(self, z: Union[List[List[int]], Tensor], is_idx: bool = False) -> Tensor:
        """ decode code index to motion rep.
        :param z: code index[B N] or latent feature[B T C]
        :param is_idx: If z is a tensor of code index, is_idx == True; else z is a latent motion feature, shape in [b t c]
        :return:
        """
        if is_idx:
            z = self.quantizer.dequantize(z)
        z = rearrange(z, 'b t c -> b c t')
        # decoder
        z = self.decoder(z)
        z = rearrange(z, 'b c t -> b t c')
        return z

    @torch.no_grad()
    def forward_predict(self, inputs: Dict, data_samples: DataSample) -> SampleList:
        """
        :param data_samples: data samples
        :param inputs: input data
        :return: output dict
        """
        # b t c
        output_dict = self.forward_tensor(inputs, data_samples)

        output_dict.pop('perplexity', None)
        output_dict.pop('commit_loss', None)
        output_dict['pred_motion'] = self.data_preprocessor.destruct(output_dict['pred_motion'], data_samples)
        output_dict['gt_motion'] = self.data_preprocessor.destruct(output_dict['gt_motion'], data_samples)
        out_data_sample = DataSample(
            **output_dict,
            **data_samples.to_dict()
        )

        data_sample_list = out_data_sample.split(allow_nonseq_value=True)
        # batch smpl dict will be split to a batch list of smpl dicts
        return data_sample_list

    def decompose_vector(self, output_dict: Dict, data_sample: DataSample) -> Dict:
        """ Get joints, rotation, feet_contact, ... from predicted motion vectors.
        """
        normalized_gt_motion = output_dict["gt_motion"]
        normalized_pred_motion = output_dict["pred_motion"]

        gt_motion = self.data_preprocessor.destruct(normalized_gt_motion, data_sample)
        pred_motion = self.data_preprocessor.destruct(normalized_pred_motion, data_sample)

        gt_joints = self.data_preprocessor.vec2joints(gt_motion, data_sample)
        pred_joints = self.data_preprocessor.vec2joints(pred_motion, data_sample)

        gt_transl = gt_joints[..., 0, :]
        pred_transl = pred_joints[..., 0, :]

        gt_rotation = self.data_preprocessor.vec2rotation(gt_motion, data_sample)
        pred_rotation = self.data_preprocessor.vec2rotation(pred_motion, data_sample)

        output_dict.update(
            {
                'gt_joints': gt_joints,
                'pred_joints': pred_joints,
                'gt_transl': gt_transl,
                'pred_transl': pred_transl,
                'gt_rotation': gt_rotation,
                'pred_rotation': pred_rotation

            }
        )
        return output_dict

    @property
    def codebook_size(self):
        return self.quantizer.codebook_size

    @property
    def code_dim(self):
        return self.quantizer.code_dim

    @property
    def motion_dim(self):
        return self.encoder.motion_dim

    @property
    def downsample_rate(self):
        return self.encoder.downsample_rate
