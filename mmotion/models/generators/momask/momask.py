from typing import Dict

import os

from mmengine import Config
from mmengine.model import BaseModel

from mmotion.models.generators.momask.temporal_transformer import MomaskTemporalTransformer
from mmotion.models.generators.momask.residual_transformer import MomaskResidualTransformer
from mmotion.registry import MODELS
from mmotion.structures import DataSample


@MODELS.register_module()
class Momask(BaseModel):
    def __init__(self, temp_transformer: Dict, res_transformer: Dict, data_preprocessor=None, init_cfg=None):
        super(Momask, self).__init__(data_preprocessor, init_cfg)
        self.temp_transformer: MomaskTemporalTransformer = self.build_transformer(temp_transformer)
        self.res_transformer: MomaskResidualTransformer = self.build_transformer(res_transformer)

    def build_transformer(self, transformer_cfg: Dict):
        type = transformer_cfg['type']
        if os.path.isfile(type):
            # allow using config file path as type, simplify config writing.
            init_cfg = transformer_cfg.pop('init_cfg', None)
            transformer_cfg = Config.fromfile(type)['model']
            if init_cfg is not None:
                transformer_cfg['init_cfg'] = init_cfg

        transformer = MODELS.build(transformer_cfg).eval()
        if transformer_cfg.get('init_cfg', None) is not None:
            transformer.init_weights()
        return transformer

    def forward(self, inputs, data_samples, mode='predict'):
        if mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        raise NotImplementedError(f"{mode} not implemented, only 'predict' is supported")

    def forward_predict(self, inputs: Dict, data_samples: DataSample):
        caption = data_samples.get('caption')
        num_frames = data_samples.get('num_frames')
        num_tokens = [nf // self.temp_transformer.rvq.downsample_rate for nf in num_frames]
        cond_token = self.temp_transformer.encode_text(caption)
        pred_top_motion_id = self.temp_transformer.generate_tokens(cond_token, num_tokens)
        pred_motion_ids = self.res_transformer.generate_tokens(pred_top_motion_id, cond_token, num_tokens)
        pred_motion = self.temp_transformer.rvq.decode(pred_motion_ids, is_idx=True)
        data_samples.set_field(pred_motion, 'pred_motion')
        data_samples.set_data(inputs)
        for key, value in data_samples.to_dict().items():
            if key.endswith('motion') and value is not None:
                value = self.data_preprocessor.destruct(value, data_samples)
                data_samples.set_field(value, key)
                joints_key = key.replace('motion', 'joints')
                joints = self.data_preprocessor.vec2joints(value, data_samples)
                data_samples.set_field(joints, joints_key)
        data_samples = data_samples.split(allow_nonseq_value=True)
        return data_samples
