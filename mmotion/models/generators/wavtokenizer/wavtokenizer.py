import sys
from typing import Union, Any, Tuple, Dict

import os

import torchaudio
import yaml
from einops import rearrange
from torch import nn
import torch

sys.path.append(os.curdir)
from mmotion.datasets.transforms.loading import convert_audio
from mmotion.models.generators.sound_tokenizer.encodec import EncodecFeatures
from mmotion.models.generators.sound_tokenizer.modules.istft_head import ISTFTHead
from mmotion.models.generators.sound_tokenizer.modules.utils import save_audio
from mmotion.models.generators.sound_tokenizer.modules.vocos import VocosBackbone
from mmotion.registry import MODELS


def find_config_and_checkpoint(repo):
    config_path = None
    checkpoint_path = None

    for root, dirs, files in os.walk(repo):
        for file in files:
            if file.endswith(".yaml"):
                config_path = os.path.join(root, file)
            elif file.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, file)

            if config_path and checkpoint_path:
                return config_path, checkpoint_path

    return config_path, checkpoint_path


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


@MODELS.register_module(force=True)
class WavTokenizer(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
            self, pretrained: str = None,
            feature_extractor: EncodecFeatures = None,
            backbone: VocosBackbone = None,
            head: ISTFTHead = None,
    ):
        super().__init__()
        if pretrained is None:
            self.feature_extractor = feature_extractor
            self.backbone = backbone
            self.head = head
        else:
            config_path, checkpoint_path = find_config_and_checkpoint(pretrained)
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.feature_extractor: EncodecFeatures = instantiate_class(args=(), init=config['model']['init_args'][
                "feature_extractor"])
            self.backbone: VocosBackbone = instantiate_class(args=(), init=config['model']['init_args']["backbone"])
            self.head: ISTFTHead = instantiate_class(args=(), init=config['model']['init_args']["head"])

            state_dict_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)['state_dict']
            state_dict = dict()
            for k, v in state_dict_raw.items():
                if k.startswith('backbone.') or k.startswith('head.') or k.startswith('feature_extractor.'):
                    state_dict[k] = v

            self.load_state_dict(state_dict)

    @classmethod
    def from_hparams(cls, config_path: str) -> "Vocos":
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(args=(), init=config['model']['init_args']["feature_extractor"])
        backbone = instantiate_class(args=(), init=config['model']['init_args']["backbone"])
        head = instantiate_class(args=(), init=config['model']['init_args']["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(self, config_path, model_path):
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        model = self.from_hparams(config_path)
        state_dict_raw = torch.load(model_path, map_location="cpu", weights_only=True)['state_dict']
        state_dict = dict()
        for k, v in state_dict_raw.items():
            if k.startswith('backbone.') or k.startswith('head.') or k.startswith('feature_extractor.'):
                state_dict[k] = v

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a copy-synthesis from audio waveform. The feature extractor first processes the audio input,
        which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        features, _, _ = self.feature_extractor(audio_input, **kwargs)  # 0818
        audio_output = self.decode(features, **kwargs)
        return audio_output

    # 0818
    def encode_train(self, audio_input: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        features, discrete_codes, _ = self.feature_extractor(audio_input, **kwargs)
        return features, discrete_codes

    # 0818
    def encode(self, audio_input: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param audio_input: b t
        :param kwargs:
        :return: b d n, b n
        """
        bandwidth_id = torch.tensor([0]).to(audio_input.device)
        features, discrete_codes, _ = self.feature_extractor.infer(audio_input, bandwidth_id=bandwidth_id, **kwargs)
        # remove the num_quantizers dim
        discrete_codes = discrete_codes.squeeze(0)
        return features, discrete_codes

    def decode(self, features_input: torch.Tensor, is_idx: bool = False, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        if is_idx:
            features_input = self.codes_to_features(features_input)
        bandwidth_id = torch.tensor([0]).to(features_input.device)
        x = self.backbone(features_input, bandwidth_id=bandwidth_id, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Transforms an input sequence of discrete tokens (codes) into feature embeddings using the feature extractor's
        codebook weights.

        Args:
            codes (Tensor): The input tensor. Expected shape is (L) or (B, L),
                            where K is the number of codebooks, B is the batch size and L is the sequence length.

        Returns:
            Tensor: Features of shape (B, C, L), where B is the batch size, C denotes the feature dimension,
                    and L is the sequence length.
        """
        assert isinstance(
            self.feature_extractor, EncodecFeatures
        ), "Feature extractor should be an instance of EncodecFeatures"

        if codes.dim() == 1:
            codes = rearrange(codes, 'l -> 1 1 l')
        if codes.dim() == 2:
            codes = rearrange(codes, 'b l -> 1 b l')
        n_bins = self.feature_extractor.encodec.quantizer.bins
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)

        tmp = torch.cat([vq.codebook for vq in self.feature_extractor.encodec.quantizer.vq.layers], dim=0)
        embeddings_idxs = embeddings_idxs.to(tmp.device)
        features = torch.nn.functional.embedding(embeddings_idxs, tmp).sum(dim=0)
        features = features.transpose(1, 2)

        return features

    @property
    def codebook_size(self):
        return self.feature_extractor.codebook_size

    @property
    def downsample_rate(self):
        return self.feature_extractor.downsample_rate


if __name__ == '__main__':
    model = WavTokenizer(pretrained='checkpoints/wav_tokenizer/wav_tokenizer_small_600_24k_4096').cuda()
    audio, sr = torchaudio.load('/data/lzy/projects/motion_llama/data/aist/raw_music/gBR_sBM_cAll_d04_mBR0_ch01.wav')
    audio = convert_audio(audio, sr, 24000, 1).cuda()
    print(audio.shape)
    codes, indices = model.encode(audio)
    print(codes.shape, indices.shape)
    recon = model.decode(indices, is_idx=True).cpu()
    print(recon.shape)
    os.makedirs('vis_results', exist_ok=True)
    save_audio(recon, 'vis_results/res.wav', sample_rate=24000)
