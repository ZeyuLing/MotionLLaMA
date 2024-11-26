import pickle
from torch import nn
import torch

from mmotion.models.generators.emage.mlp import MLP
from mmotion.models.generators.emage.period_pe import PeriodicPositionalEncoding
from mmotion.models.generators.emage.vq import MotionEncoder
from mmotion.models.generators.emage.wav_encoder import WavEncoder


class MAGE_Transformer(nn.Module):
    def __init__(self, lang_model: str = 'checkpoints/weights/vocab.pkl',
                 input_feats: int = 312,
                 audio_f: int = 256,
                 codebook_size: int = 256,
                 motion_f: int = 256,
                 smpl_dim: int = 312,  # 52 * 6
                 hidden_size: int = 768):
        super(MAGE_Transformer, self).__init__()
        self.hidden_size = hidden_size
        with open(lang_model, 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights

        self.text_pre_encoder_body = nn.Embedding.from_pretrained(
            torch.FloatTensor(pre_trained_embedding), freeze=False)

        self.text_encoder_body = nn.Linear(300, audio_f)
        self.text_encoder_body = nn.Linear(300, audio_f)

        self.audio_pre_encoder_body = WavEncoder(audio_f, audio_in=2)

        self.at_attn_body = nn.Linear(audio_f * 2, audio_f * 2)

        self.motion_encoder = MotionEncoder(
            input_feats=input_feats,
            num_layers=3,
            hidden_size=hidden_size
        )  # masked motion to latent bs t 333 to bs t 256

        # face decoder
        self.feature2face = nn.Linear(audio_f * 2, hidden_size)
        self.face2latent = nn.Linear(hidden_size, codebook_size)
        transformer_de_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            batch_first=True
        )
        self.face_decoder = nn.TransformerDecoder(transformer_de_layer, num_layers=4)
        self.position_embeddings = PeriodicPositionalEncoding(hidden_size, period=300)

        # motion decoder
        transformer_en_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            batch_first=True
        )
        self.motion_self_encoder = nn.TransformerEncoder(transformer_en_layer, num_layers=1)
        self.audio_feature2motion = nn.Linear(audio_f, hidden_size)
        self.feature2motion = nn.Linear(motion_f, hidden_size)

        self.bodyhints_body = MLP(motion_f, hidden_size, motion_f)
        self.motion2latent_upper = MLP(hidden_size, hidden_size, hidden_size)
        self.motion2latent_hands = MLP(hidden_size, hidden_size, hidden_size)
        self.motion2latent_lower = MLP(hidden_size, hidden_size, hidden_size)
        self.wordhints_decoder = nn.TransformerDecoder(transformer_de_layer, num_layers=8)

        self.upper_decoder = nn.TransformerDecoder(transformer_de_layer, num_layers=1)
        self.hands_decoder = nn.TransformerDecoder(transformer_de_layer, num_layers=1)
        self.lower_decoder = nn.TransformerDecoder(transformer_de_layer, num_layers=1)

        self.upper_classifier = MLP(codebook_size, hidden_size, codebook_size)
        self.hands_classifier = MLP(codebook_size, hidden_size, codebook_size)
        self.lower_classifier = MLP(codebook_size, hidden_size, codebook_size)

        self.mask_embeddings = nn.Parameter(torch.zeros(1, 1, smpl_dim + 3 + 4))
        self.motion_down_upper = nn.Linear(hidden_size, codebook_size)
        self.motion_down_hands = nn.Linear(hidden_size, codebook_size)
        self.motion_down_lower = nn.Linear(hidden_size, codebook_size)
        self._reset_parameters()

        self.spearker_encoder_body = nn.Embedding(25, hidden_size)

    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.hidden_size ** -0.5)

    def forward(self, audio, text, speaker_id, target_frames, mask=None):

        text_body = self.text_pre_encoder_body(text)
        text_body = self.text_encoder_body(text_body)

        audio_body = self.audio_pre_encoder_body(audio)
        if audio_body.shape[1] != target_frames:
            diff_length = target_frames - audio_body.shape[1]
            if diff_length < 0:
                audio_body = audio_body[:, :diff_length, :]
            else:
                audio_body = torch.cat((audio_body, audio_body[:, -diff_length:]), 1)
        bs, t, c = audio_body.shape
        alpha_at_body = torch.cat([text_body, audio_body], dim=-1).reshape(bs, t, c * 2)
        alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
        alpha_at_body = alpha_at_body.softmax(dim=-1)
        fusion_body = text_body * alpha_at_body[:, :, :, 1] + audio_body * alpha_at_body[:, :, :, 0]

        masked_embeddings = self.mask_embeddings.expand_as(in_motion)
        masked_motion = torch.where(mask == 1, masked_embeddings, in_motion)  # bs, t, 256
        body_hint = self.motion_encoder(masked_motion)  # bs t 256
        speaker_embedding_body = self.spearker_encoder_body(speaker_id).squeeze(2)

        # motion spatial encoder
        body_hint_body = self.bodyhints_body(body_hint)
        motion_embeddings = self.feature2motion(body_hint_body)
        motion_embeddings = speaker_embedding_body + motion_embeddings
        motion_embeddings = self.position_embeddings(motion_embeddings)

        # bi-directional self-attention
        motion_refined_embeddings = self.motion_self_encoder(motion_embeddings)

        # audio to gesture cross-modal attention
        a2g_motion = self.audio_feature2motion(fusion_body)
        motion_refined_embeddings_in = self.position_embeddings(motion_refined_embeddings)
        word_hints = self.wordhints_decoder(tgt=motion_refined_embeddings_in, memory=a2g_motion)
        motion_refined_embeddings = motion_refined_embeddings + word_hints

        # feedforward
        upper_latent = self.motion2latent_upper(motion_refined_embeddings)
        hands_latent = self.motion2latent_hands(motion_refined_embeddings)
        lower_latent = self.motion2latent_lower(motion_refined_embeddings)

        upper_latent_in = upper_latent + speaker_embedding_body
        upper_latent_in = self.position_embeddings(upper_latent_in)
        hands_latent_in = hands_latent + speaker_embedding_body
        hands_latent_in = self.position_embeddings(hands_latent_in)
        lower_latent_in = lower_latent + speaker_embedding_body
        lower_latent_in = self.position_embeddings(lower_latent_in)

        # transformer decoder
        motion_upper = self.upper_decoder(tgt=upper_latent_in, memory=hands_latent + lower_latent)
        motion_hands = self.hands_decoder(tgt=hands_latent_in, memory=upper_latent + lower_latent)
        motion_lower = self.lower_decoder(tgt=lower_latent_in, memory=upper_latent + hands_latent)
        upper_latent = self.motion_down_upper(motion_upper + upper_latent)
        hands_latent = self.motion_down_hands(motion_hands + hands_latent)
        lower_latent = self.motion_down_lower(motion_lower + lower_latent)
        logits_lower = self.lower_classifier(lower_latent)
        logits_upper = self.upper_classifier(upper_latent)
        logits_hands = self.hands_classifier(hands_latent)

        return {
            "rec_upper": upper_latent,
            "rec_lower": lower_latent,
            "rec_hands": hands_latent,
            "logits_upper": logits_upper,
            "logits_lower": logits_lower,
            "logits_hands": logits_hands,
        }
