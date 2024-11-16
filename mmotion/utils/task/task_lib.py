from typing import List

from mmotion.utils.task import Task
from mmotion.utils.task.modality import UnionCaption, InteractorCaption, InteractorMotion, Motion, Caption, Audio, \
    Duration, \
    PastMotion, FutureMotion, MiddleMotion, Music, Genre
from mmotion.utils.task.prompt.prompt_template import T2M_TEMPLATE, N2M_TEMPLATE, PRED_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.completion.inbetween import INBETWEEN_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.im2m import IM2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.im2t import IM2T_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.in2m import IN2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.in2stm import IN2STM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.in2t import IN2T_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.in2tm import IN2TM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.in2ustm import IN2USTM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.ist2m import IST2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.interaction_text_motion.it2m import IT2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.d2g import D2G_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.d2m import D2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.g2d import G2D_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.g2dm import G2DM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.g2m import G2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.m2d import M2D_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.m2g import M2G_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.n2d import N2D_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.n2dm import N2DM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.n2gd import N2GD_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.n2gdm import N2GDM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.n2gm import N2GM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.n2music import N2MUSIC_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.music_dance.tm2d import TM2D_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.single_text_motion.l2m import L2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.single_text_motion.l2tm import L2TM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.single_text_motion.lt2m import LT2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.single_text_motion.m2t import M2T_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.single_text_motion.n2t import N2T_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.single_text_motion.n2tm import N2TM_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.speech_gesture.a2g import A2G_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.speech_gesture.g2a import G2A_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.speech_gesture.n2a import N2A_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.speech_gesture.n2ag import N2AG_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.speech_gesture.n2g import N2G_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.speech_gesture.ta2g import TA2G_TEMPLATE


# the keys needed for the tasks to load from dataset

class Caption2Motion(Task):
    abbr = 't2m'
    description = 'caption to motion'
    templates = T2M_TEMPLATE
    num_persons = 1
    input_modality: List = [Caption]
    output_modality: List = [Motion]


class Motion2Text(Task):
    abbr = 'm2t'
    description = 'motion to caption'
    templates = M2T_TEMPLATE
    input_modality: List = [Motion]
    output_modality: List = [Caption]


class Random2Motion(Task):
    abbr = 'n2m'
    description = 'randomly generate motion'
    templates = N2M_TEMPLATE
    output_modality: List = [Motion]


class Random2Caption(Task):
    abbr = 'n2t'
    description = 'randomly generate caption'
    templates = N2T_TEMPLATE
    output_modality: List = [Caption]


class Random2CaptionMotion(Task):
    abbr = 'n2tm'
    description = 'randomly generate motion and caption pairs'
    templates = N2TM_TEMPLATE
    output_modality: List = [Caption, Motion]


class Duration2Motion(Task):
    abbr = 'l2m'
    description = 'generate motion w.r.t given duration'
    templates = L2M_TEMPLATE
    input_modality: List = [Duration]
    output_modality: List = [Motion]


class DurationCaption2Motion(Task):
    abbr = 'lt2m'
    description = 'generate motion w.r.t given duration and caption'
    templates = LT2M_TEMPLATE
    input_modality: List = [Duration, Caption]
    output_modality: List = [Motion]


class Duration2CaptionMotion(Task):
    abbr = 'l2tm'
    description = 'generate motion and corresponding caption w.r.t given duration'
    templates = L2TM_TEMPLATE
    input_modality: List = [Duration]
    output_modality: List = [Motion, Caption]


class Audio2Gesture(Task):
    abbr = 'a2g'
    description = 'generate speaking motion w.r.t speech audio'
    templates = A2G_TEMPLATE
    input_modality: List = [Audio]
    output_modality: List = [Motion]


class Gesture2Audio(Task):
    abbr = 'g2a'
    description = 'Guess speech audio w.r.t speech gestures'
    templates = G2A_TEMPLATE
    input_modality: List = [Motion]
    output_modality: List = [Audio]


class CaptionAudio2Gesture(Task):
    abbr = 'ta2g'
    description = 'Given description and speech audio, infer gestures'
    templates = TA2G_TEMPLATE
    input_modality: List = [Caption, Audio]
    output_modality: List = [Motion]


class Random2Gesture(Task):
    abbr = 'n2g'
    description = 'Randomly generate a piece of co-speech gesture'
    templates = N2G_TEMPLATE
    output_modality: List = [Motion]


class Random2Audio(Task):
    abbr = 'n2a'
    description = 'Randomly generate a piece of co-speech gesture together with speech audio'
    templates = N2A_TEMPLATE
    output_modality: List = [Audio]


class Random2AudioGesture(Task):
    abbr = 'n2ag'
    description = 'Randomly generate a piece of co-speech gesture together with speech audio'
    templates = N2AG_TEMPLATE
    output_modality: List = [Audio, Motion]


class CaptionMusic2Dance(Task):
    abbr = 'tm2d'
    description = 'Generate dance movements according to music and caption.'
    templates = TM2D_TEMPLATE
    input_modality: List = [Caption, Music]
    output_modality: List = [Motion]


class Music2Dance(Task):
    abbr = 'm2d'
    description = 'Generate dance movements according to music.'
    templates = M2D_TEMPLATE
    input_modality: List = [Music]
    output_modality: List = [Motion]


class Music2Genre(Task):
    abbr = 'm2g'
    description = 'Guess the genre from the music piece'
    templates = M2G_TEMPLATE
    input_modality = [Music]
    output_modality = [Genre]


class Genre2Music(Task):
    abbr = 'g2m'
    description = 'make up music w.r.t genre'
    templates = G2M_TEMPLATE
    input_modality = [Genre]
    output_modality = [Music]


class Genre2Dance(Task):
    abbr = 'g2d'
    description = 'make up dance w.r.t genre'
    templates = G2D_TEMPLATE
    input_modality = [Genre]
    output_modality = [Motion]


class Dance2Genre(Task):
    abbr = 'd2g'
    description = 'make up dance w.r.t genre'
    templates = D2G_TEMPLATE
    input_modality = [Motion]
    output_modality = [Genre]


class Dance2Music(Task):
    abbr = 'd2m'
    description = 'Compose music according to dance motion'
    templates = D2M_TEMPLATE
    input_modality: List = [Motion]
    output_modality: List = [Music]


class Genre2DanceMusic(Task):
    abbr = 'g2dm'
    description = 'Compose music and dance according to genre'
    templates = G2DM_TEMPLATE
    input_modality: List = [Genre]
    output_modality: List = [Music, Motion]


class Random2Music(Task):
    abbr = 'n2music'
    description = 'Randomly generate music'
    templates = N2MUSIC_TEMPLATE
    output_modality: List = [Music]


class Random2Dance(Task):
    abbr = 'n2d'
    description = 'Randomly generate dance movements'
    templates = N2D_TEMPLATE
    output_modality: List = [Motion]


class Random2GenreMusic(Task):
    abbr = 'n2gm'
    description = 'Randomly generate genre and music'
    templates = N2GM_TEMPLATE
    output_modality: List = [Genre, Music]


class Random2GenreDance(Task):
    abbr = 'n2gd'
    description = 'Randomly generate dance and genre'
    templates = N2GD_TEMPLATE
    output_modality: List = [Genre, Motion]


class Random2DanceMusic(Task):
    abbr = 'n2dm'
    description = 'Randomly generate music together with dance'
    templates = N2DM_TEMPLATE
    output_modality: List = [Audio, Motion]


class Random2GenreDanceMusic(Task):
    abbr = 'n2gdm'
    description = 'Randomly generate dance and genre'
    templates = N2GDM_TEMPLATE
    output_modality: List = [Genre, Motion, Music]


# Motion Completion task
class MotionPrediction(Task):
    def __init__(self, ratio: float = 0.4):
        self.ratio = ratio

    abbr = 'pred'
    description = 'Motion prediction'
    templates = PRED_TEMPLATE
    input_modality: List = [PastMotion]
    output_modality: List = [FutureMotion]


class MotionInbetween(Task):
    def __init__(self, past_ratio: float = 0.3, future_ratio: float = 0.7):
        self.past_ratio = past_ratio
        self.future_ratio = future_ratio

    abbr = 'inbetween'
    input_modality: List = [PastMotion, FutureMotion]
    output_modality: List = [MiddleMotion]
    templates = INBETWEEN_TEMPLATE


# Multi-person motion
class InterUnionCaption2Motion(Task):
    abbr = 'it2m'
    description = 'Generate interactive motion from a union caption for both persons'
    templates = IT2M_TEMPLATE
    input_modality: List = [UnionCaption]
    output_modality: List = [Motion, InteractorMotion]
    num_persons = 2


class InterMotion2UnionCaption(Task):
    abbr = 'im2t'
    description = 'Describe the interactive motion'
    templates = IM2T_TEMPLATE
    input_modality: List = [Motion, InteractorMotion]
    output_modality: List = [UnionCaption]
    num_persons = 2


class InterSeparateCaption2Motion(Task):
    abbr = 'ist2m'
    description = 'Generate interaction motion from captions of two persons separately'
    num_persons = 2
    templates = IST2M_TEMPLATE
    input_modality: List = [Caption, InteractorCaption]
    output_modality: List = [Motion, InteractorMotion]


class InterMotionSeparateCaption(Task):
    abbr = 'im2st'
    description = 'Describe the interaction motion separately'
    num_persons = 2
    templates = IST2M_TEMPLATE
    input_modality: List = [Motion, InteractorMotion]
    output_modality: List = [Caption, InteractorCaption]


class InterMotion2Motion(Task):
    abbr = 'im2m'
    description = "Generate one person's motion from another"
    templates = IM2M_TEMPLATE
    input_modality: List = [Motion]
    output_modality: List = [InteractorMotion]


class InterRandom2Motion(Task):
    abbr = 'in2m'
    description = 'Randomly generate an interaction motion'
    templates = IN2M_TEMPLATE
    output_modality: List = [Motion, InteractorMotion]


class InterRandom2UnionCaption(Task):
    abbr = 'in2t'
    description = 'Randomly describe an interaction motion'
    templates = IN2T_TEMPLATE
    output_modality: List = [UnionCaption]


class InterRandom2UnionCaptionMotion(Task):
    abbr = 'in2tm'
    description = 'Randomly generate a pair of union caption and interaction motion'
    templates = IN2TM_TEMPLATE
    output_modality: List = [Motion, InteractorMotion, UnionCaption]


class InterRandom2SeparateCaptionMotion(Task):
    abbr = 'in2stm'
    description = 'Randomly generate an interaction motion and separately describe the involved 2 persons'
    templates = IN2STM_TEMPLATE
    output_modality: List = [Motion, InteractorMotion, Caption, UnionCaption]


class InterRandom2UnionSeparateCaptionMotion(Task):
    abbr = 'in2ustm'
    description = 'Randomly generate an interaction motion, and describe the involved 2 persons union and separately'
    templates = IN2USTM_TEMPLATE
    output_modality: List = [Motion, InteractorMotion, Caption, UnionCaption, InteractorCaption]
