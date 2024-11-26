"""
    MotionHub datasets involves multi-modal data.
    Each modal can be reached via the following keys in the annotation files(train.json for exp).
"""
motion_keys = ['motion', 'interhuman', 'humanml3d']

caption_keys = ['caption']
audio_keys = ['audio']
music_keys = ['music']
union_caption_keys = ['union_caption']
duration_keys = ['duration']
script_keys = ['script']
genre_keys = ['genre']
