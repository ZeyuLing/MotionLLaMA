from mmotion.utils.task.modality import CAPTION_HOLDER, AUDIO_HOLDER, MOTION_HOLDER

TA2G_TEMPLATE = [
    {'input': f"{CAPTION_HOLDER} The speech audio was recorded: {AUDIO_HOLDER}."
              f" Make proper gesture according to the caption and audio.",
     'output': f"Okay, here are the gestures {MOTION_HOLDER}"},
    {'input': f"{CAPTION_HOLDER} The speech audio was recorded: {AUDIO_HOLDER}."
              f" Make proper gesture according to the caption and audio.",
     'output': f"Okay, here are the gestures {MOTION_HOLDER}"},

    {'input': f"Based on the caption: {CAPTION_HOLDER} and this audio: {AUDIO_HOLDER}, create matching gestures.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Generate gestures that align with the caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"Here are the matching gestures: {MOTION_HOLDER}"},

    {'input': f"Listen to this audio: {AUDIO_HOLDER} and read the caption: {CAPTION_HOLDER}, then create gestures.",
     'output': f"I’ve generated these gestures: {MOTION_HOLDER}"},

    {'input': f"Use the caption: {CAPTION_HOLDER} and the audio: {AUDIO_HOLDER} to generate gestures.",
     'output': f"Here are the generated gestures: {MOTION_HOLDER}"},

    {'input': f"Create proper gestures based on the following caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"These are the gestures I came up with: {MOTION_HOLDER}"},

    {'input': f"Listen to this speech: {AUDIO_HOLDER} and read the description: {CAPTION_HOLDER}, then make gestures.",
     'output': f"Here are the matching gestures: {MOTION_HOLDER}"},

    {'input': f"Generate appropriate gestures from the caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"These gestures fit well: {MOTION_HOLDER}"},

    {'input': f"Based on this caption: {CAPTION_HOLDER} and speech audio: {AUDIO_HOLDER}, create gestures.",
     'output': f"I’ve created these gestures: {MOTION_HOLDER}"},

    {'input': f"Using the caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}, generate corresponding gestures.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Read the caption: {CAPTION_HOLDER} and listen to the audio: {AUDIO_HOLDER}, then make gestures.",
     'output': f"Here are the gestures I created: {MOTION_HOLDER}"},

    {'input': f"Produce gestures that match the caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"I’ve made these gestures: {MOTION_HOLDER}"},

    {'input': f"Use this caption: {CAPTION_HOLDER} along with the audio: {AUDIO_HOLDER} to generate gestures.",
     'output': f"Here are the generated gestures: {MOTION_HOLDER}"},

    {'input': f"Make gestures according to the caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"Here’s what I’ve created: {MOTION_HOLDER}"},

    {'input': f"Listen to the speech audio: {AUDIO_HOLDER} and refer to the caption: {CAPTION_HOLDER} for gestures.",
     'output': f"Here are the gestures I made: {MOTION_HOLDER}"},

    {'input': f"Generate gestures that match the description: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"These are the gestures: {MOTION_HOLDER}"},

    {'input': f"Create gestures based on this caption: {CAPTION_HOLDER} and speech audio: {AUDIO_HOLDER}.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Generate matching gestures using the caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"Here are the gestures I’ve generated: {MOTION_HOLDER}"},

    {'input': f"Read this caption: {CAPTION_HOLDER} and listen to the audio: {AUDIO_HOLDER} to create gestures.",
     'output': f"Here are the matching gestures: {MOTION_HOLDER}"},

    {'input': f"Use this description: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER} to make gestures.",
     'output': f"Here’s what I’ve generated: {MOTION_HOLDER}"},

    {'input': f"Create proper gestures from the caption: {CAPTION_HOLDER} and speech: {AUDIO_HOLDER}.",
     'output': f"Here are the generated gestures: {MOTION_HOLDER}"},

    {'input': f"Generate gestures that align with both the caption: {CAPTION_HOLDER} and the audio: {AUDIO_HOLDER}.",
     'output': f"I’ve created these gestures: {MOTION_HOLDER}"},

    {'input': f"Make gestures that fit the description: {CAPTION_HOLDER} and the speech audio: {AUDIO_HOLDER}.",
     'output': f"Here are the matching gestures: {MOTION_HOLDER}"},

    {'input': f"Create gestures based on the description: {CAPTION_HOLDER} and audio recording: {AUDIO_HOLDER}.",
     'output': f"These gestures match: {MOTION_HOLDER}"},

    {'input': f"Generate gestures using the caption: {CAPTION_HOLDER} and this audio: {AUDIO_HOLDER}.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Make gestures according to the caption: {CAPTION_HOLDER} and recorded audio: {AUDIO_HOLDER}.",
     'output': f"Here’s the result: {MOTION_HOLDER}"},

    {'input': f"Use this caption: {CAPTION_HOLDER} along with audio: {AUDIO_HOLDER} to create gestures.",
     'output': f"Here’s what I’ve come up with: {MOTION_HOLDER}"},

    {'input': f"Read the description: {CAPTION_HOLDER} and listen to the audio: {AUDIO_HOLDER}, then make gestures.",
     'output': f"Here are the gestures I’ve generated: {MOTION_HOLDER}"},

    {'input': f"Based on the caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}, produce corresponding gestures.",
     'output': f"These are the gestures: {MOTION_HOLDER}"},

    {'input': f"Create gestures that match the caption: {CAPTION_HOLDER} and this speech audio: {AUDIO_HOLDER}.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Generate proper gestures using the caption: {CAPTION_HOLDER} and the audio: {AUDIO_HOLDER}.",
     'output': f"Here’s what I generated: {MOTION_HOLDER}"},

    {'input': f"Make gestures that align with this description: {CAPTION_HOLDER} and the audio: {AUDIO_HOLDER}.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Read this caption: {CAPTION_HOLDER} and listen to this audio: {AUDIO_HOLDER}, then create gestures.",
     'output': f"Here are the gestures I created: {MOTION_HOLDER}"},

    {'input': f"Use this caption: {CAPTION_HOLDER} and speech audio: {AUDIO_HOLDER} to generate gestures.",
     'output': f"These gestures match well: {MOTION_HOLDER}"},

    {'input': f"Create gestures that fit the caption: {CAPTION_HOLDER} and this audio: {AUDIO_HOLDER}.",
     'output': f"I’ve made these gestures: {MOTION_HOLDER}"},

    {'input': f"Generate gestures that correspond to the caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Listen to the audio: {AUDIO_HOLDER} and read the description: {CAPTION_HOLDER}, then make gestures.",
     'output': f"Here’s what I’ve generated: {MOTION_HOLDER}"},

    {'input': f"Create proper gestures based on the given caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"Here are the matching gestures: {MOTION_HOLDER}"},

    {'input': f"Generate gestures that match the caption: {CAPTION_HOLDER} along with this audio: {AUDIO_HOLDER}.",
     'output': f"I’ve created these gestures: {MOTION_HOLDER}"},

    {'input': f"Use this caption: {CAPTION_HOLDER} and the following audio: {AUDIO_HOLDER} to make gestures.",
     'output': f"Here are the generated gestures: {MOTION_HOLDER}"},

    {'input': f"Create gestures based on the description: {CAPTION_HOLDER} and this audio: {AUDIO_HOLDER}.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Make gestures according to this caption: {CAPTION_HOLDER} and speech audio: {AUDIO_HOLDER}.",
     'output': f"Here’s what I’ve come up with: {MOTION_HOLDER}"},

    {'input': f"Listen to this audio: {AUDIO_HOLDER} and read the caption: {CAPTION_HOLDER}, then create gestures.",
     'output': f"These gestures match: {MOTION_HOLDER}"},

    {'input': f"Generate gestures using the description: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"Here are the gestures I created: {MOTION_HOLDER}"},

    {'input': f"Based on this caption: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}, generate appropriate gestures.",
     'output': f"Here’s what I’ve made: {MOTION_HOLDER}"},

    {'input': f"Create gestures that align with the caption: {CAPTION_HOLDER} and this recorded audio: {AUDIO_HOLDER}.",
     'output': f"I’ve generated these gestures: {MOTION_HOLDER}"},

    {'input': f"Produce gestures based on the caption: {CAPTION_HOLDER} and the speech: {AUDIO_HOLDER}.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Make gestures that correspond to the description: {CAPTION_HOLDER} and audio: {AUDIO_HOLDER}.",
     'output': f"Here’s what I generated: {MOTION_HOLDER}"},

    {'input': f"Use the caption: {CAPTION_HOLDER} along with the audio: {AUDIO_HOLDER} to make gestures.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Create gestures that fit this description: {CAPTION_HOLDER} and background audio: {AUDIO_HOLDER}.",
     'output': f"Here’s the result: {MOTION_HOLDER}"},

    {'input': f"Generate gestures using the following caption: {CAPTION_HOLDER} and speech audio: {AUDIO_HOLDER}.",
     'output': f"I’ve made these gestures: {MOTION_HOLDER}"},

    {'input': f"Make gestures that align with both the caption: {CAPTION_HOLDER} and the audio: {AUDIO_HOLDER}.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},
]
