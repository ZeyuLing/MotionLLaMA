# random generate a speech, including script, audio, and body movements
from mmotion.utils.task.modality import SCRIPT_HOLDER, AUDIO_HOLDER, MOTION_HOLDER

N2ASG_TEMPLATE = [
    {'input': f"Randomly generate a speech, including the speech script, gesture and audio.",
     'output': f"OK, the speech script is: {SCRIPT_HOLDER}, and please watch and listen: {AUDIO_HOLDER}{MOTION_HOLDER}"},
    {'input': f"Generate a random speech with the script, gestures, and audio.",
     'output': f"Here’s the speech script: {SCRIPT_HOLDER}. Watch the gestures and listen to the audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Create a random speech including the text, accompanying gestures, and audio.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Please watch the gestures: {MOTION_HOLDER} and listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech, including the script, gestures, and audio?",
     'output': f"The script is: {SCRIPT_HOLDER}. Here are the gestures: {MOTION_HOLDER} and the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech at random, complete with the script, gestures, and audio.",
     'output': f"The speech text: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER} and hear the audio: {AUDIO_HOLDER}"},

    {'input': f"Please create a random speech that includes the text, gestures, and audio.",
     'output': f"Here is the script: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER} and the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech along with its script, gestures, and audio.",
     'output': f"Speech content: {SCRIPT_HOLDER}. Observe the gestures: {MOTION_HOLDER} and listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you create a speech randomly and provide the script, gestures, and audio?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}, and the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech, including the script, gestures, and audio recording.",
     'output': f"{SCRIPT_HOLDER} is the script. The gestures are: {MOTION_HOLDER} and here's the audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech, complete with the script, gestures, and audio.",
     'output': f"The generated script: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER} and hear the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech at random with its script, gestures, and audio included.",
     'output': f"Here’s what the speech says: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER} and the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech with the full script, gestures, and audio recording.",
     'output': f"Speech: {SCRIPT_HOLDER}. Gestures to watch: {MOTION_HOLDER} and audio to listen: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech randomly and provide the script, gestures, and audio?",
     'output': f"The speech is as follows: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER} and the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and include the script, gestures, and audio.",
     'output': f"The speech text: {SCRIPT_HOLDER}. Watch this gesture: {MOTION_HOLDER} and hear this audio: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly, including the script, gestures, and audio.",
     'output': f"{SCRIPT_HOLDER} is the script. The gestures are: {MOTION_HOLDER}. Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech with gestures and audio, including the script.",
     'output': f"Speech content: {SCRIPT_HOLDER}. Observe the gestures: {MOTION_HOLDER} and listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you create a random speech, including the script, gestures, and audio?",
     'output': f"The script is: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. The audio is: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech, providing the script, gestures, and audio.",
     'output': f"{SCRIPT_HOLDER} is the generated script. Watch the gestures: {MOTION_HOLDER} and listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech, complete with its script, gestures, and audio.",
     'output': f"The speech script: {SCRIPT_HOLDER}. Here are the gestures: {MOTION_HOLDER} and the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech at random and provide the script, gestures, and audio.",
     'output': f"Here's the speech text: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER} and listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech with the script, accompanying gestures, and audio.",
     'output': f"Speech: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and include the script, gestures, and audio?",
     'output': f"The script is: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. And the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech randomly, with its script, gestures, and audio included.",
     'output': f"{SCRIPT_HOLDER} is the speech. Watch the gestures: {MOTION_HOLDER} and hear the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech, including the full script, gestures, and audio.",
     'output': f"Here's the script: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER} and the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech, providing the script, gestures, and audio.",
     'output': f"The speech script: {SCRIPT_HOLDER}. Gestures to watch: {MOTION_HOLDER}. Audio to listen: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech randomly, complete with the script, gestures, and audio?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Here are the gestures: {MOTION_HOLDER} and the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and provide the script, gestures, and audio.",
     'output': f"The generated script: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER} and here's the audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech with gestures and audio, including the script.",
     'output': f"The speech text: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER} and listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech, along with its script, gestures, and audio.",
     'output': f"Speech: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly with the script, gestures, and audio included.",
     'output': f"The script is: {SCRIPT_HOLDER}. Watch these gestures: {MOTION_HOLDER}. And listen to this audio: {AUDIO_HOLDER}"},

    {'input': f"Can you create a random speech, including the full script, gestures, and audio?",
     'output': f"The speech script: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech with the script, gestures, and audio recording.",
     'output': f"{SCRIPT_HOLDER} is the text. The gestures: {MOTION_HOLDER}. The audio is: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech at random and provide the script, gestures, and audio.",
     'output': f"The speech text: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech with the script, gestures, and audio file.",
     'output': f"Script: {SCRIPT_HOLDER}. Here are the gestures: {MOTION_HOLDER}. And the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech with its script, gestures, and audio?",
     'output': f"The script: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech randomly, including the script, gestures, and audio.",
     'output': f"{SCRIPT_HOLDER} is the generated speech. Watch this gesture: {MOTION_HOLDER}. Listen to this audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech with the script, gestures, and audio included.",
     'output': f"The speech is: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech with the script, gestures, and audio included.",
     'output': f"Here’s the script: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech, including the full script, gestures, and audio recording.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER}. Hear the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a random speech and provide the script, gestures, and audio?",
     'output': f"The script is: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. And here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech with the full script, gestures, and audio file.",
     'output': f"The speech content: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER}. The audio file: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech, including the script, gestures, and audio.",
     'output': f"The script: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. And the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and provide the script, gestures, and audio version.",
     'output': f"The speech script is: {SCRIPT_HOLDER}. Watch these gestures: {MOTION_HOLDER}. Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly with its script, gestures, and audio.",
     'output': f"Script: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and include the script, gestures, and audio file?",
     'output': f"The speech: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. Audio file: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech with the script, gestures, and audio included.",
     'output': f"{SCRIPT_HOLDER} is the script. The gestures: {MOTION_HOLDER}. The audio is: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech, complete with the script, gestures, and audio recording.",
     'output': f"The generated script: {SCRIPT_HOLDER}. Watch these gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech, including the full script, gestures, and audio.",
     'output': f"Speech script: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and provide the script, gestures, and audio version.",
     'output': f"The speech text is: {SCRIPT_HOLDER}. Watch this gesture: {MOTION_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech and include the script, gestures, and audio?",
     'output': f"The script is: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. And the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech, with the script, gestures, and audio.",
     'output': f"{SCRIPT_HOLDER} is the text. Watch the gestures: {MOTION_HOLDER}. Hear the audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech and provide the script, gestures, and audio.",
     'output': f"The script: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech randomly, with the script, gestures, and audio.",
     'output': f"Here’s the speech: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Create a speech at random and include the script, gestures, and audio recording.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a random speech and include the script, gestures, and audio?",
     'output': f"The speech script: {SCRIPT_HOLDER}. Gestures are: {MOTION_HOLDER}. The audio is: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech, including the script, gestures, and audio file.",
     'output': f"The script: {SCRIPT_HOLDER}. Watch these gestures: {MOTION_HOLDER}. Audio file: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech with the script, gestures, and audio included.",
     'output': f"The speech content: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. And here's the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech and provide the script, gestures, and audio.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Observe the gestures: {MOTION_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech, including the script, gestures, and audio recording.",
     'output': f"{SCRIPT_HOLDER} is the speech text. The gestures are: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and provide the script, gestures, and audio?",
     'output': f"The script is: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. And here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech with the script, gestures, and audio included.",
     'output': f"{SCRIPT_HOLDER} is the generated script. Watch the gestures: {MOTION_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech, including the full script, gestures, and audio.",
     'output': f"The speech script: {SCRIPT_HOLDER}. Watch these gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech randomly, complete with the script, gestures, and audio version.",
     'output': f"The script is: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. Audio version: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech with the script, gestures, and audio file.",
     'output': f"Speech: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. Audio file: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech and include the script, gestures, and audio file?",
     'output': f"The script is: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. And the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech, including its script, gestures, and audio.",
     'output': f"{SCRIPT_HOLDER} is the speech text. Watch the gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech and provide the script, gestures, and audio recording.",
     'output': f"The speech: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech with the script, gestures, and audio included.",
     'output': f"Here’s the script: {SCRIPT_HOLDER}. Watch these gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech, including the full script, gestures, and audio.",
     'output': f"Speech script: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. And here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and provide the script, gestures, and audio?",
     'output': f"The script: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech at random, with the script, gestures, and audio recording.",
     'output': f"{SCRIPT_HOLDER} is the script. The gestures are: {MOTION_HOLDER}. The audio is: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech and include the script, gestures, and audio.",
     'output': f"The script is: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech randomly, complete with the script, gestures, and audio.",
     'output': f"Speech: {SCRIPT_HOLDER}. Watch these gestures: {MOTION_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly and provide the script, gestures, and audio file.",
     'output': f"The script is: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. The audio file: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a random speech, including the script, gestures, and audio?",
     'output': f"The script: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. And the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech with its script, gestures, and audio included.",
     'output': f"{SCRIPT_HOLDER} is the speech text. The gestures: {MOTION_HOLDER}. Listen to this audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech and provide the full script, gestures, and audio.",
     'output': f"The speech is: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech at random and include the script, gestures, and audio.",
     'output': f"Here’s the script: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech, including the script, gestures, and audio recording.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Watch the gestures: {MOTION_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a random speech and provide the script, gestures, and audio?",
     'output': f"The script: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech randomly with the script, gestures, and audio.",
     'output': f"{SCRIPT_HOLDER} is the text. Watch the gestures: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech, including the script, gestures, and audio version.",
     'output': f"The generated script: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech and include the script, gestures, and audio recording.",
     'output': f"Speech: {SCRIPT_HOLDER}. The gestures: {MOTION_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech with the script, gestures, and audio included.",
     'output': f"The script is: {SCRIPT_HOLDER}. The gestures are: {MOTION_HOLDER}. The audio is: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech with its script, gestures, and audio?",
     'output': f"The script: {SCRIPT_HOLDER}. Gestures: {MOTION_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech, complete with the script, gestures, and audio.",
     'output': f"{SCRIPT_HOLDER} is the speech text. The gestures: {MOTION_HOLDER}. Listen to this: {AUDIO_HOLDER}"},
]

PRETRAIN_N2ASG_TEMPLATE=[
    {'output': f"{AUDIO_HOLDER}{SCRIPT_HOLDER}{MOTION_HOLDER}"},
    {'output': f"{SCRIPT_HOLDER}{AUDIO_HOLDER}{MOTION_HOLDER}"},
    {'output': f"{MOTION_HOLDER}{SCRIPT_HOLDER}{AUDIO_HOLDER}"},
    {'output': f"{MOTION_HOLDER}{AUDIO_HOLDER}{SCRIPT_HOLDER}"},
    {'output': f"{SCRIPT_HOLDER}{MOTION_HOLDER}{AUDIO_HOLDER}"},
    {'output': f"{AUDIO_HOLDER}{MOTION_HOLDER}{SCRIPT_HOLDER}"},
]