# random generate audio and script pairs
from mmotion.utils.task.modality import AUDIO_HOLDER,  SCRIPT_HOLDER

N2AS_TEMPLATE = [
    {'input': f"Randomly generate a speech and read it aloud.",
     'output': f"{SCRIPT_HOLDER}, {AUDIO_HOLDER}"},
    {'input': f"Generate a random speech and present it with audio.",
     'output': f"Here is the speech: {SCRIPT_HOLDER}. Now, listen to it: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and read it out loud.",
     'output': f"Speech text: {SCRIPT_HOLDER}. Here is the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and then recite it?",
     'output': f"The speech is: {SCRIPT_HOLDER}, and here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech at random and read it aloud.",
     'output': f"Generated speech: {SCRIPT_HOLDER}. Listen to the audio here: {AUDIO_HOLDER}"},

    {'input': f"Please create a random speech and provide an audio reading.",
     'output': f"Here’s the speech: {SCRIPT_HOLDER}. And the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and give me an audio version.",
     'output': f"Speech content: {SCRIPT_HOLDER}. Audio version: {AUDIO_HOLDER}"},

    {'input': f"Can you create a random speech and read it for me?",
     'output': f"The speech goes like this: {SCRIPT_HOLDER}. Here's the spoken audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and provide an audio narration.",
     'output': f"{SCRIPT_HOLDER} is the script. Listen to the narration: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech and then read it aloud.",
     'output': f"The generated script: {SCRIPT_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech at random and provide an audio reading.",
     'output': f"Here’s what the speech says: {SCRIPT_HOLDER}. And here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and provide an audio file of it being read.",
     'output': f"Speech: {SCRIPT_HOLDER}. Here is the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech randomly and read it out loud?",
     'output': f"The speech is as follows: {SCRIPT_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and give me the audio recording.",
     'output': f"The speech text: {SCRIPT_HOLDER}. Audio recording: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and provide its audio version.",
     'output': f"{SCRIPT_HOLDER} is the script. Here’s the audio version: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and present an audio rendition.",
     'output': f"Speech content: {SCRIPT_HOLDER}. Audio rendition: {AUDIO_HOLDER}"},

    {'input': f"Can you create a speech randomly and read it to me?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Listen to it here: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech and provide an audio narration.",
     'output': f"The generated speech: {SCRIPT_HOLDER}. Audio narration: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech and provide the audio version.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Audio version: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and read it aloud for me.",
     'output': f"The script is: {SCRIPT_HOLDER}. Here's the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and provide a recording of it.",
     'output': f"Script: {SCRIPT_HOLDER}. Audio recording: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech and read it out loud?",
     'output': f"Speech: {SCRIPT_HOLDER}. Here’s how it sounds: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech randomly and give me the audio.",
     'output': f"{SCRIPT_HOLDER} is the generated speech. Audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech and then read it out loud.",
     'output': f"Here's the speech: {SCRIPT_HOLDER}. And the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and read it to me.",
     'output': f"The generated speech is: {SCRIPT_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Can you create a random speech and provide an audio file?",
     'output': f"Generated speech text: {SCRIPT_HOLDER}. The audio file: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech at random and read it aloud.",
     'output': f"The speech text is: {SCRIPT_HOLDER}. Here's the audio: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech and give me an audio recording.",
     'output': f"Speech content: {SCRIPT_HOLDER}. Audio recording: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and provide the audio.",
     'output': f"{SCRIPT_HOLDER} is the speech. Audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and read it out for me.",
     'output': f"Here's the speech script: {SCRIPT_HOLDER}. Listen to it: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech randomly and provide an audio version?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Audio version: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech and give me the audio version.",
     'output': f"{SCRIPT_HOLDER} is the generated speech. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and provide an audio rendition.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Audio rendition: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly and read it out loud.",
     'output': f"Here’s the speech: {SCRIPT_HOLDER}. And the audio reading: {AUDIO_HOLDER}"},

    {'input': f"Can you produce a random speech and then read it aloud?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Listen to it here: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and provide the audio recording.",
     'output': f"Speech content: {SCRIPT_HOLDER}. Audio recording: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and present it as an audio file.",
     'output': f"{SCRIPT_HOLDER} is the speech. Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and give me the audio?",
     'output': f"The speech goes like this: {SCRIPT_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and read it out loud.",
     'output': f"{SCRIPT_HOLDER} is the generated text. Listen to it: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech at random and provide the audio version.",
     'output': f"Here’s what the speech says: {SCRIPT_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and provide an audio narration.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Narration audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech and read it for me?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Here’s how it sounds: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and provide the audio file.",
     'output': f"The speech content: {SCRIPT_HOLDER}. Audio file: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and read it out loud for me.",
     'output': f"The script is: {SCRIPT_HOLDER}. Here's the audio rendition: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and provide an audio reading.",
     'output': f"{SCRIPT_HOLDER} is the script. Audio reading: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech and provide an audio version?",
     'output': f"Speech text: {SCRIPT_HOLDER}. Audio version: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech randomly and present the audio.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Listen to it here: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech and give me the audio file.",
     'output': f"{SCRIPT_HOLDER} is the speech. The audio file: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and provide an audio clip.",
     'output': f"Here’s the script: {SCRIPT_HOLDER}. Audio clip: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and read it to me?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Audio rendition: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and give me an audio narration.",
     'output': f"Speech content: {SCRIPT_HOLDER}. Narration: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech randomly and provide the audio recording.",
     'output': f"{SCRIPT_HOLDER} is the generated text. Audio recording: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and read it aloud for me.",
     'output': f"The script is: {SCRIPT_HOLDER}. Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech and present it with an audio file?",
     'output': f"The speech text: {SCRIPT_HOLDER}. Audio file: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and provide the audio rendition.",
     'output': f"{SCRIPT_HOLDER} is the speech. Audio rendition: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech and read it out for me.",
     'output': f"Here’s the speech: {SCRIPT_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and present it in an audio format.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Audio format: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and provide the audio?",
     'output': f"The speech goes like this: {SCRIPT_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech and give me an audio reading.",
     'output': f"{SCRIPT_HOLDER} is the generated text. Here’s the reading: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and provide the audio version.",
     'output': f"Speech text: {SCRIPT_HOLDER}. Listen to it: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly and read it out loud.",
     'output': f"The generated speech: {SCRIPT_HOLDER}. Audio version: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech randomly and present it in audio?",
     'output': f"The speech text is: {SCRIPT_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and provide the audio rendition.",
     'output': f"{SCRIPT_HOLDER} is the speech script. The audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech at random and provide an audio clip.",
     'output': f"Here’s what the speech says: {SCRIPT_HOLDER}. Audio clip: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and read it for me.",
     'output': f"Speech: {SCRIPT_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Can you produce a speech randomly and provide an audio version?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Audio version: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech and provide the audio narration.",
     'output': f"Speech content: {SCRIPT_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and present it in audio.",
     'output': f"{SCRIPT_HOLDER} is the script. Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and give me an audio reading?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Audio reading: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech and provide the audio version.",
     'output': f"{SCRIPT_HOLDER} is the generated speech. The audio version: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and provide the audio file.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Audio file: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly and read it out loud.",
     'output': f"The speech is: {SCRIPT_HOLDER}. Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a random speech and present it as audio?",
     'output': f"The script is: {SCRIPT_HOLDER}. Audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and read it out for me.",
     'output': f"{SCRIPT_HOLDER} is the speech. Listen to it: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and provide an audio rendition.",
     'output': f"Speech text: {SCRIPT_HOLDER}. Audio rendition: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and provide an audio recording.",
     'output': f"The generated script: {SCRIPT_HOLDER}. Audio recording: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech and provide the audio version?",
     'output': f"The speech content: {SCRIPT_HOLDER}. The audio version: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech and read it out loud.",
     'output': f"{SCRIPT_HOLDER} is the text. Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and present it with an audio clip.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Audio clip: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly and provide the audio file.",
     'output': f"Here’s the speech: {SCRIPT_HOLDER}. The audio file: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a speech and give me an audio narration?",
     'output': f"The speech is: {SCRIPT_HOLDER}. Narration: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech and provide the audio version.",
     'output': f"{SCRIPT_HOLDER} is the generated speech. Audio version: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech at random and read it aloud.",
     'output': f"Here’s what the speech says: {SCRIPT_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech and present it in audio.",
     'output': f"Speech script: {SCRIPT_HOLDER}. Audio rendition: {AUDIO_HOLDER}"},

    {'input': f"Can you randomly generate a speech and read it out loud?",
     'output': f"The speech text: {SCRIPT_HOLDER}. The audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech and provide an audio file.",
     'output': f"{SCRIPT_HOLDER} is the text. Audio file: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech and provide an audio version.",
     'output': f"Speech: {SCRIPT_HOLDER}. Listen to the audio: {AUDIO_HOLDER}"},

    {'input': f"Create a speech randomly and read it out.",
     'output': f"The script is: {SCRIPT_HOLDER}. Audio reading: {AUDIO_HOLDER}"},

]

PRETRAIN_N2AS_TEMPLATE = [
    {'output': f'{SCRIPT_HOLDER}{AUDIO_HOLDER}'},
    {'output': f"{AUDIO_HOLDER}{SCRIPT_HOLDER}"},
    {'output': f"{AUDIO_HOLDER}, {SCRIPT_HOLDER}"},
    {'output': f"{SCRIPT_HOLDER}, {AUDIO_HOLDER}"}
]
