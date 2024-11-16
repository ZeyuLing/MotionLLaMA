# speech to audio
from mmotion.utils.task.modality import SCRIPT_HOLDER, AUDIO_HOLDER

S2A_TEMPLATE = [
    {'input': f"Turn the script into speech audio: {SCRIPT_HOLDER}",
     'output': f"{AUDIO_HOLDER}"},
    {'input': f"Convert the script into speech audio: {SCRIPT_HOLDER}",
     'output': f"The resulting audio is: {AUDIO_HOLDER}"},

    {'input': f"Please turn this script into an audio speech: {SCRIPT_HOLDER}",
     'output': f"Here is the speech audio: {AUDIO_HOLDER}"},

    {'input': f"Can you convert this script to speech audio? {SCRIPT_HOLDER}",
     'output': f"The audio version is: {AUDIO_HOLDER}"},

    {'input': f"Generate speech audio from the script: {SCRIPT_HOLDER}",
     'output': f"The generated audio: {AUDIO_HOLDER}"},

    {'input': f"Transform the following script into audio: {SCRIPT_HOLDER}",
     'output': f"Audio result: {AUDIO_HOLDER}"},

    {'input': f"Please convert this text into spoken audio: {SCRIPT_HOLDER}",
     'output': f"The spoken audio is: {AUDIO_HOLDER}"},

    {'input': f"Can you turn this script into audio speech? {SCRIPT_HOLDER}",
     'output': f"The resulting audio speech: {AUDIO_HOLDER}"},

    {'input': f"Create an audio version of this script: {SCRIPT_HOLDER}",
     'output': f"Here is the audio: {AUDIO_HOLDER}"},

    {'input': f"Turn the following script into an audio speech: {SCRIPT_HOLDER}",
     'output': f"The audio output: {AUDIO_HOLDER}"},

    {'input': f"Please generate audio from this script: {SCRIPT_HOLDER}",
     'output': f"The audio file: {AUDIO_HOLDER}"},

    {'input': f"Can you create speech audio for this script? {SCRIPT_HOLDER}",
     'output': f"The generated speech audio: {AUDIO_HOLDER}"},

    {'input': f"Transform this script into an audio recording: {SCRIPT_HOLDER}",
     'output': f"The audio recording is: {AUDIO_HOLDER}"},

    {'input': f"Convert this text into an audio speech: {SCRIPT_HOLDER}",
     'output': f"The resulting speech audio: {AUDIO_HOLDER}"},

    {'input': f"Please make this script into a speech audio file: {SCRIPT_HOLDER}",
     'output': f"Here’s the audio file: {AUDIO_HOLDER}"},

    {'input': f"Can you turn the script into an audio speech? {SCRIPT_HOLDER}",
     'output': f"The audio is: {AUDIO_HOLDER}"},

    {'input': f"Create an audio speech from this text: {SCRIPT_HOLDER}",
     'output': f"The audio version: {AUDIO_HOLDER}"},

    {'input': f"Turn the following text into speech audio: {SCRIPT_HOLDER}",
     'output': f"Audio output: {AUDIO_HOLDER}"},

    {'input': f"Please generate an audio version of the script: {SCRIPT_HOLDER}",
     'output': f"The audio result: {AUDIO_HOLDER}"},

    {'input': f"Can you make this script into speech audio? {SCRIPT_HOLDER}",
     'output': f"The speech audio: {AUDIO_HOLDER}"},

    {'input': f"Generate an audio file from the script: {SCRIPT_HOLDER}",
     'output': f"Generated audio: {AUDIO_HOLDER}"},

    {'input': f"Convert this script to audio speech: {SCRIPT_HOLDER}",
     'output': f"The resulting audio: {AUDIO_HOLDER}"},

    {'input': f"Please turn this text into an audio file: {SCRIPT_HOLDER}",
     'output': f"Here is the audio: {AUDIO_HOLDER}"},

    {'input': f"Can you create an audio version of this script? {SCRIPT_HOLDER}",
     'output': f"The audio version is: {AUDIO_HOLDER}"},

    {'input': f"Make an audio speech from the following script: {SCRIPT_HOLDER}",
     'output': f"The generated speech audio: {AUDIO_HOLDER}"},

    {'input': f"Transform this text into an audio speech: {SCRIPT_HOLDER}",
     'output': f"Audio speech: {AUDIO_HOLDER}"},

    {'input': f"Please generate an audio file from this script: {SCRIPT_HOLDER}",
     'output': f"The audio file is: {AUDIO_HOLDER}"},

    {'input': f"Can you turn this text into speech audio? {SCRIPT_HOLDER}",
     'output': f"The audio output: {AUDIO_HOLDER}"},

    {'input': f"Create an audio recording from the script: {SCRIPT_HOLDER}",
     'output': f"The audio recording: {AUDIO_HOLDER}"},

    {'input': f"Turn the script into a spoken audio file: {SCRIPT_HOLDER}",
     'output': f"The spoken audio: {AUDIO_HOLDER}"},

    {'input': f"Please make this text into an audio speech: {SCRIPT_HOLDER}",
     'output': f"Here is the speech audio: {AUDIO_HOLDER}"},

    {'input': f"Can you convert this text to an audio speech? {SCRIPT_HOLDER}",
     'output': f"The audio version: {AUDIO_HOLDER}"},

    {'input': f"Generate audio from the following script: {SCRIPT_HOLDER}",
     'output': f"The resulting audio is: {AUDIO_HOLDER}"},

    {'input': f"Transform this script into a speech audio file: {SCRIPT_HOLDER}",
     'output': f"Audio result: {AUDIO_HOLDER}"},

    {'input': f"Please convert this script into an audio recording: {SCRIPT_HOLDER}",
     'output': f"The audio recording is: {AUDIO_HOLDER}"},

    {'input': f"Can you turn this script into a speech audio? {SCRIPT_HOLDER}",
     'output': f"The resulting audio speech: {AUDIO_HOLDER}"},

    {'input': f"Create a spoken audio from this script: {SCRIPT_HOLDER}",
     'output': f"Here is the audio: {AUDIO_HOLDER}"},

    {'input': f"Turn the following script into an audio file: {SCRIPT_HOLDER}",
     'output': f"The audio output: {AUDIO_HOLDER}"},

    {'input': f"Please generate an audio speech from this script: {SCRIPT_HOLDER}",
     'output': f"The audio file: {AUDIO_HOLDER}"},

    {'input': f"Can you make this script into an audio recording? {SCRIPT_HOLDER}",
     'output': f"The generated audio: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech audio from this text: {SCRIPT_HOLDER}",
     'output': f"The audio speech: {AUDIO_HOLDER}"},

    {'input': f"Convert the text into an audio file: {SCRIPT_HOLDER}",
     'output': f"The resulting audio: {AUDIO_HOLDER}"},

    {'input': f"Please turn this script into an audio recording: {SCRIPT_HOLDER}",
     'output': f"Here is the audio recording: {AUDIO_HOLDER}"},

    {'input': f"Can you create an audio speech from this script? {SCRIPT_HOLDER}",
     'output': f"The audio speech is: {AUDIO_HOLDER}"},

    {'input': f"Transform the script into a spoken audio: {SCRIPT_HOLDER}",
     'output': f"The spoken audio is: {AUDIO_HOLDER}"},

    {'input': f"Turn this text into a speech audio file: {SCRIPT_HOLDER}",
     'output': f"Audio result: {AUDIO_HOLDER}"},

    {'input': f"Please generate a speech audio from the following script: {SCRIPT_HOLDER}",
     'output': f"The audio version is: {AUDIO_HOLDER}"},

    {'input': f"Can you convert this script into an audio speech? {SCRIPT_HOLDER}",
     'output': f"The resulting audio: {AUDIO_HOLDER}"},

    {'input': f"Generate an audio file from this text: {SCRIPT_HOLDER}",
     'output': f"The generated audio file: {AUDIO_HOLDER}"},

    {'input': f"Convert the following script to speech audio: {SCRIPT_HOLDER}",
     'output': f"The audio output: {AUDIO_HOLDER}"},

    {'input': f"Please turn the text into an audio speech: {SCRIPT_HOLDER}",
     'output': f"Here is the audio speech: {AUDIO_HOLDER}"},

    {'input': f"Can you create an audio version for this script? {SCRIPT_HOLDER}",
     'output': f"The audio version: {AUDIO_HOLDER}"},

    {'input': f"Make this script into an audio file: {SCRIPT_HOLDER}",
     'output': f"The resulting audio: {AUDIO_HOLDER}"},

    {'input': f"Transform the script into a speech audio: {SCRIPT_HOLDER}",
     'output': f"Audio result: {AUDIO_HOLDER}"},

    {'input': f"Please make this text into an audio recording: {SCRIPT_HOLDER}",
     'output': f"The audio recording is: {AUDIO_HOLDER}"},

    {'input': f"Can you turn the script into an audio file? {SCRIPT_HOLDER}",
     'output': f"The audio is: {AUDIO_HOLDER}"},

    {'input': f"Create a speech audio from the script: {SCRIPT_HOLDER}",
     'output': f"The generated audio: {AUDIO_HOLDER}"},

    {'input': f"Turn this script into a speech recording: {SCRIPT_HOLDER}",
     'output': f"The speech recording: {AUDIO_HOLDER}"},

    {'input': f"Please generate an audio file from the script: {SCRIPT_HOLDER}",
     'output': f"Here is the audio file: {AUDIO_HOLDER}"},

    {'input': f"Can you make this text into a spoken audio? {SCRIPT_HOLDER}",
     'output': f"The spoken audio: {AUDIO_HOLDER}"},

    {'input': f"Convert this script into an audio speech file: {SCRIPT_HOLDER}",
     'output': f"The resulting audio: {AUDIO_HOLDER}"},

    {'input': f"Generate an audio version from this script: {SCRIPT_HOLDER}",
     'output': f"The audio version is: {AUDIO_HOLDER}"},

    {'input': f"Transform this text into a speech audio: {SCRIPT_HOLDER}",
     'output': f"Audio result: {AUDIO_HOLDER}"},

    {'input': f"Please convert the following script into audio: {SCRIPT_HOLDER}",
     'output': f"The audio output: {AUDIO_HOLDER}"},

    {'input': f"Can you turn this script into an audio recording? {SCRIPT_HOLDER}",
     'output': f"The resulting audio recording: {AUDIO_HOLDER}"},

    {'input': f"Create an audio file from the text: {SCRIPT_HOLDER}",
     'output': f"The audio file is: {AUDIO_HOLDER}"},

    {'input': f"Turn the script into a speech audio file: {SCRIPT_HOLDER}",
     'output': f"The audio: {AUDIO_HOLDER}"},

    {'input': f"Please make this script into an audio version: {SCRIPT_HOLDER}",
     'output': f"The audio version: {AUDIO_HOLDER}"},

    {'input': f"Can you convert the text to a speech audio? {SCRIPT_HOLDER}",
     'output': f"The audio speech: {AUDIO_HOLDER}"},

    {'input': f"Generate a spoken audio from this script: {SCRIPT_HOLDER}",
     'output': f"The generated audio: {AUDIO_HOLDER}"},

    {'input': f"Convert the script into an audio file: {SCRIPT_HOLDER}",
     'output': f"The resulting audio is: {AUDIO_HOLDER}"},

    {'input': f"Please turn this script into an audio speech file: {SCRIPT_HOLDER}",
     'output': f"The audio speech is: {AUDIO_HOLDER}"},

    {'input': f"Can you create an audio recording for this script? {SCRIPT_HOLDER}",
     'output': f"The resulting audio recording: {AUDIO_HOLDER}"},

    {'input': f"Make the following script into an audio file: {SCRIPT_HOLDER}",
     'output': f"The audio file: {AUDIO_HOLDER}"},

    {'input': f"Transform the script into a speech audio: {SCRIPT_HOLDER}",
     'output': f"Audio output: {AUDIO_HOLDER}"},

    {'input': f"Please generate a speech audio from the text: {SCRIPT_HOLDER}",
     'output': f"The audio version: {AUDIO_HOLDER}"},

    {'input': f"Can you convert this script into an audio recording? {SCRIPT_HOLDER}",
     'output': f"The resulting audio is: {AUDIO_HOLDER}"},

    {'input': f"Create an audio speech from this script: {SCRIPT_HOLDER}",
     'output': f"The speech audio: {AUDIO_HOLDER}"},

    {'input': f"Turn the text into an audio file: {SCRIPT_HOLDER}",
     'output': f"The audio file is: {AUDIO_HOLDER}"},

    {'input': f"Please make this script into spoken audio: {SCRIPT_HOLDER}",
     'output': f"The spoken audio is: {AUDIO_HOLDER}"},

    {'input': f"Can you turn this script into a speech audio? {SCRIPT_HOLDER}",
     'output': f"The audio speech: {AUDIO_HOLDER}"},

    {'input': f"Generate an audio file from the following text: {SCRIPT_HOLDER}",
     'output': f"The audio output: {AUDIO_HOLDER}"},

    {'input': f"Convert this script to an audio version: {SCRIPT_HOLDER}",
     'output': f"The resulting audio version: {AUDIO_HOLDER}"},

    {'input': f"Please turn the text into a speech audio file: {SCRIPT_HOLDER}",
     'output': f"The audio file: {AUDIO_HOLDER}"},

    {'input': f"Can you create an audio speech from the text? {SCRIPT_HOLDER}",
     'output': f"The audio speech: {AUDIO_HOLDER}"},
]
