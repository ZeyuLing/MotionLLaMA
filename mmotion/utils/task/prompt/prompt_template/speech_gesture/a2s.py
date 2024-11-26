# audio to speech script
from mmotion.utils.task.modality import AUDIO_HOLDER,  SCRIPT_HOLDER

A2S_TEMPLATE = [
    {'input': f"Listen to the speech and write down the content: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},
    {'input': f"Listen to this audio and transcribe the speech content: {AUDIO_HOLDER}",
     'output': f"The transcribed speech is: {SCRIPT_HOLDER}"},

    {'input': f"Can you listen to the following audio and write down what is being said? {AUDIO_HOLDER}",
     'output': f"Here's the written content: {SCRIPT_HOLDER}"},

    {'input': f"Please transcribe the content of this speech audio: {AUDIO_HOLDER}",
     'output': f"The speech script is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to the audio and provide the text of the speech: {AUDIO_HOLDER}",
     'output': f"The text content is: {SCRIPT_HOLDER}"},

    {'input': f"Transcribe the speech from this audio: {AUDIO_HOLDER}",
     'output': f"The transcription is: {SCRIPT_HOLDER}"},

    {'input': f"Can you write down what’s being said in this audio clip? {AUDIO_HOLDER}",
     'output': f"The speech content is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this speech and write down the content: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Please listen to the audio and transcribe the speech: {AUDIO_HOLDER}",
     'output': f"Transcribed content: {SCRIPT_HOLDER}"},

    {'input': f"Can you transcribe the speech from the following audio? {AUDIO_HOLDER}",
     'output': f"The script is: {SCRIPT_HOLDER}"},

    {'input': f"Write down the speech from this audio clip: {AUDIO_HOLDER}",
     'output': f"The speech text: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this audio and type out the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Can you listen to the following audio and provide the speech in text form? {AUDIO_HOLDER}",
     'output': f"The written speech is: {SCRIPT_HOLDER}"},

    {'input': f"Please transcribe what is being said in this audio: {AUDIO_HOLDER}",
     'output': f"The content is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to the audio and give me the text of the speech: {AUDIO_HOLDER}",
     'output': f"The speech content: {SCRIPT_HOLDER}"},

    {'input': f"Transcribe the following audio speech: {AUDIO_HOLDER}",
     'output': f"The transcript is: {SCRIPT_HOLDER}"},

    {'input': f"Can you write down the speech from this audio clip? {AUDIO_HOLDER}",
     'output': f"The transcription: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this and transcribe the speech content: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Please listen to the audio and provide a written version of the speech: {AUDIO_HOLDER}",
     'output': f"Written speech: {SCRIPT_HOLDER}"},

    {'input': f"Can you transcribe the content of this speech audio? {AUDIO_HOLDER}",
     'output': f"The text is: {SCRIPT_HOLDER}"},

    {'input': f"Write down what’s being said in the audio: {AUDIO_HOLDER}",
     'output': f"The speech is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to the audio and jot down the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Can you listen to this audio and provide the transcription of the speech? {AUDIO_HOLDER}",
     'output': f"Transcription: {SCRIPT_HOLDER}"},

    {'input': f"Please transcribe the speech from this audio clip: {AUDIO_HOLDER}",
     'output': f"The speech script is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this and write down the content of the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe what is being said in this audio clip: {AUDIO_HOLDER}",
     'output': f"The written content: {SCRIPT_HOLDER}"},

    {'input': f"Can you write down the speech content from this audio? {AUDIO_HOLDER}",
     'output': f"The script: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this audio and provide a text version of the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Please listen to the following audio and write down the speech: {AUDIO_HOLDER}",
     'output': f"The speech text is: {SCRIPT_HOLDER}"},

    {'input': f"Can you transcribe the following speech audio? {AUDIO_HOLDER}",
     'output': f"The transcript: {SCRIPT_HOLDER}"},

    {'input': f"Write down the content of the speech from this audio: {AUDIO_HOLDER}",
     'output': f"The transcribed speech: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this audio clip and transcribe the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Can you listen to the audio and provide the speech script? {AUDIO_HOLDER}",
     'output': f"The transcription is: {SCRIPT_HOLDER}"},

    {'input': f"Please transcribe the following audio speech: {AUDIO_HOLDER}",
     'output': f"The speech content is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this and write down what is being said: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe the content from this audio clip: {AUDIO_HOLDER}",
     'output': f"The written speech: {SCRIPT_HOLDER}"},

    {'input': f"Can you write down the content of the speech in this audio? {AUDIO_HOLDER}",
     'output': f"The script is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this audio and provide a written transcript: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Please listen to the audio and transcribe what is being said: {AUDIO_HOLDER}",
     'output': f"Transcribed content: {SCRIPT_HOLDER}"},

    {'input': f"Can you transcribe the speech from this audio clip? {AUDIO_HOLDER}",
     'output': f"The speech is: {SCRIPT_HOLDER}"},

    {'input': f"Write down what’s being said in the following audio: {AUDIO_HOLDER}",
     'output': f"The transcription: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this and provide the speech in text form: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe the speech from this audio file: {AUDIO_HOLDER}",
     'output': f"The written content is: {SCRIPT_HOLDER}"},

    {'input': f"Can you write down the speech from the audio? {AUDIO_HOLDER}",
     'output': f"The script: {SCRIPT_HOLDER}"},

    {'input': f"Listen to the audio and transcribe the text: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Please transcribe the speech content from this audio: {AUDIO_HOLDER}",
     'output': f"The speech content: {SCRIPT_HOLDER}"},

    {'input': f"Can you transcribe the following audio to text? {AUDIO_HOLDER}",
     'output': f"The text is: {SCRIPT_HOLDER}"},

    {'input': f"Write down the text of the speech from this audio: {AUDIO_HOLDER}",
     'output': f"The transcribed speech is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this audio and provide the speech in text: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Please listen to this audio and transcribe the speech: {AUDIO_HOLDER}",
     'output': f"The transcription is: {SCRIPT_HOLDER}"},

    {'input': f"Can you transcribe what is being said in this audio? {AUDIO_HOLDER}",
     'output': f"The speech text: {SCRIPT_HOLDER}"},

    {'input': f"Write down the content of this speech from the audio: {AUDIO_HOLDER}",
     'output': f"The transcription: {SCRIPT_HOLDER}"},

    {'input': f"Listen to the audio and write down the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe this speech from the audio clip: {AUDIO_HOLDER}",
     'output': f"The speech script is: {SCRIPT_HOLDER}"},

    {'input': f"Can you listen to this audio and transcribe the speech content? {AUDIO_HOLDER}",
     'output': f"The written content is: {SCRIPT_HOLDER}"},

    {'input': f"Please transcribe the speech in this audio: {AUDIO_HOLDER}",
     'output': f"The script: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this audio clip and write down the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe the speech from the following audio: {AUDIO_HOLDER}",
     'output': f"The transcribed text: {SCRIPT_HOLDER}"},

    {'input': f"Can you write down the speech from this audio file? {AUDIO_HOLDER}",
     'output': f"The text is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to the audio and provide a written transcript of the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Please listen to the audio and transcribe it into text: {AUDIO_HOLDER}",
     'output': f"The transcribed speech: {SCRIPT_HOLDER}"},

    {'input': f"Can you transcribe the audio content to text? {AUDIO_HOLDER}",
     'output': f"The script is: {SCRIPT_HOLDER}"},

    {'input': f"Write down the text from this speech audio: {AUDIO_HOLDER}",
     'output': f"The transcription: {SCRIPT_HOLDER}"},

    {'input': f"Listen to the audio and provide the speech in written form: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe the speech audio into text: {AUDIO_HOLDER}",
     'output': f"The speech script: {SCRIPT_HOLDER}"},

    {'input': f"Can you listen to the following audio and transcribe the content? {AUDIO_HOLDER}",
     'output': f"The transcribed content: {SCRIPT_HOLDER}"},

    {'input': f"Please transcribe the audio into speech text: {AUDIO_HOLDER}",
     'output': f"The speech text: {SCRIPT_HOLDER}"},

    {'input': f"Listen to the audio clip and write down what is being said: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe the content of the speech in this audio: {AUDIO_HOLDER}",
     'output': f"The written speech: {SCRIPT_HOLDER}"},

    {'input': f"Can you write down the speech from this audio? {AUDIO_HOLDER}",
     'output': f"The transcript is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this audio and provide the speech as text: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Please transcribe what is being said in this audio to text: {AUDIO_HOLDER}",
     'output': f"The speech content: {SCRIPT_HOLDER}"},

    {'input': f"Can you transcribe the audio to speech text? {AUDIO_HOLDER}",
     'output': f"The text is: {SCRIPT_HOLDER}"},

    {'input': f"Write down what’s being said in this speech audio: {AUDIO_HOLDER}",
     'output': f"The transcribed speech: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this and provide the speech content in text: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe the following audio into a written speech: {AUDIO_HOLDER}",
     'output': f"The speech script: {SCRIPT_HOLDER}"},

    {'input': f"Can you listen to the audio and write down the speech? {AUDIO_HOLDER}",
     'output': f"The transcription is: {SCRIPT_HOLDER}"},

    {'input': f"Please transcribe the speech audio into text: {AUDIO_HOLDER}",
     'output': f"The content is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this and write down the speech content: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"},

    {'input': f"Transcribe the audio speech into text: {AUDIO_HOLDER}",
     'output': f"The written content is: {SCRIPT_HOLDER}"},

    {'input': f"Can you write down the speech text from the audio? {AUDIO_HOLDER}",
     'output': f"The script is: {SCRIPT_HOLDER}"},

    {'input': f"Listen to this audio and provide a written version of the speech: {AUDIO_HOLDER}",
     'output': f"{SCRIPT_HOLDER}"}
]
