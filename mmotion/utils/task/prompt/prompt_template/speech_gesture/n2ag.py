from mmotion.utils.task.modality import AUDIO_HOLDER, MOTION_HOLDER

N2AG_TEMPLATE = [
    {'input': f'Generate a random speech, and it should be accompanied by gestures.',
     'output': f"Okay, here's speech clip: {AUDIO_HOLDER} {MOTION_HOLDER}"},
    {'input': f"I need a speech with both audio and gestures.",
     'output': f"The audio is as follows: {AUDIO_HOLDER},"
               f" possibly accompanied by these kinds of body movements: {MOTION_HOLDER}."},
    {'input': f"Create a random speech along with some gestures.",
     'output': f"Sure, here is the speech: {AUDIO_HOLDER} and the corresponding gestures: {MOTION_HOLDER}."},

    {'input': f"I would like a speech that includes both audio and gestures.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, accompanied by these movements: {MOTION_HOLDER}."},

    {'input': f"Generate an audio clip with synchronized gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER} and the accompanying gestures: {MOTION_HOLDER}."},

    {'input': f"I want a speech with gestures to go with the audio.",
     'output': f"The audio sounds like this: {AUDIO_HOLDER}, and the body movements are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech along with matching gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER} and the gestures: {MOTION_HOLDER}."},

    {'input': f"Please generate a speech clip that comes with some gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER} and the corresponding body motions: {MOTION_HOLDER}."},

    {'input': f"Create a speech with some gestures for emphasis.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the accompanying gestures are: {MOTION_HOLDER}."},

    {'input': f"Generate a speech that includes hand and body gestures.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the gestures that go with it: {MOTION_HOLDER}."},

    {'input': f"I need a random speech along with gestures.",
     'output': f"The speech sounds like this: {AUDIO_HOLDER}, and the movements are: {MOTION_HOLDER}."},

    {'input': f"Produce a random speech clip that includes gestures.",
     'output': f"Here’s the audio clip: {AUDIO_HOLDER}, accompanied by these gestures: {MOTION_HOLDER}."},

    {'input': f"Generate a speech with gestures as part of the presentation.",
     'output': f"Here is the speech: {AUDIO_HOLDER}, accompanied by these gestures: {MOTION_HOLDER}."},

    {'input': f"Can you create a speech with gestures?",
     'output': f"Sure, the audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Generate an audio clip with synchronized body movements.",
     'output': f"Here is the speech: {AUDIO_HOLDER}, accompanied by the following gestures: {MOTION_HOLDER}."},

    {'input': f"Please provide a speech with accompanying gestures.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the body gestures: {MOTION_HOLDER}."},

    {'input': f"Create a speech that includes both the audio and gestures.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the movements are: {MOTION_HOLDER}."},

    {'input': f"I’d like a speech with gestures included.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, with the following gestures: {MOTION_HOLDER}."},

    {'input': f"Generate a random speech along with corresponding gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"I need a speech that comes with both audio and gestures.",
     'output': f"The speech sounds like this: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech and include gestures with it.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, accompanied by these movements: {MOTION_HOLDER}."},

    {'input': f"Generate a speech that includes both sound and gestures.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the accompanying gestures are: {MOTION_HOLDER}."},

    {'input': f"Create a random speech with gestures added.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, along with the gestures: {MOTION_HOLDER}."},

    {'input': f"Generate a speech with gestures for emphasis.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the corresponding gestures are: {MOTION_HOLDER}."},

    {'input': f"Please provide a speech and gestures together.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the gestures: {MOTION_HOLDER}."},

    {'input': f"I need a speech clip with some gestures to match.",
     'output': f"Here is the speech: {AUDIO_HOLDER}, accompanied by gestures: {MOTION_HOLDER}."},

    {'input': f"Create a speech that includes both sound and body language.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Generate a speech along with the necessary gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, along with these body movements: {MOTION_HOLDER}."},

    {'input': f"I would like a speech to be accompanied by gestures.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the matching gestures: {MOTION_HOLDER}."},

    {'input': f"Produce a speech with hand and body gestures.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Create a random speech with some accompanying gestures.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the corresponding movements are: {MOTION_HOLDER}."},

    {'input': f"Generate a speech clip with synchronized gestures.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, accompanied by these gestures: {MOTION_HOLDER}."},

    {'input': f"I want a speech with both sound and movements.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, along with the following gestures: {MOTION_HOLDER}."},

    {'input': f"Please create a speech that comes with gestures.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, accompanied by body movements: {MOTION_HOLDER}."},

    {'input': f"I need a speech clip with gestures for emphasis.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the accompanying gestures are: {MOTION_HOLDER}."},

    {'input': f"Create a random speech with gestures to match.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Generate a speech with matching body movements.",
     'output': f"The speech sounds like this: {AUDIO_HOLDER}, and the movements are: {MOTION_HOLDER}."},

    {'input': f"I want a speech clip along with gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the gestures: {MOTION_HOLDER}."},

    {'input': f"Please provide a speech and accompanying gestures.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the corresponding gestures are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech clip with synchronized gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, accompanied by the gestures: {MOTION_HOLDER}."},

    {'input': f"I need a speech clip with matching gestures.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Generate a random speech with corresponding gestures.",
     'output': f"The speech sounds like this: {AUDIO_HOLDER}, and the movements are: {MOTION_HOLDER}."},

    {'input': f"Create a speech with body language to accompany the audio.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the body language: {MOTION_HOLDER}."},

    {'input': f"Generate a speech with hand gestures and body movements.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, accompanied by these movements: {MOTION_HOLDER}."},

    {'input': f"Please produce a speech clip with synchronized body language.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Create a random speech with gestures included.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech with matching body gestures.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the gestures: {MOTION_HOLDER}."},

    {'input': f"Please create a speech clip that includes gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, accompanied by gestures: {MOTION_HOLDER}."},

    {'input': f"I need a speech clip with synchronized movements.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the body movements are: {MOTION_HOLDER}."},

    {'input': f"Create a speech with gestures to accompany it.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Generate a random speech with corresponding body movements.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the accompanying movements are: {MOTION_HOLDER}."},

    {'input': f"I need a speech with gestures to accompany it.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the body movements: {MOTION_HOLDER}."},

    {'input': f"Please create a random speech with matching gestures.",
     'output': f"The speech sounds like this: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech with synchronized gestures for emphasis.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the matching gestures: {MOTION_HOLDER}."},

    {'input': f"Generate a speech clip with synchronized body language.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the gestures: {MOTION_HOLDER}."},

    {'input': f"Create a random speech with gestures for added effect.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech with gestures that match the audio.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the body movements: {MOTION_HOLDER}."},

    {'input': f"Please provide a speech that includes both audio and gestures.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Create a speech with both sound and body movements.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the gestures: {MOTION_HOLDER}."},

    {'input': f"I need a speech clip with synchronized body gestures.",
     'output': f"The audio sounds like this: {AUDIO_HOLDER}, and the movements are: {MOTION_HOLDER}."},

    {'input': f"Please generate a speech with body movements to match.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech with both audio and gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the corresponding gestures: {MOTION_HOLDER}."},

    {'input': f"Generate a speech with gestures to go with it.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Create a random speech with synchronized gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the matching gestures: {MOTION_HOLDER}."},

    {'input': f"Please provide a speech clip that comes with gestures.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Create a speech with both sound and body gestures.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}, and the corresponding gestures: {MOTION_HOLDER}."},

    {'input': f"Generate a random speech clip along with gestures.",
     'output': f"The speech sounds like this: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech that includes synchronized gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the gestures: {MOTION_HOLDER}."},

    {'input': f"I need a speech that comes with some gestures.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Please generate a speech with accompanying gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the body movements: {MOTION_HOLDER}."},

    {'input': f"Create a random speech that comes with gestures.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech with gestures to go with it.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the body gestures: {MOTION_HOLDER}."},

    {'input': f"Generate a random speech with synchronized body gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the matching gestures: {MOTION_HOLDER}."},

    {'input': f"Create a speech with audio and gestures together.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the corresponding gestures: {MOTION_HOLDER}."},

    {'input': f"Please create a speech clip with synchronized movements.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the movements are: {MOTION_HOLDER}."},

    {'input': f"Produce a speech that includes audio and body gestures.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the accompanying gestures: {MOTION_HOLDER}."},

    {'input': f"Create a random speech with synchronized gestures included.",
     'output': f"The audio is: {AUDIO_HOLDER}, and the gestures are: {MOTION_HOLDER}."},

    {'input': f"Generate a speech clip with synchronized body movements.",
     'output': f"The speech is: {AUDIO_HOLDER}, and the corresponding gestures: {MOTION_HOLDER}."},

    {'input': f"Produce a speech with gestures added for emphasis.",
     'output': f"Here’s the speech: {AUDIO_HOLDER}, and the accompanying gestures: {MOTION_HOLDER}."},

]
