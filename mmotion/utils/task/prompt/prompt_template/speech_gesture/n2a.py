from mmotion.utils.task.modality import AUDIO_HOLDER

N2A_TEMPLATE = [
    {
        'input': f'Randomly create a speech for me',
        'output': f"Okay, here's the speech audio: {AUDIO_HOLDER}"
    },
    {'input': f"Can you whip up a random speech for me?",
     'output': f"Sure thing! Here's the speech audio: {AUDIO_HOLDER}"},

    {'input': f"Give me a spontaneous speech, totally random.",
     'output': f"Alright, here you go with the speech audio: {AUDIO_HOLDER}"},

    {'input': f"I need you to create a speech for me, but make it random.",
     'output': f"Done! Here’s the audio you asked for: {AUDIO_HOLDER}"},

    {'input': f"Quick, I need a random speech generated on the spot.",
     'output': f"No problem! Here’s the speech clip: {AUDIO_HOLDER}"},

    {'input': f"Can you just create a speech out of thin air for me?",
     'output': f"Got it! Here’s a freshly generated speech: {AUDIO_HOLDER}"},

    {'input': f"Surprise me with a random speech!",
     'output': f"Sure! Here’s the speech audio I came up with: {AUDIO_HOLDER}"},

    {'input': f"I need a speech, but go ahead and make it completely random.",
     'output': f"Got you! Here’s the speech audio: {AUDIO_HOLDER}"},

    {'input': f"Can you cook up a random speech for me?",
     'output': f"Here’s something I whipped up: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech for me, any topic will do.",
     'output': f"Here’s what I came up with: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech clip, I’m curious what you’ll make.",
     'output': f"Here’s an audio clip, totally random: {AUDIO_HOLDER}"},

    {'input': f"Make up a random speech for me, please.",
     'output': f"Got it! Here’s the speech audio for you: {AUDIO_HOLDER}"},

    {'input': f"Can you give me a random speech, no specific theme?",
     'output': f"Sure! Here's a speech for you: {AUDIO_HOLDER}"},

    {'input': f"Surprise me with a speech, completely random.",
     'output': f"Here’s a random speech audio I generated: {AUDIO_HOLDER}"},

    {'input': f"Give me a random speech, it can be about anything.",
     'output': f"Here’s an audio clip I generated randomly: {AUDIO_HOLDER}"},

    {'input': f"I’d love to hear a random speech, can you make one?",
     'output': f"Absolutely! Here’s the speech audio: {AUDIO_HOLDER}"},

    {'input': f"Just create a random speech, I’m eager to hear what comes up.",
     'output': f"Here’s a spontaneous speech clip: {AUDIO_HOLDER}"},

    {'input': f"Could you generate a speech for me? Totally random please.",
     'output': f"Here’s a randomly generated speech: {AUDIO_HOLDER}"},

    {'input': f"Give me a speech, randomly created of course.",
     'output': f"Sure! Here’s the speech audio I made: {AUDIO_HOLDER}"},

    {'input': f"I’d like a speech, just make it random.",
     'output': f"Here’s what I randomly generated: {AUDIO_HOLDER}"},

    {'input': f"Hit me with a random speech!",
     'output': f"Here’s the audio I came up with: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech for me, let’s see what happens.",
     'output': f"Here’s the result, enjoy: {AUDIO_HOLDER}"},

    {'input': f"Randomly create a speech for me, could be anything.",
     'output': f"Here’s your speech, randomly generated: {AUDIO_HOLDER}"},

    {'input': f"Please, make up a speech on the spot, totally random.",
     'output': f"Here’s what I created: {AUDIO_HOLDER}"},

    {'input': f"Give me something random, like a speech.",
     'output': f"Sure thing! Here’s your random speech: {AUDIO_HOLDER}"},

    {'input': f"Can you improvise a speech for me, something random?",
     'output': f"Absolutely! Here’s what I improvised: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech out of nowhere for me, random style.",
     'output': f"Here’s the speech clip I generated randomly: {AUDIO_HOLDER}"},

    {'input': f"I want a random speech, just go with the flow.",
     'output': f"Here’s the audio I created: {AUDIO_HOLDER}"},

    {'input': f"Make me a random speech, let’s see what comes out.",
     'output': f"Here’s your randomly generated speech: {AUDIO_HOLDER}"},

    {'input': f"Throw together a speech, anything goes.",
     'output': f"Here’s a random audio clip for you: {AUDIO_HOLDER}"},

    {'input': f"Can you throw me a random speech, whatever works?",
     'output': f"Here’s a random speech audio: {AUDIO_HOLDER}"},

    {'input': f"Hit me with a spontaneous speech, no planning!",
     'output': f"Here’s a random speech I put together: {AUDIO_HOLDER}"},

    {'input': f"Give me a random speech, anything you can come up with.",
     'output': f"Here’s what I generated randomly: {AUDIO_HOLDER}"},

    {'input': f"Make a random speech, I’m interested in what you’ll create.",
     'output': f"Here’s a speech clip I made randomly: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech for me, but make it random.",
     'output': f"Here’s the speech clip you asked for: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech clip, whatever comes to mind.",
     'output': f"Here’s the speech audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate something totally random, like a speech?",
     'output': f"Sure, here’s the speech I came up with: {AUDIO_HOLDER}"},

    {'input': f"Give me a random speech, no theme or anything.",
     'output': f"Here’s the audio I generated: {AUDIO_HOLDER}"},

    {'input': f"Just create a speech, make it completely random.",
     'output': f"Here’s what I randomly generated: {AUDIO_HOLDER}"},

    {'input': f"Can you make a random speech for me?",
     'output': f"Sure thing! Here’s your speech: {AUDIO_HOLDER}"},

    {'input': f"Create a speech at random, anything is fine.",
     'output': f"Here’s the random speech audio: {AUDIO_HOLDER}"},

    {'input': f"Surprise me with a totally random speech.",
     'output': f"Here’s what I came up with: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech clip, doesn’t matter what it’s about.",
     'output': f"Here’s the speech audio I made: {AUDIO_HOLDER}"},

    {'input': f"Can you create a speech for me? No specific topic, just random.",
     'output': f"Sure, here’s your randomly generated speech: {AUDIO_HOLDER}"},

    {'input': f"Randomly generate a speech for me, anything goes.",
     'output': f"Here’s what I generated: {AUDIO_HOLDER}"},

    {'input': f"Make up a speech, let it be random.",
     'output': f"Here’s your random speech: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech, I’m curious what will come out.",
     'output': f"Here’s the speech audio I made: {AUDIO_HOLDER}"},

    {'input': f"Can you just create a speech, any random topic?",
     'output': f"Here’s the speech audio clip: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech, I’m ready to listen!",
     'output': f"Here’s what I came up with: {AUDIO_HOLDER}"},

    {'input': f"Make me a speech, anything goes.",
     'output': f"Here’s the random speech audio: {AUDIO_HOLDER}"},

    {'input': f"Please randomly generate a speech clip, no theme needed.",
     'output': f"Here’s your random speech audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech for me, I trust your randomness!",
     'output': f"Here’s the speech clip I made: {AUDIO_HOLDER}"},

    {'input': f"Make up a speech, surprise me.",
     'output': f"Here’s the speech audio I generated: {AUDIO_HOLDER}"},

    {'input': f"Can you improvise a random speech for me?",
     'output': f"Sure thing! Here’s your speech: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech on the spot.",
     'output': f"Here’s your speech audio clip: {AUDIO_HOLDER}"},

    {'input': f"Just create a speech randomly, anything’s fine.",
     'output': f"Here’s what I made for you: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech with no specific direction, totally random.",
     'output': f"Here’s the speech I randomly generated: {AUDIO_HOLDER}"},

    {'input': f"Create a speech at random, no particular topic.",
     'output': f"Here’s the random speech audio: {AUDIO_HOLDER}"},

    {'input': f"Can you generate a random speech, no planning required?",
     'output': f"Here’s your speech audio clip: {AUDIO_HOLDER}"},

    {'input': f"Give me a speech, any random topic will do.",
     'output': f"Here’s what I came up with: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech audio, no rules.",
     'output': f"Here’s the speech clip: {AUDIO_HOLDER}"},

    {'input': f"Generate a speech, let it be random and fun.",
     'output': f"Here’s your random speech: {AUDIO_HOLDER}"},

    {'input': f"Can you whip up a random speech for me, any theme?",
     'output': f"Sure! Here’s the speech I created: {AUDIO_HOLDER}"},

    {'input': f"Create a random speech, just surprise me with it.",
     'output': f"Here’s the speech I randomly generated: {AUDIO_HOLDER}"},

    {'input': f"Can you quickly create a random speech audio clip?",
     'output': f"Here’s your speech: {AUDIO_HOLDER}"},

    {'input': f"Give me a random speech, no specific instructions.",
     'output': f"Here’s the audio I generated: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech for me, anything you want.",
     'output': f"Here’s the speech audio I made: {AUDIO_HOLDER}"},

    {'input': f"Make up a random speech, no particular theme.",
     'output': f"Here’s your random speech: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech, it can be about anything.",
     'output': f"Here’s the speech clip I created: {AUDIO_HOLDER}"},

    {'input': f"Can you improvise a random speech for me?",
     'output': f"Here’s what I improvised: {AUDIO_HOLDER}"},

    {'input': f"Generate a spontaneous speech for me, anything goes.",
     'output': f"Here’s the speech I generated randomly: {AUDIO_HOLDER}"},

    {'input': f"Just make a random speech, surprise me with the content.",
     'output': f"Here’s the speech I generated: {AUDIO_HOLDER}"},

    {'input': f"Give me a spontaneous speech, anything at all.",
     'output': f"Here’s the speech audio clip: {AUDIO_HOLDER}"},

    {'input': f"Produce a speech out of nowhere for me.",
     'output': f"Here’s the speech clip I randomly generated: {AUDIO_HOLDER}"},

    {'input': f"Can you create a speech audio, completely random?",
     'output': f"Here’s your speech audio clip: {AUDIO_HOLDER}"},

    {'input': f"Generate a random speech, whatever you can think of.",
     'output': f"Here’s the audio clip: {AUDIO_HOLDER}"},

    {'input': f"Please make up a speech randomly for me.",
     'output': f"Here’s your random speech audio: {AUDIO_HOLDER}"},

    {'input': f"Produce a random speech for me, no planning necessary.",
     'output': f"Here’s what I came up with: {AUDIO_HOLDER}"},
]
