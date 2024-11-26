from mmotion.utils.task.modality import CAPTION_HOLDER, MUSIC_HOLDER, MOTION_HOLDER

TM2D_TEMPLATE = [
    {'input': f"Perform a dance based on my description: {CAPTION_HOLDER} and this background music: {MUSIC_HOLDER}",
     'output': f"Okay, {MOTION_HOLDER}"},
    {'input': f"Perform a dance based on my description: {CAPTION_HOLDER} and this background music: {MUSIC_HOLDER}",
     'output': f"Okay, {MOTION_HOLDER}"},

    {'input': f"Can you dance to this music: {MUSIC_HOLDER} with the following description: {CAPTION_HOLDER}?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance that matches the description: {CAPTION_HOLDER} with the music: {MUSIC_HOLDER}.",
     'output': f"I’ve got it! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Please dance according to the description: {CAPTION_HOLDER}, using this music: {MUSIC_HOLDER}.",
     'output': f"Alright, here’s how it goes: {MOTION_HOLDER}"},

    {'input': f"Generate a dance that fits this description: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance I’ve come up with: {MOTION_HOLDER}"},

    {'input': f"Use the following description: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER} to perform a dance.",
     'output': f"Here’s the dance for you: {MOTION_HOLDER}"},

    {'input': f"Dance to this music: {MUSIC_HOLDER} with these instructions: {CAPTION_HOLDER}.",
     'output': f"Got it! Here’s the dance: {MOTION_HOLDER}"},

    {
        'input': f"Perform a dance that matches this description: {CAPTION_HOLDER} and plays with this music: {MUSIC_HOLDER}.",
        'output': f"I’ve created this dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance based on this description: {CAPTION_HOLDER} and music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance performance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance using these guidelines: {CAPTION_HOLDER} along with this music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance I’ve made: {MOTION_HOLDER}"},

    {'input': f"Perform a dance based on the given description: {CAPTION_HOLDER} with this music: {MUSIC_HOLDER}.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {
        'input': f"Can you create a dance to this music: {MUSIC_HOLDER} while following this description: {CAPTION_HOLDER}?",
        'output': f"Sure! Here’s how the dance looks: {MOTION_HOLDER}"},

    {
        'input': f"Please make a dance that matches this description: {CAPTION_HOLDER} with background music: {MUSIC_HOLDER}.",
        'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Dance according to this description: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER}.",
     'output': f"Here’s the result: {MOTION_HOLDER}"},

    {'input': f"Use this description: {CAPTION_HOLDER} and music: {MUSIC_HOLDER} to create a dance.",
     'output': f"Here’s the dance performance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance using the following description: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER}.",
     'output': f"Here’s what I’ve come up with: {MOTION_HOLDER}"},

    {'input': f"Perform a dance based on these guidelines: {CAPTION_HOLDER} and music: {MUSIC_HOLDER}.",
     'output': f"Okay, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance using this description: {CAPTION_HOLDER} and this background music: {MUSIC_HOLDER}.",
     'output': f"I’ve generated this dance: {MOTION_HOLDER}"},

    {'input': f"Use this description: {CAPTION_HOLDER} and the given music: {MUSIC_HOLDER} to perform a dance.",
     'output': f"Here’s the dance I created: {MOTION_HOLDER}"},

    {
        'input': f"Dance according to the following instructions: {CAPTION_HOLDER} while playing this music: {MUSIC_HOLDER}.",
        'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance that follows this description: {CAPTION_HOLDER} and fits this music: {MUSIC_HOLDER}.",
     'output': f"I’ve choreographed this dance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance using these details: {CAPTION_HOLDER} with background music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you create a dance based on this description: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER}?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Perform a dance that matches this description: {CAPTION_HOLDER} along with this music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance I came up with: {MOTION_HOLDER}"},

    {'input': f"Use the following description: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER} to make a dance.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance using this description: {CAPTION_HOLDER} and background music: {MUSIC_HOLDER}.",
     'output': f"I’ve created this dance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance that aligns with this description: {CAPTION_HOLDER} and music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Dance according to this description: {CAPTION_HOLDER} with the music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance I’ve made: {MOTION_HOLDER}"},

    {'input': f"Can you perform a dance based on these instructions: {CAPTION_HOLDER} with this music: {MUSIC_HOLDER}?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance that follows this description: {CAPTION_HOLDER} with this music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance I created: {MOTION_HOLDER}"},

    {'input': f"Use this description: {CAPTION_HOLDER} and music: {MUSIC_HOLDER} to generate a dance.",
     'output': f"Here’s the result: {MOTION_HOLDER}"},

    {'input': f"Generate a dance according to this description: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Perform a dance based on the following description: {CAPTION_HOLDER} with music: {MUSIC_HOLDER}.",
     'output': f"Alright, here’s how it looks: {MOTION_HOLDER}"},

    {'input': f"Create a dance that matches these instructions: {CAPTION_HOLDER} using this music: {MUSIC_HOLDER}.",
     'output': f"I’ve choreographed this dance: {MOTION_HOLDER}"},

    {'input': f"Dance according to the description: {CAPTION_HOLDER} and play this music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance that fits the description: {CAPTION_HOLDER} along with the music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance I came up with: {MOTION_HOLDER}"},

    {'input': f"Can you create a dance based on the given description: {CAPTION_HOLDER} and music: {MUSIC_HOLDER}?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Use this description: {CAPTION_HOLDER} with the following music: {MUSIC_HOLDER} to create a dance.",
     'output': f"Here’s the result: {MOTION_HOLDER}"},

    {'input': f"Create a dance using the instructions: {CAPTION_HOLDER} and this background music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Perform a dance that matches the description: {CAPTION_HOLDER} with the music: {MUSIC_HOLDER}.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance using these guidelines: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER}.",
     'output': f"I’ve created this dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance based on the description: {CAPTION_HOLDER} with this music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance performance: {MOTION_HOLDER}"},

    {'input': f"Use the following description: {CAPTION_HOLDER} and music: {MUSIC_HOLDER} to perform a dance.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Perform a dance according to this description: {CAPTION_HOLDER} with the music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance I generated: {MOTION_HOLDER}"},

    {
        'input': f"Create a dance that fits this description: {CAPTION_HOLDER} with this background music: {MUSIC_HOLDER}.",
        'output': f"Here’s the dance I choreographed: {MOTION_HOLDER}"},

    {'input': f"Dance to this music: {MUSIC_HOLDER} following the instructions: {CAPTION_HOLDER}.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance using this description: {CAPTION_HOLDER} and this music: {MUSIC_HOLDER}.",
     'output': f"I’ve made this dance: {MOTION_HOLDER}"},

    {'input': f"Perform a dance based on this description: {CAPTION_HOLDER} with the music: {MUSIC_HOLDER}.",
     'output': f"Here’s the dance performance: {MOTION_HOLDER}"},
]
