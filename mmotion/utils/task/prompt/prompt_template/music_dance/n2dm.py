from mmotion.utils.task.modality import MOTION_HOLDER, MUSIC_HOLDER

N2DM_TEMPLATE = [
    {'input': 'Compose a piece of music for me and pair it with a suitable dance.',
     'output': f"Here’s the music I composed: {MUSIC_HOLDER} along with a dance to match it {MOTION_HOLDER}"},
    {'input': 'Generate a dance along with background music',
     'output': f"Okay, here is the dance and music: {MOTION_HOLDER} {MUSIC_HOLDER}"},
    {'input': f"Create a piece of music and choreograph a dance to go with it.",
     'output': f"Here’s the music I composed: {MUSIC_HOLDER} and the dance to match: {MOTION_HOLDER}"},

    {'input': f"Generate a dance sequence that fits with some music.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the accompanying music: {MUSIC_HOLDER}"},

    {'input': f"Please compose a song and design a dance for it.",
     'output': f"The music is: {MUSIC_HOLDER} and the dance moves are: {MOTION_HOLDER}"},

    {'input': f"Make a piece of music and suggest a dance to go with it.",
     'output': f"Here’s the song: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Can you create some music and a matching dance?",
     'output': f"Here’s the music: {MUSIC_HOLDER} and here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Compose music and pair it with a dance.",
     'output': f"Here’s the music I made: {MUSIC_HOLDER} and the dance to go with it: {MOTION_HOLDER}"},

    {'input': f"Create a dance and a piece of background music to match.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Generate a dance that fits the music.",
     'output': f"The dance is: {MOTION_HOLDER} and the music is: {MUSIC_HOLDER}"},

    {'input': f"Please create a song and some dance moves to match it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance moves: {MOTION_HOLDER}"},

    {'input': f"Produce music and a dance that go together.",
     'output': f"The music is: {MUSIC_HOLDER} and the dance is: {MOTION_HOLDER}"},

    {'input': f"Can you come up with a dance and create a piece of music for it?",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Create a dance routine and compose some background music.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and here’s the music: {MUSIC_HOLDER}"},

    {'input': f"Compose a song and choreograph a dance for it.",
     'output': f"Here’s the song: {MUSIC_HOLDER} and the dance moves: {MOTION_HOLDER}"},

    {'input': f"Generate a music track and pair it with a dance.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Create music and a dance to complement each other.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the complementary dance: {MOTION_HOLDER}"},

    {'input': f"Can you compose music and match it with a suitable dance?",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the matching dance: {MOTION_HOLDER}"},

    {'input': f"Create a piece of music and a dance to accompany it.",
     'output': f"Here’s the music I composed: {MUSIC_HOLDER} and the accompanying dance: {MOTION_HOLDER}"},

    {'input': f"Compose a song and design a dance to go with it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance moves: {MOTION_HOLDER}"},

    {'input': f"Generate a dance routine and create some music to go with it.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Please come up with a piece of music and a dance that match.",
     'output': f"The music is: {MUSIC_HOLDER} and the matching dance: {MOTION_HOLDER}"},

    {'input': f"Produce a dance and some background music to go along with it.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Can you create a dance and an original music track?",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Make a dance routine and compose the music for it.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music I made: {MUSIC_HOLDER}"},

    {'input': f"Generate a piece of music and suggest a dance to go with it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Compose music and design a dance that goes well with it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance moves: {MOTION_HOLDER}"},

    {'input': f"Please generate a dance and create the music for it.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the accompanying music: {MUSIC_HOLDER}"},

    {'input': f"Make some music and create a dance to go with it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance routine and pair it with a suitable song.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Compose a piece of music and generate a dance to accompany it.",
     'output': f"Here’s the music I composed: {MUSIC_HOLDER} and the accompanying dance: {MOTION_HOLDER}"},

    {'input': f"Please generate music and a matching dance sequence.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance sequence: {MOTION_HOLDER}"},

    {'input': f"Create a piece of music and a dance to match.",
     'output': f"Here’s the music I made: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Produce a dance routine and background music that match.",
     'output': f"The dance is: {MOTION_HOLDER} and the music is: {MUSIC_HOLDER}"},

    {'input': f"Generate a song and a dance that go well together.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Can you compose a piece of music and pair it with a dance?",
     'output': f"Here’s the music I composed: {MUSIC_HOLDER} and the dance to go with it: {MOTION_HOLDER}"},

    {'input': f"Make a dance routine and a background track for it.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the background track: {MUSIC_HOLDER}"},

    {'input': f"Generate a music track and some dance moves to match.",
     'output': f"The music is: {MUSIC_HOLDER} and the dance is: {MOTION_HOLDER}"},

    {'input': f"Compose a dance and music that complement each other.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Produce some music and a dance that fit together.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Create a piece of music and a dance sequence for it.",
     'output': f"Here’s the music I created: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance routine and compose a music track.",
     'output': f"Here’s the dance I made: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Please compose a song and suggest a dance for it.",
     'output': f"Here’s the song: {MUSIC_HOLDER} and the suggested dance: {MOTION_HOLDER}"},

    {'input': f"Make some music and create a dance routine to match.",
     'output': f"The music is: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Create a song and design a dance to go along with it.",
     'output': f"Here’s the song: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a music track and a dance routine.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Please create a dance sequence and background music for it.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the background music: {MUSIC_HOLDER}"},

    {'input': f"Produce some music and generate a matching dance.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Can you make a dance and compose some music for it?",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Create a piece of music and suggest a dance to go with it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance and a piece of music that fit together.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Produce a song and a dance that match perfectly.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Please create a dance and compose music for it.",
     'output': f"Here’s the dance I made: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Make up a dance routine and create some background music.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the background music: {MUSIC_HOLDER}"},

    {'input': f"Can you compose a piece of music and suggest a dance?",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Create a song and some dance moves that match it.",
     'output': f"The music is: {MUSIC_HOLDER} and the dance is: {MOTION_HOLDER}"},

    {'input': f"Generate a dance routine along with a music track.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Produce a song and a dance that complement each other.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Create a piece of music and a dance routine to match.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance routine: {MOTION_HOLDER}"},

    {'input': f"Please generate a song and a dance to go along with it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the accompanying dance: {MOTION_HOLDER}"},

    {'input': f"Compose a dance routine and some music to match.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music I composed: {MUSIC_HOLDER}"},

    {'input': f"Produce a dance routine and suggest some background music.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Create a song and choreograph a dance to go with it.",
     'output': f"Here’s the song: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a music track and a dance routine that go well together.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Please create a piece of music and a dance that fits it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the matching dance: {MOTION_HOLDER}"},

    {'input': f"Produce a dance and compose music that goes along with it.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Make a dance routine and suggest a song to match.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the song: {MUSIC_HOLDER}"},

    {'input': f"Can you generate a dance and pair it with music?",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Please create music and a dance routine that match.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Compose a song and generate some dance moves to match.",
     'output': f"The music is: {MUSIC_HOLDER} and the dance is: {MOTION_HOLDER}"},

    {'input': f"Produce some music and pair it with a dance routine.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Make a song and choreograph a dance for it.",
     'output': f"Here’s the song: {MUSIC_HOLDER} and the dance moves: {MOTION_HOLDER}"},

    {'input': f"Create a piece of music and some dance moves to go with it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a dance routine and a music track to go with it.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Can you create a dance and produce some music for it?",
     'output': f"Here’s the dance I created: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Compose a song and pair it with a matching dance.",
     'output': f"Here’s the song: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Please make a dance routine and some background music.",
     'output': f"Here’s the dance: {MOTION_HOLDER} and the music: {MUSIC_HOLDER}"},

    {'input': f"Generate some music and suggest a dance for it.",
     'output': f"The music is: {MUSIC_HOLDER} and the suggested dance: {MOTION_HOLDER}"},

    {'input': f"Make a song and create a dance that fits with it.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}"},

    {'input': f"Produce music and a dance that complement each other.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the complementary dance: {MOTION_HOLDER}"},
]
