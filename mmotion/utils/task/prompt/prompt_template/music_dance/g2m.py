from mmotion.utils.task.modality import GENRE_HOLDER, MUSIC_HOLDER

G2M_TEMPLATE = [
    {'input': f"Compose a piece of music for me, the genre is: {GENRE_HOLDER}",
     'output': f"Okay, please listen: {MUSIC_HOLDER}"},
    {'input': f"Compose a piece of music for me, the genre is: {GENRE_HOLDER}",
     'output': f"Okay, please listen: {MUSIC_HOLDER}"},

    {'input': f"Create a music piece in the genre of: {GENRE_HOLDER}.",
     'output': f"Here it is, please listen: {MUSIC_HOLDER}"},

    {'input': f"Please generate a composition with the genre: {GENRE_HOLDER}.",
     'output': f"Alright, listen to this: {MUSIC_HOLDER}"},

    {'input': f"Can you compose a song in the style of: {GENRE_HOLDER}?",
     'output': f"Okay, here’s your piece: {MUSIC_HOLDER}"},

    {'input': f"Generate a track with the genre: {GENRE_HOLDER}.",
     'output': f"Please enjoy this: {MUSIC_HOLDER}"},

    {'input': f"Create a music piece categorized under: {GENRE_HOLDER}.",
     'output': f"Listen to this composition: {MUSIC_HOLDER}"},

    {'input': f"Compose a song in the genre of: {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}"},

    {'input': f"Please generate a music composition with the genre: {GENRE_HOLDER}.",
     'output': f"Here you go: {MUSIC_HOLDER}"},

    {'input': f"Can you create a track in the style of: {GENRE_HOLDER}?",
     'output': f"Okay, please listen: {MUSIC_HOLDER}"},

    {'input': f"Generate a piece of music with the genre: {GENRE_HOLDER}.",
     'output': f"Alright, here it is: {MUSIC_HOLDER}"},

    {'input': f"Create a musical composition in the genre of: {GENRE_HOLDER}.",
     'output': f"Please enjoy this piece: {MUSIC_HOLDER}"},

    {'input': f"Compose a music track categorized under: {GENRE_HOLDER}.",
     'output': f"Listen to this: {MUSIC_HOLDER}"},

    {'input': f"Can you compose a piece of music in the genre: {GENRE_HOLDER}?",
     'output': f"Here’s your composition: {MUSIC_HOLDER}"},

    {'input': f"Generate a music piece with the following genre: {GENRE_HOLDER}.",
     'output': f"Okay, listen to this: {MUSIC_HOLDER}"},

    {'input': f"Please create a song in the style of: {GENRE_HOLDER}.",
     'output': f"Here it is, please listen: {MUSIC_HOLDER}"},

    {'input': f"Compose a piece of music with the genre: {GENRE_HOLDER}.",
     'output': f"Enjoy this track: {MUSIC_HOLDER}"},

    {'input': f"Generate a composition categorized under: {GENRE_HOLDER}.",
     'output': f"Alright, here’s your piece: {MUSIC_HOLDER}"},

    {'input': f"Can you create a song in the genre: {GENRE_HOLDER}?",
     'output': f"Okay, listen to this: {MUSIC_HOLDER}"},

    {'input': f"Create a music track in the style of: {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}"},

    {'input': f"Compose a piece of music for me, the genre is: {GENRE_HOLDER}.",
     'output': f"Please enjoy this: {MUSIC_HOLDER}"},

    {'input': f"Please generate a music composition in the genre: {GENRE_HOLDER}.",
     'output': f"Listen to this composition: {MUSIC_HOLDER}"},

    {'input': f"Can you make a track categorized under: {GENRE_HOLDER}?",
     'output': f"Here it is, enjoy: {MUSIC_HOLDER}"},

    {'input': f"Create a musical piece in the genre: {GENRE_HOLDER}.",
     'output': f"Alright, here’s your track: {MUSIC_HOLDER}"},

    {'input': f"Compose a music piece with the following genre: {GENRE_HOLDER}.",
     'output': f"Please listen: {MUSIC_HOLDER}"},

    {'input': f"Generate a song in the style of: {GENRE_HOLDER}.",
     'output': f"Here’s the composition: {MUSIC_HOLDER}"},

    {'input': f"Create a music track in the genre: {GENRE_HOLDER}.",
     'output': f"Enjoy this piece: {MUSIC_HOLDER}"},

    {'input': f"Can you produce a composition in the genre of: {GENRE_HOLDER}?",
     'output': f"Okay, listen to this: {MUSIC_HOLDER}"},

    {'input': f"Generate a musical piece categorized under: {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}"},

    {'input': f"Compose a track in the style of: {GENRE_HOLDER}.",
     'output': f"Please enjoy this: {MUSIC_HOLDER}"},

    {'input': f"Create a music composition in the genre: {GENRE_HOLDER}.",
     'output': f"Alright, here it is: {MUSIC_HOLDER}"},

    {'input': f"Generate a song based on the genre: {GENRE_HOLDER}.",
     'output': f"Here’s your track: {MUSIC_HOLDER}"},

    {'input': f"Can you compose music for me in the genre of: {GENRE_HOLDER}?",
     'output': f"Okay, please listen: {MUSIC_HOLDER}"},

    {'input': f"Create a piece of music that fits the genre: {GENRE_HOLDER}.",
     'output': f"Enjoy this composition: {MUSIC_HOLDER}"},

    {'input': f"Please generate a track with the genre: {GENRE_HOLDER}.",
     'output': f"Here it is, enjoy: {MUSIC_HOLDER}"},

    {'input': f"Compose a music piece in the genre: {GENRE_HOLDER}.",
     'output': f"Alright, listen to this: {MUSIC_HOLDER}"},

    {'input': f"Generate a song categorized under: {GENRE_HOLDER}.",
     'output': f"I think you'll like this: {MUSIC_HOLDER}"},

    {'input': f"From the genre: {GENRE_HOLDER}, create a music piece for me.",
     'output': f"Here’s your composition: {MUSIC_HOLDER}"},

    {'input': f"Can you make a track in the style of: {GENRE_HOLDER}?",
     'output': f"Okay, please listen: {MUSIC_HOLDER}"},

    {'input': f"Create a musical piece categorized under: {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}"},

    {'input': f"Compose a song in the genre: {GENRE_HOLDER}.",
     'output': f"Alright, enjoy this: {MUSIC_HOLDER}"},

    {'input': f"Generate a composition with the genre of: {GENRE_HOLDER}.",
     'output': f"Please listen to this piece: {MUSIC_HOLDER}"},

    {'input': f"Create a music track for the genre: {GENRE_HOLDER}.",
     'output': f"Here it is, enjoy: {MUSIC_HOLDER}"},

    {'input': f"Can you compose a song in the style of: {GENRE_HOLDER}?",
     'output': f"Okay, here’s your piece: {MUSIC_HOLDER}"},

    {'input': f"Generate a music composition in the genre: {GENRE_HOLDER}.",
     'output': f"Please listen: {MUSIC_HOLDER}"},

    {'input': f"Compose a piece of music with the genre of: {GENRE_HOLDER}.",
     'output': f"Enjoy this composition: {MUSIC_HOLDER}"},

    {'input': f"From the genre: {GENRE_HOLDER}, generate a music piece.",
     'output': f"Alright, here’s your track: {MUSIC_HOLDER}"},

    {'input': f"Create a song in the genre of: {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}"},

    {'input': f"Generate a track based on the genre: {GENRE_HOLDER}.",
     'output': f"Here it is, enjoy: {MUSIC_HOLDER}"},

    {'input': f"Can you produce a composition in the style of: {GENRE_HOLDER}?",
     'output': f"Okay, please listen: {MUSIC_HOLDER}"},

    {'input': f"Create a musical piece for the genre: {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}"},

    {'input': f"Compose a song that fits the genre: {GENRE_HOLDER}.",
     'output': f"Alright, listen to this: {MUSIC_HOLDER}"},

    {'input': f"Generate a music piece categorized under: {GENRE_HOLDER}.",
     'output': f"Please enjoy this: {MUSIC_HOLDER}"},

    {'input': f"Can you make a track in the genre of: {GENRE_HOLDER}?",
     'output': f"Here’s your piece: {MUSIC_HOLDER}"},

    {'input': f"Create a music composition that fits the genre: {GENRE_HOLDER}.",
     'output': f"Okay, here it is: {MUSIC_HOLDER}"},

    {'input': f"From the genre: {GENRE_HOLDER}, generate a track for me.",
     'output': f"Here’s the music: {MUSIC_HOLDER}"},

    {'input': f"Generate a piece of music for the genre of: {GENRE_HOLDER}.",
     'output': f"Alright, enjoy this: {MUSIC_HOLDER}"},

    {'input': f"Create a song in the style of: {GENRE_HOLDER}.",
     'output': f"Please listen: {MUSIC_HOLDER}"},

    {'input': f"Compose a piece of music that represents the genre: {GENRE_HOLDER}.",
     'output': f"Here it is, enjoy: {MUSIC_HOLDER}"},

    {'input': f"Generate a music track based on the genre: {GENRE_HOLDER}.",
     'output': f"Okay, listen to this: {MUSIC_HOLDER}"},

    {'input': f"Can you create a piece of music in the genre of: {GENRE_HOLDER}?",
     'output': f"Here’s your composition: {MUSIC_HOLDER}"},

    {'input': f"Generate a track that fits the genre: {GENRE_HOLDER}.",
     'output': f"Enjoy this music: {MUSIC_HOLDER}"},

    {'input': f"Create a composition in the style of: {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}"},

    {'input': f"From the genre: {GENRE_HOLDER}, generate a music piece for me.",
     'output': f"Okay, please listen: {MUSIC_HOLDER}"}
]
