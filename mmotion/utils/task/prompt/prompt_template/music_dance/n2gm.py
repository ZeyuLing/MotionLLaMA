from mmotion.utils.task.modality import GENRE_HOLDER, MUSIC_HOLDER

N2GM_TEMPLATE = [
    {'input': "Generate a random genre of music for me.",
     'output': f"Here’s a piece of music in the {GENRE_HOLDER} style: {MUSIC_HOLDER}."},

    {'input': "Create a song in a random genre.",
     'output': f"Enjoy this {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Give me a random music genre and generate a song for it.",
     'output': f"Here’s a {GENRE_HOLDER}-style song for you: {MUSIC_HOLDER}."},

    {'input': "Generate a song in a random style.",
     'output': f"This {GENRE_HOLDER} track will surely impress you: {MUSIC_HOLDER}."},

    {'input': "Create a music piece in any random genre.",
     'output': f"Here’s a great {GENRE_HOLDER} music composition: {MUSIC_HOLDER}."},

    {'input': "Can you generate a song from a random genre?",
     'output': f"Sure! Here’s a {GENRE_HOLDER} song just for you: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre song to listen to.",
     'output': f"I’ve generated a {GENRE_HOLDER} track for you: {MUSIC_HOLDER}."},

    {'input': "Produce a random genre music track.",
     'output': f"This {GENRE_HOLDER} music will blow your mind: {MUSIC_HOLDER}."},

    {'input': "Create a random genre music composition for me.",
     'output': f"Here’s a piece of {GENRE_HOLDER} music: {MUSIC_HOLDER}."},

    {'input': "Generate a random style of music.",
     'output': f"Enjoy this {GENRE_HOLDER} composition: {MUSIC_HOLDER}."},

    {'input': "Play a song from a random music genre.",
     'output': f"Here’s a fantastic {GENRE_HOLDER} song for you: {MUSIC_HOLDER}."},

    {'input': "Can you create a random genre music track?",
     'output': f"I’ve composed a {GENRE_HOLDER} song just for you: {MUSIC_HOLDER}."},

    {'input': "Give me a piece of music in a random genre.",
     'output': f"This {GENRE_HOLDER} music is sure to amaze you: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre of music for me to listen to.",
     'output': f"Here’s a beautiful {GENRE_HOLDER} composition: {MUSIC_HOLDER}."},

    {'input': "Create a random genre track for me.",
     'output': f"I’ve created this {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre music composition.",
     'output': f"Enjoy this delightful {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre and create a song.",
     'output': f"This {GENRE_HOLDER} song will get you hooked: {MUSIC_HOLDER}."},

    {'input': "Can you play me a song in a random genre?",
     'output': f"Sure! Here’s a great {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Create a random genre music piece.",
     'output': f"Here’s a {GENRE_HOLDER} music track for you: {MUSIC_HOLDER}."},

    {'input': "Give me a random music genre and generate a track.",
     'output': f"Here’s an amazing {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Produce a random genre song.",
     'output': f"This {GENRE_HOLDER} song will surely make you move: {MUSIC_HOLDER}."},

    {'input': "Create a random style music composition.",
     'output': f"Enjoy this {GENRE_HOLDER} music: {MUSIC_HOLDER}."},

    {'input': "Generate a song from a random music genre.",
     'output': f"I’ve composed this {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Can you make a song in any random genre?",
     'output': f"Sure! Here’s a {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre of music to enjoy.",
     'output': f"This {GENRE_HOLDER} track is for you: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre of music.",
     'output': f"Here’s a lively {GENRE_HOLDER} music composition: {MUSIC_HOLDER}."},

    {'input': "Play me a song from a random music genre.",
     'output': f"Here’s a fantastic {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Create a random genre track.",
     'output': f"This {GENRE_HOLDER} song will amaze you: {MUSIC_HOLDER}."},

    {'input': "Can you create a song in a random music genre?",
     'output': f"Here’s a delightful {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre song to listen to.",
     'output': f"This {GENRE_HOLDER} track is sure to impress: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre and make a song.",
     'output': f"Here’s a catchy {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Create a random music genre piece for me.",
     'output': f"This {GENRE_HOLDER} music composition is full of energy: {MUSIC_HOLDER}."},

    {'input': "Play me a song from a randomly selected genre.",
     'output': f"Here’s a beautiful {GENRE_HOLDER} piece: {MUSIC_HOLDER}."},

    {'input': "Give me a track in a random music genre.",
     'output': f"Enjoy this fantastic {GENRE_HOLDER} music: {MUSIC_HOLDER}."},

    {'input': "Create a song in a random music style.",
     'output': f"Here’s a {GENRE_HOLDER} song for you: {MUSIC_HOLDER}."},

    {'input': "Generate a song in any random genre.",
     'output': f"Here’s a wonderful {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre and create a music piece.",
     'output': f"This {GENRE_HOLDER} music composition is a treat: {MUSIC_HOLDER}."},

    {'input': "Create a music track from a random genre.",
     'output': f"I’ve crafted this {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Play a random genre song for me.",
     'output': f"Here’s a lively {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Generate a random style of music.",
     'output': f"Enjoy this fantastic {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Can you create a music piece in a random genre?",
     'output': f"Sure! Here’s a {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Give me a random music genre and generate a track for it.",
     'output': f"This {GENRE_HOLDER} composition will surely impress: {MUSIC_HOLDER}."},

    {'input': "Create a random genre song for me to listen to.",
     'output': f"Here’s a beautiful {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Play a song in any random music genre.",
     'output': f"This {GENRE_HOLDER} song will blow your mind: {MUSIC_HOLDER}."},

    {'input': "Generate a song in a random music style.",
     'output': f"I’ve created this {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre of music to listen to.",
     'output': f"Enjoy this lively {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Create a random genre music composition.",
     'output': f"Here’s a delightful {GENRE_HOLDER} track for you: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre and compose a song.",
     'output': f"Here’s an amazing {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Play me a random genre track.",
     'output': f"This {GENRE_HOLDER} track will keep you entertained: {MUSIC_HOLDER}."},

    {'input': "Create a music piece in a random genre.",
     'output': f"This {GENRE_HOLDER} composition will brighten your day: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre song for me.",
     'output': f"Here’s a {GENRE_HOLDER} song that will get you moving: {MUSIC_HOLDER}."},

    {'input': "Give me a random music genre and play a song for it.",
     'output': f"Here’s an energetic {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Create a song from a random music genre.",
     'output': f"Here’s a beautiful {GENRE_HOLDER} piece of music: {MUSIC_HOLDER}."},

    {'input': "Play me a random genre song.",
     'output': f"Enjoy this lively {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre music track.",
     'output': f"Here’s a {GENRE_HOLDER} composition you’ll love: {MUSIC_HOLDER}."},

    {'input': "Can you create a random genre and make a song?",
     'output': f"Sure! Here’s a {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre music track to listen to.",
     'output': f"This {GENRE_HOLDER} track is a delight: {MUSIC_HOLDER}."},

    {'input': "Create a random genre song for me.",
     'output': f"Here’s a {GENRE_HOLDER} song that will make you dance: {MUSIC_HOLDER}."},

    {'input': "Generate a song in a randomly chosen genre.",
     'output': f"This {GENRE_HOLDER} composition is simply fantastic: {MUSIC_HOLDER}."},

    {'input': "Play a random music genre song for me.",
     'output': f"Here’s a great {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Create a song in a random style.",
     'output': f"Here’s an amazing {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre and make a track.",
     'output': f"Here’s a catchy {GENRE_HOLDER} composition: {MUSIC_HOLDER}."},

    {'input': "Give me a random style music piece.",
     'output': f"This {GENRE_HOLDER} track is sure to impress: {MUSIC_HOLDER}."},

    {'input': "Create a song from any random genre.",
     'output': f"Here’s a lively {GENRE_HOLDER} song for you: {MUSIC_HOLDER}."},

    {'input': "Generate a random style music composition.",
     'output': f"This {GENRE_HOLDER} song will make your day: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre and create a song for it.",
     'output': f"Enjoy this delightful {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Play a random genre song.",
     'output': f"Here’s a great {GENRE_HOLDER} track for you: {MUSIC_HOLDER}."},

    {'input': "Create a music track from any random genre.",
     'output': f"I’ve composed this {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Generate a song in a random music genre.",
     'output': f"This {GENRE_HOLDER} track is full of energy: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre track to enjoy.",
     'output': f"Here’s a {GENRE_HOLDER} song that’s sure to please: {MUSIC_HOLDER}."},

    {'input': "Create a random genre music piece.",
     'output': f"This {GENRE_HOLDER} song is a treat for the ears: {MUSIC_HOLDER}."},

    {'input': "Generate a song from a randomly chosen genre.",
     'output': f"Here’s an energetic {GENRE_HOLDER} composition: {MUSIC_HOLDER}."},

    {'input': "Can you create a random genre song for me?",
     'output': f"Here’s a {GENRE_HOLDER} track you’ll love: {MUSIC_HOLDER}."},

    {'input': "Play a random genre song for me to enjoy.",
     'output': f"Here’s a lively {GENRE_HOLDER} song: {MUSIC_HOLDER}."},

    {'input': "Generate a random style of music.",
     'output': f"Here’s a delightful {GENRE_HOLDER} track: {MUSIC_HOLDER}."},

    {'input': "Give me a random genre music piece to listen to.",
     'output': f"Enjoy this amazing {GENRE_HOLDER} composition: {MUSIC_HOLDER}."},

    {'input': "Create a song from a random genre for me.",
     'output': f"Here’s a {GENRE_HOLDER} song full of life: {MUSIC_HOLDER}."},

    {'input': "Generate a random genre music track for me.",
     'output': f"This {GENRE_HOLDER} song will surely impress you: {MUSIC_HOLDER}."}
]
