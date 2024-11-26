from mmotion.utils.task.modality import GENRE_HOLDER, MOTION_HOLDER, MUSIC_HOLDER

N2GDM_TEMPLATE = [
    {'input': f"Pick any genre, then compose a piece of music and add a dance to it.",
     'output': f"Alright, the genre I chose is: {GENRE_HOLDER}. The dance and music are as follows: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and create both music and a matching dance.",
     'output': f"I picked the genre {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a random genre and generate a song with a dance to go with it.",
     'output': f"The genre is {GENRE_HOLDER}. Enjoy the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Generate music and a dance for any genre you choose.",
     'output': f"I’ve selected {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre and create a song and dance for it.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here’s the music and the matching dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre, compose music, and generate a dance to accompany it.",
     'output': f"I chose {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre and give me a piece of music with a dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the music and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre and create a song along with a dance.",
     'output': f"The genre I’ve chosen is {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a random genre and make both a music track and a dance.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Generate a song and a dance based on any genre you like.",
     'output': f"I selected {GENRE_HOLDER}. Here are your music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick any genre and generate a track along with a dance routine.",
     'output': f"The genre I picked is {GENRE_HOLDER}. Here’s the song and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and create both a piece of music and a dance.",
     'output': f"I selected {GENRE_HOLDER}. Here’s the music and the matching dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre of your choice and generate music and a dance.",
     'output': f"I chose {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre, then compose a song and create a dance for it.",
     'output': f"The genre I selected is {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre, compose music, and generate a dance performance.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre and generate a song along with a dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre and create both a song and a dance routine.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the track and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and generate a piece of music with a dance.",
     'output': f"I’ve selected {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre of your choice and make both a track and a dance.",
     'output': f"I picked {GENRE_HOLDER}. Here’s the music and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a random genre and create a song and a dance for it.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and compose both music and a matching dance.",
     'output': f"I’ve selected {GENRE_HOLDER}. Here are the song and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre and generate a track with a dance routine.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre and make a song along with a dance performance.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre, compose a song, and generate a dance.",
     'output': f"The genre I selected is {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a random genre and create both a track and a dance.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick any genre, then compose music and generate a dance for it.",
     'output': f"The genre I picked is {GENRE_HOLDER}. Here’s the song and the matching dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and create both a piece of music and a dance.",
     'output': f"I’ve selected {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select any genre and generate a track with a dance performance.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here are your music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre, compose a song, and create a matching dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the track and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre and generate a song with a dance routine.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a random genre and create both music and a dance.",
     'output': f"The genre I picked is {GENRE_HOLDER}. Here’s the song and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre, then compose a piece of music and a dance.",
     'output': f"The genre I selected is {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and generate both a song and a dance for it.",
     'output': f"I’ve selected {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre and create both a track and a dance performance.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the music and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre, make a song, and generate a matching dance.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre, then generate both music and a dance routine.",
     'output': f"The genre I selected is {GENRE_HOLDER}. Here’s the song and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a random genre and create a song with a matching dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick any genre and compose a piece of music with a dance.",
     'output': f"I’ve selected {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and generate both a music track and a dance.",
     'output': f"I picked {GENRE_HOLDER}. Here’s the song and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre, then make a song and a matching dance routine.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick any genre, create a song, and generate a dance for it.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here’s the track and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre and generate both a song and a dance performance.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre, then compose a piece of music and create a dance.",
     'output': f"The genre I picked is {GENRE_HOLDER}. Here’s the music and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre, create both music and a dance for it.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here are the music and the matching dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre, compose a track, and generate a dance routine.",
     'output': f"The genre I selected is {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select any genre and make a song along with a dance.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here’s the track and the matching dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick any genre, then create both a track and a dance for it.",
     'output': f"The genre is {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre, generate a piece of music and a dance performance.",
     'output': f"I selected {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a random genre and generate both a song and a dance.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre and create both music and a matching dance performance.",
     'output': f"I’ve selected {GENRE_HOLDER}. Here’s the song and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre, compose music, and generate a dance to match.",
     'output': f"The genre I picked is {GENRE_HOLDER}. Here’s the track and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre and create both a song and a dance routine.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre and make a piece of music along with a dance.",
     'output': f"I selected {GENRE_HOLDER}. Here’s the music and the matching dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre, then generate both music and a dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here are the track and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select any genre and create both a music track and a dance.",
     'output': f"I picked {GENRE_HOLDER}. Here’s the song and its matching dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a random genre and compose a song along with a dance.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and generate both a track and a dance performance.",
     'output': f"The genre I picked is {GENRE_HOLDER}. Here’s the song and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre, then create a piece of music and a dance for it.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here’s the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick any genre, compose music, and generate a matching dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and make both a song and a dance routine.",
     'output': f"The genre I picked is {GENRE_HOLDER}. Here’s the track and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a random genre and generate both a song and a dance.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre and create a song along with a matching dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the song and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre, then make both a track and a dance performance.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre and create both music and a dance routine.",
     'output': f"I picked {GENRE_HOLDER}. Here’s the song and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick any genre, generate a track, and create a dance for it.",
     'output': f"The genre I selected is {GENRE_HOLDER}. Here’s the track and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose a genre and compose both a song and a matching dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here are the music and dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a random genre and generate both music and a dance performance.",
     'output': f"The genre I chose is {GENRE_HOLDER}. Here’s the song and its dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Pick a genre and create a song and a dance for it.",
     'output': f"I’ve chosen {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Choose any genre, then compose a track and generate a dance.",
     'output': f"The genre is {GENRE_HOLDER}. Here’s the song and the matching dance: {MUSIC_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Select a genre of your choice and generate both a song and a dance.",
     'output': f"I selected {GENRE_HOLDER}. Here are the music and the dance: {MUSIC_HOLDER}{MOTION_HOLDER}"}
]
