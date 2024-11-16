from mmotion.utils.task.modality import GENRE_HOLDER, MUSIC_HOLDER, MOTION_HOLDER

G2DM_TEMPLATE = [
    {'input': f"The genre is {GENRE_HOLDER}, generate a matching music piece and dance.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a music track and a dance for the genre: {GENRE_HOLDER}.",
     'output': f"Please enjoy this music: {MUSIC_HOLDER} and dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, generate a piece of music and a dance.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and here’s a dance to match: {MOTION_HOLDER}."},

    {'input': f"Generate a music track and a dance sequence for the genre: {GENRE_HOLDER}.",
     'output': f"Listen to this: {MUSIC_HOLDER}, and watch this dance: {MOTION_HOLDER}."},

    {'input': f"Compose a piece of music in the genre {GENRE_HOLDER} and generate a dance.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a music composition and a corresponding dance for {GENRE_HOLDER}.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Please create a music piece and dance that fit the genre {GENRE_HOLDER}.",
     'output': f"Here is the music: {MUSIC_HOLDER} and a matching dance: {MOTION_HOLDER}."},

    {'input': f"Create a music track in {GENRE_HOLDER} and generate an accompanying dance.",
     'output': f"Here’s your music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a music piece and dance based on the genre: {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance moves: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, produce a matching song and dance.",
     'output': f"The music track is: {MUSIC_HOLDER}, and the dance is: {MOTION_HOLDER}."},

    {'input': f"Create music and a dance routine for the genre: {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and here’s the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a song and a dance based on the genre {GENRE_HOLDER}.",
     'output': f"Enjoy the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose music and a dance for the genre: {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER} and the dance is: {MOTION_HOLDER}."},

    {'input': f"Please generate a song and dance for the genre {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, generate both a song and a dance.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a music track and a dance to accompany the genre: {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance moves: {MOTION_HOLDER}."},

    {'input': f"Generate a song and a dance routine for {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a piece of music and a dance to match the genre {GENRE_HOLDER}.",
     'output': f"Listen to the music: {MUSIC_HOLDER} and watch the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a song and a dance in the genre {GENRE_HOLDER}.",
     'output': f"Here’s the song: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, generate a matching music and dance.",
     'output': f"The music track is: {MUSIC_HOLDER} and the dance is: {MOTION_HOLDER}."},

    {'input': f"Create a music piece and a dance routine for the genre: {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate music and dance that fit the genre {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER} and the dance is: {MOTION_HOLDER}."},

    {'input': f"Please compose a piece of music and a dance for {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a track in the genre {GENRE_HOLDER} and generate a dance.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance moves: {MOTION_HOLDER}."},

    {'input': f"Create a music composition and a dance to match the genre {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a song and a dance performance based on the genre {GENRE_HOLDER}.",
     'output': f"The music track: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose music and create a dance routine for the genre {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance routine: {MOTION_HOLDER}."},

    {'input': f"Please generate a piece of music and a dance for the genre {GENRE_HOLDER}.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a track and a dance sequence to match the genre {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a musical piece and a corresponding dance for {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, compose a song and create a dance.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a music track and a dance routine for the genre {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a song and dance for the genre {GENRE_HOLDER}.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance moves: {MOTION_HOLDER}."},

    {'input': f"Compose a track in the genre {GENRE_HOLDER} and generate a dance.",
     'output': f"Listen to the music: {MUSIC_HOLDER} and watch the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, produce a song and a dance performance.",
     'output': f"The song: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Please generate a music composition and a dance routine for {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER} and the dance: {MOTION_HOLDER}."},

    {'input': f"Create music and a dance to match the genre {GENRE_HOLDER}.",
     'output': f"Here’s the track: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a piece of music and a dance sequence for the genre {GENRE_HOLDER}.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a music track and generate a dance for the genre {GENRE_HOLDER}.",
     'output': f"Listen to this: {MUSIC_HOLDER}, and watch this: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, generate both music and a dance.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a piece of music and a dance to match the genre {GENRE_HOLDER}.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance moves: {MOTION_HOLDER}."},

    {'input': f"Generate music and a dance sequence for the genre {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER}, and the dance is: {MOTION_HOLDER}."},

    {'input': f"Compose a song and a dance performance for the genre {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, generate a piece of music and dance routine.",
     'output': f"The music track: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a track and a dance for the genre {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and the dance moves: {MOTION_HOLDER}."},

    {'input': f"Generate a music composition and a dance performance for {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a song and a dance for the genre {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a track and a dance that fit the genre {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a piece of music and a dance routine in the genre {GENRE_HOLDER}.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance moves: {MOTION_HOLDER}."},

    {'input': f"Generate both music and dance based on the genre {GENRE_HOLDER}.",
     'output': f"The song: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, compose a piece of music and create a dance.",
     'output': f"The music track: {MUSIC_HOLDER}, and the dance routine: {MOTION_HOLDER}."},

    {'input': f"Please create music and a dance sequence for the genre {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a song and generate a dance routine for the genre {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER}, and the dance is: {MOTION_HOLDER}."},

    {'input': f"Generate music and a dance that fit the genre {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a music track and generate a dance for the genre {GENRE_HOLDER}.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance moves: {MOTION_HOLDER}."},

    {'input': f"Create a song and a dance for the genre {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a song and a matching dance for the genre {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, produce a music composition and a dance routine.",
     'output': f"The track: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a music track and a corresponding dance for the genre {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a piece of music and a dance performance for the genre {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a song and a matching dance for the genre {GENRE_HOLDER}.",
     'output': f"The track is: {MUSIC_HOLDER}, and the dance is: {MOTION_HOLDER}."},

    {'input': f"Generate music and a dance routine that fit the genre {GENRE_HOLDER}.",
     'output': f"Here’s the song: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create a piece of music and a dance in the genre {GENRE_HOLDER}.",
     'output': f"The music track: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a song and a dance for the genre {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER}, and the dance moves: {MOTION_HOLDER}."},

    {'input': f"Generate a piece of music and create a dance performance for the genre {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, generate both a song and a dance routine.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Create music and a dance for the genre {GENRE_HOLDER}.",
     'output': f"The track is: {MUSIC_HOLDER}, and the dance is: {MOTION_HOLDER}."},

    {'input': f"Generate a song and a dance performance for the genre {GENRE_HOLDER}.",
     'output': f"Here’s the music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a music track and generate a matching dance for {GENRE_HOLDER}.",
     'output': f"Here’s your music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Please create a music composition and a dance sequence for {GENRE_HOLDER}.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Generate a song and a dance routine to match the genre {GENRE_HOLDER}.",
     'output': f"The music track: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"For the genre {GENRE_HOLDER}, create both music and a dance performance.",
     'output': f"The music: {MUSIC_HOLDER}, and the dance: {MOTION_HOLDER}."},

    {'input': f"Compose a song and generate a matching dance for the genre {GENRE_HOLDER}.",
     'output': f"The music is: {MUSIC_HOLDER}, and the dance is: {MOTION_HOLDER}."}

]
