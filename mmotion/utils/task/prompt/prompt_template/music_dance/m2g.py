from mmotion.utils.task.modality import AUDIO_HOLDER, GENRE_HOLDER

M2G_TEMPLATE = [
    {'input': f"Tell me what's the genre of the music: {AUDIO_HOLDER}",
     'output': f"I think it's {GENRE_HOLDER}."},

    {'input': f"Based on this audio clip: {AUDIO_HOLDER}, what genre does it belong to?",
     'output': f"The genre appears to be {GENRE_HOLDER}."},

    {'input': f"What genre is represented by the following music: {AUDIO_HOLDER}?",
     'output': f"I would classify it as {GENRE_HOLDER}."},

    {'input': f"Identify the genre of this track: {AUDIO_HOLDER}.",
     'output': f"It sounds like {GENRE_HOLDER}."},

    {'input': f"Can you tell me the genre of the following music: {AUDIO_HOLDER}?",
     'output': f"I believe it's {GENRE_HOLDER}."},

    {'input': f"What genre does this audio sample: {AUDIO_HOLDER} fall under?",
     'output': f"This piece is likely {GENRE_HOLDER}."},

    {'input': f"Based on this music: {AUDIO_HOLDER}, can you identify its genre?",
     'output': f"I think it fits into the {GENRE_HOLDER} genre."},

    {'input': f"From this audio: {AUDIO_HOLDER}, what genre would you say it is?",
     'output': f"It's most likely {GENRE_HOLDER}."},

    {'input': f"Listen to this track: {AUDIO_HOLDER} and tell me its genre.",
     'output': f"I would categorize it as {GENRE_HOLDER}."},

    {'input': f"What genre is this music piece: {AUDIO_HOLDER}?",
     'output': f"I classify it as {GENRE_HOLDER}."},

    {'input': f"Can you determine the genre of this audio clip: {AUDIO_HOLDER}?",
     'output': f"The genre seems to be {GENRE_HOLDER}."},

    {'input': f"Identify the genre for this music: {AUDIO_HOLDER}.",
     'output': f"I think it's {GENRE_HOLDER}."},

    {'input': f"Based on this audio, what genre does it represent? {AUDIO_HOLDER}",
     'output': f"It appears to be {GENRE_HOLDER}."},

    {'input': f"What genre does the following music belong to: {AUDIO_HOLDER}?",
     'output': f"I would say it's {GENRE_HOLDER}."},

    {'input': f"Can you tell me the genre of this track: {AUDIO_HOLDER}?",
     'output': f"This music is likely {GENRE_HOLDER}."},

    {'input': f"Identify the genre of this audio sample: {AUDIO_HOLDER}.",
     'output': f"It sounds like {GENRE_HOLDER}."},

    {'input': f"What genre do you think this music represents? {AUDIO_HOLDER}",
     'output': f"I classify it as {GENRE_HOLDER}."},

    {'input': f"Listen to this audio: {AUDIO_HOLDER} and tell me the genre.",
     'output': f"I believe it's {GENRE_HOLDER}."},

    {'input': f"What genre is this music piece: {AUDIO_HOLDER} classified under?",
     'output': f"This sounds like {GENRE_HOLDER}."},

    {'input': f"Based on this track: {AUDIO_HOLDER}, what genre does it belong to?",
     'output': f"It's most likely {GENRE_HOLDER}."},

    {'input': f"From this audio: {AUDIO_HOLDER}, can you identify the genre?",
     'output': f"I think it's {GENRE_HOLDER}."},

    {'input': f"Tell me the genre of the following music: {AUDIO_HOLDER}.",
     'output': f"The genre appears to be {GENRE_HOLDER}."},

    {'input': f"Identify the genre of this music: {AUDIO_HOLDER}.",
     'output': f"It sounds like {GENRE_HOLDER}."},

    {'input': f"What genre is represented by this track: {AUDIO_HOLDER}?",
     'output': f"I classify it as {GENRE_HOLDER}."},

    {'input': f"Can you determine the genre of the audio: {AUDIO_HOLDER}?",
     'output': f"I believe it's {GENRE_HOLDER}."},

    {'input': f"Listen to this piece: {AUDIO_HOLDER} and tell me its genre.",
     'output': f"This piece is likely {GENRE_HOLDER}."},

    {'input': f"What genre does this audio sample: {AUDIO_HOLDER} belong to?",
     'output': f"It's probably {GENRE_HOLDER}."},

    {'input': f"From this audio: {AUDIO_HOLDER}, can you tell me the genre?",
     'output': f"The genre seems to be {GENRE_HOLDER}."},

    {'input': f"Identify the genre of this track: {AUDIO_HOLDER}.",
     'output': f"I think it fits into the {GENRE_HOLDER} genre."},

    {'input': f"Can you tell me the genre of the following music: {AUDIO_HOLDER}?",
     'output': f"I believe it's {GENRE_HOLDER}."},

    {'input': f"What genre does this audio piece belong to? {AUDIO_HOLDER}",
     'output': f"It sounds like {GENRE_HOLDER}."},

    {'input': f"Based on this audio: {AUDIO_HOLDER}, what genre do you think it is?",
     'output': f"I would categorize it as {GENRE_HOLDER}."},

    {'input': f"From the music: {AUDIO_HOLDER}, can you identify its genre?",
     'output': f"I think it's {GENRE_HOLDER}."},

    {'input': f"Listen to this audio clip: {AUDIO_HOLDER} and tell me its genre.",
     'output': f"It appears to be {GENRE_HOLDER}."},

    {'input': f"Can you determine the genre for this music: {AUDIO_HOLDER}?",
     'output': f"The genre seems to be {GENRE_HOLDER}."},

    {'input': f"Generate a genre classification for this audio: {AUDIO_HOLDER}.",
     'output': f"I would classify it as {GENRE_HOLDER}."},

    {'input': f"What genre would you assign to the following music: {AUDIO_HOLDER}?",
     'output': f"I think it fits in the {GENRE_HOLDER} category."},

    {'input': f"Identify the music genre for this track: {AUDIO_HOLDER}.",
     'output': f"I would say it's {GENRE_HOLDER}."},

    {'input': f"From the audio sample: {AUDIO_HOLDER}, what genre can you identify?",
     'output': f"I classify it as {GENRE_HOLDER}."},

    {'input': f"What genre does this music piece: {AUDIO_HOLDER} belong to?",
     'output': f"The genre appears to be {GENRE_HOLDER}."},

    {'input': f"Based on this track: {AUDIO_HOLDER}, can you tell me the genre?",
     'output': f"It's most likely {GENRE_HOLDER}."},

    {'input': f"Listen to this audio: {AUDIO_HOLDER} and tell me what genre it is.",
     'output': f"I believe it's {GENRE_HOLDER}."},

    {'input': f"Can you identify the genre of this track: {AUDIO_HOLDER}?",
     'output': f"The genre seems to be {GENRE_HOLDER}."},

    {'input': f"Generate a genre classification for this audio: {AUDIO_HOLDER}.",
     'output': f"I think it's {GENRE_HOLDER}."},

    {'input': f"From the audio sample: {AUDIO_HOLDER}, can you identify the genre?",
     'output': f"The genre is likely {GENRE_HOLDER}."},

    {'input': f"Identify the genre for this music: {AUDIO_HOLDER}.",
     'output': f"I would classify it as {GENRE_HOLDER}."},

    {'input': f"What genre does this audio represent? {AUDIO_HOLDER}",
     'output': f"It sounds like {GENRE_HOLDER}."},

    {'input': f"Based on the audio clip: {AUDIO_HOLDER}, what genre do you think it is?",
     'output': f"I would say it's {GENRE_HOLDER}."},

    {'input': f"Can you tell me the genre of this music track: {AUDIO_HOLDER}?",
     'output': f"The genre appears to be {GENRE_HOLDER}."},

    {'input': f"What genre is represented by this audio sample: {AUDIO_HOLDER}?",
     'output': f"I think it's {GENRE_HOLDER}."},

    {'input': f"From this track: {AUDIO_HOLDER}, can you identify its genre?",
     'output': f"It seems to be {GENRE_HOLDER}."},

    {'input': f"Generate a genre classification for the following music: {AUDIO_HOLDER}.",
     'output': f"I would classify it as {GENRE_HOLDER}."},

    {'input': f"What genre does this audio clip belong to? {AUDIO_HOLDER}",
     'output': f"The genre appears to be {GENRE_HOLDER}."},

    {'input': f"Can you tell me the genre of this music piece: {AUDIO_HOLDER}?",
     'output': f"I think it's {GENRE_HOLDER}."},

    {'input': f"Based on this audio: {AUDIO_HOLDER}, what genre would you assign?",
     'output': f"It sounds like {GENRE_HOLDER}."},

    {'input': f"Listen to this track: {AUDIO_HOLDER} and tell me its genre.",
     'output': f"I classify it as {GENRE_HOLDER}."},

    {'input': f"Identify the genre of this audio sample: {AUDIO_HOLDER}.",
     'output': f"I would say it's {GENRE_HOLDER}."},

    {'input': f"What genre does this music belong to? {AUDIO_HOLDER}",
     'output': f"It seems to be {GENRE_HOLDER}."},

    {'input': f"Based on this track: {AUDIO_HOLDER}, what genre can you identify?",
     'output': f"The genre is likely {GENRE_HOLDER}."},

    {'input': f"Can you determine the genre of this audio: {AUDIO_HOLDER}?",
     'output': f"I think it's {GENRE_HOLDER}."},

    {'input': f"From this music: {AUDIO_HOLDER}, what genre would you say it is?",
     'output': f"The genre seems to be {GENRE_HOLDER}."},

    {'input': f"Generate a genre classification for the audio sample: {AUDIO_HOLDER}.",
     'output': f"I classify it as {GENRE_HOLDER}."},

    {'input': f"What genre does this music track represent? {AUDIO_HOLDER}",
     'output': f"I would categorize it as {GENRE_HOLDER}."},

    {'input': f"Identify the genre of this track: {AUDIO_HOLDER}.",
     'output': f"It sounds like {GENRE_HOLDER}."},

    {'input': f"Based on the audio clip: {AUDIO_HOLDER}, what genre do you think it is?",
     'output': f"I believe it's {GENRE_HOLDER}."}

]
