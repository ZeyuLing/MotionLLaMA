from mmotion.utils.task.modality import GENRE_HOLDER, MOTION_HOLDER

D2G_TEMPLATE = [
    {'input': f"Watch the dance and tell me what's the genre of it: {MOTION_HOLDER}",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Observe this dance: {MOTION_HOLDER}, can you identify the genre?",
     'output': f"It looks like it's {GENRE_HOLDER}"},

    {'input': f"Watch the following dance: {MOTION_HOLDER}, what genre does it belong to?",
     'output': f"This dance seems to be {GENRE_HOLDER}"},

    {'input': f"Based on this dance: {MOTION_HOLDER}, what genre would you say it is?",
     'output': f"I would classify it as {GENRE_HOLDER}"},

    {'input': f"Look at this dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"It seems like a {GENRE_HOLDER} style dance"},

    {'input': f"Can you determine the genre of this dance: {MOTION_HOLDER}?",
     'output': f"My guess is that it's {GENRE_HOLDER}"},

    {'input': f"Observe the dance: {MOTION_HOLDER}, what genre do you think it is?",
     'output': f"I would say it's {GENRE_HOLDER}"},

    {'input': f"Look at this dance performance: {MOTION_HOLDER}, what's the genre?",
     'output': f"This seems to be {GENRE_HOLDER}"},

    {'input': f"From watching this dance: {MOTION_HOLDER}, what genre would you classify it as?",
     'output': f"I believe it's {GENRE_HOLDER}"},

    {'input': f"Watch this dance and tell me the genre: {MOTION_HOLDER}.",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Can you watch the dance and identify the genre: {MOTION_HOLDER}?",
     'output': f"It appears to be {GENRE_HOLDER}"},

    {'input': f"Look at this dance performance: {MOTION_HOLDER}, can you classify the genre?",
     'output': f"I'm pretty sure it's {GENRE_HOLDER}"},

    {'input': f"Observe the following dance: {MOTION_HOLDER}, what genre does it fit into?",
     'output': f"I would say it's {GENRE_HOLDER}"},

    {'input': f"Watch this dance sequence: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"I believe this is {GENRE_HOLDER}"},

    {'input': f"Watch this dance performance: {MOTION_HOLDER}, what genre do you think it is?",
     'output': f"My guess is {GENRE_HOLDER}"},

    {'input': f"Look at this movement: {MOTION_HOLDER}, can you tell me the genre?",
     'output': f"I think this is {GENRE_HOLDER}"},

    {'input': f"Can you identify the genre based on this dance: {MOTION_HOLDER}?",
     'output': f"It looks like a {GENRE_HOLDER} dance"},

    {'input': f"Observe this dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"This seems to be {GENRE_HOLDER}"},

    {'input': f"Watch the following movement: {MOTION_HOLDER}, what's the genre?",
     'output': f"I would classify it as {GENRE_HOLDER}"},

    {'input': f"Can you identify the genre from watching this dance: {MOTION_HOLDER}?",
     'output': f"I believe it's {GENRE_HOLDER}"},

    {'input': f"Look at this dance sequence: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"This is probably {GENRE_HOLDER}"},

    {'input': f"Watch the dance: {MOTION_HOLDER}, and tell me what genre it is.",
     'output': f"My guess is {GENRE_HOLDER}"},

    {'input': f"Observe this movement: {MOTION_HOLDER}, can you identify the genre?",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Based on this dance: {MOTION_HOLDER}, what genre would you say it belongs to?",
     'output': f"I would classify it as {GENRE_HOLDER}"},

    {'input': f"Watch the dance and tell me the genre: {MOTION_HOLDER}.",
     'output': f"It looks like {GENRE_HOLDER}"},

    {'input': f"Can you observe this dance: {MOTION_HOLDER}, and determine the genre?",
     'output': f"My guess is {GENRE_HOLDER}"},

    {'input': f"Look at this performance: {MOTION_HOLDER}, and tell me what genre it is.",
     'output': f"This dance seems to fit the {GENRE_HOLDER} genre"},

    {'input': f"Watch the following movement: {MOTION_HOLDER}, what genre do you think it is?",
     'output': f"I would say it's {GENRE_HOLDER}"},

    {'input': f"Observe this dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"It appears to be {GENRE_HOLDER}"},

    {'input': f"Look at the dance: {MOTION_HOLDER}, and tell me what genre it is.",
     'output': f"I think this is {GENRE_HOLDER}"},

    {'input': f"Watch this movement and identify the genre: {MOTION_HOLDER}.",
     'output': f"My guess is it's {GENRE_HOLDER}"},

    {'input': f"Can you identify the genre of this dance: {MOTION_HOLDER}?",
     'output': f"It seems like {GENRE_HOLDER}"},

    {'input': f"Based on this dance: {MOTION_HOLDER}, can you tell me the genre?",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Look at this movement: {MOTION_HOLDER}, what genre do you think it is?",
     'output': f"I would classify it as {GENRE_HOLDER}"},

    {'input': f"Observe this dance sequence: {MOTION_HOLDER}, and tell me what genre it belongs to.",
     'output': f"My guess is it's {GENRE_HOLDER}"},

    {'input': f"Watch the following dance performance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"I believe it's {GENRE_HOLDER}"},

    {'input': f"Watch this dance and classify its genre: {MOTION_HOLDER}.",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Can you observe the dance: {MOTION_HOLDER}, and determine the genre?",
     'output': f"My guess is it's {GENRE_HOLDER}"},

    {'input': f"Based on this dance: {MOTION_HOLDER}, what genre would you say it fits into?",
     'output': f"It seems like {GENRE_HOLDER}"},

    {'input': f"Look at this dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"I believe it's {GENRE_HOLDER}"},

    {'input': f"Watch the following movement: {MOTION_HOLDER}, and identify its genre.",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Observe this performance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"My guess is it's {GENRE_HOLDER}"},

    {'input': f"Watch the following dance and identify the genre: {MOTION_HOLDER}.",
     'output': f"It seems to be {GENRE_HOLDER}"},

    {'input': f"Look at this dance performance: {MOTION_HOLDER}, can you classify the genre?",
     'output': f"I'm pretty sure it's {GENRE_HOLDER}"},

    {'input': f"Watch this movement: {MOTION_HOLDER}, and tell me what genre it is.",
     'output': f"This looks like a {GENRE_HOLDER} dance"},

    {'input': f"Observe this dance and identify the genre: {MOTION_HOLDER}.",
     'output': f"I would say it's {GENRE_HOLDER}"},

    {'input': f"Can you identify the genre of this movement: {MOTION_HOLDER}?",
     'output': f"My guess is {GENRE_HOLDER}"},

    {'input': f"Look at this dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"It seems like {GENRE_HOLDER}"},

    {'input': f"Watch the following dance: {MOTION_HOLDER}, what genre does it belong to?",
     'output': f"I'm pretty sure it's {GENRE_HOLDER}"},

    {'input': f"Can you observe this dance and identify the genre: {MOTION_HOLDER}?",
     'output': f"My guess is it's {GENRE_HOLDER}"},

    {'input': f"Watch this movement: {MOTION_HOLDER}, and tell me what genre it is.",
     'output': f"I would classify it as {GENRE_HOLDER}"},

    {'input': f"Look at this dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Observe this performance: {MOTION_HOLDER}, and classify the genre.",
     'output': f"My guess is {GENRE_HOLDER}"},

    {'input': f"Watch this dance and tell me the genre: {MOTION_HOLDER}.",
     'output': f"It appears to be {GENRE_HOLDER}"},

    {'input': f"Based on this dance: {MOTION_HOLDER}, can you identify the genre?",
     'output': f"It looks like {GENRE_HOLDER}"},

    {'input': f"Look at this dance performance: {MOTION_HOLDER}, and tell me what genre it belongs to.",
     'output': f"I believe this is {GENRE_HOLDER}"},

    {'input': f"Can you watch this dance: {MOTION_HOLDER}, and identify the genre?",
     'output': f"My guess is {GENRE_HOLDER}"},

    {'input': f"Observe the movement: {MOTION_HOLDER}, and classify the genre.",
     'output': f"I would say it's {GENRE_HOLDER}"},

    {'input': f"Watch the following dance: {MOTION_HOLDER}, what genre do you think it is?",
     'output': f"I'm pretty sure it's {GENRE_HOLDER}"},

    {'input': f"Can you identify the genre of this dance: {MOTION_HOLDER}?",
     'output': f"It seems to be {GENRE_HOLDER}"},

    {'input': f"Look at this dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Watch the following performance: {MOTION_HOLDER}, what genre does it fit into?",
     'output': f"I believe this is {GENRE_HOLDER}"},

    {'input': f"Observe this dance: {MOTION_HOLDER}, and classify the genre.",
     'output': f"It seems like {GENRE_HOLDER}"},

    {'input': f"Watch the dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"My guess is it's {GENRE_HOLDER}"},

    {'input': f"Look at this movement: {MOTION_HOLDER}, and identify the genre.",
     'output': f"I'm pretty sure it's {GENRE_HOLDER}"},

    {'input': f"Watch this performance: {MOTION_HOLDER}, what genre would you classify it as?",
     'output': f"I believe it's {GENRE_HOLDER}"},

    {'input': f"Can you watch this dance: {MOTION_HOLDER}, and determine the genre?",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Observe the dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"I'm pretty sure it's {GENRE_HOLDER}"},

    {'input': f"Watch this movement: {MOTION_HOLDER}, and tell me what genre it is.",
     'output': f"It seems like {GENRE_HOLDER}"},

    {'input': f"Can you observe this dance: {MOTION_HOLDER}, and tell me the genre?",
     'output': f"I believe it's {GENRE_HOLDER}"},

    {'input': f"Look at this dance performance: {MOTION_HOLDER}, and classify the genre.",
     'output': f"My guess is {GENRE_HOLDER}"},

    {'input': f"Watch this dance: {MOTION_HOLDER}, and tell me what genre it fits into.",
     'output': f"It seems to be {GENRE_HOLDER}"},

    {'input': f"Can you watch this movement: {MOTION_HOLDER}, and identify the genre?",
     'output': f"My guess is it's {GENRE_HOLDER}"},

    {'input': f"Look at the dance: {MOTION_HOLDER}, and tell me the genre.",
     'output': f"I think it's {GENRE_HOLDER}"},

    {'input': f"Observe this movement: {MOTION_HOLDER}, and classify the genre.",
     'output': f"It appears to be {GENRE_HOLDER}"},

    {'input': f"Watch the following dance: {MOTION_HOLDER}, what genre would you classify it as?",
     'output': f"I believe it's {GENRE_HOLDER}"},

    {'input': f"Look at this dance: {MOTION_HOLDER}, can you tell me the genre?",
     'output': f"I'm pretty sure it's {GENRE_HOLDER}"},

    {'input': f"Can you identify the genre of this movement: {MOTION_HOLDER}?",
     'output': f"My guess is it's {GENRE_HOLDER}"},

    {'input': f"Observe the dance: {MOTION_HOLDER}, and tell me what genre it is.",
     'output': f"I think this is {GENRE_HOLDER}"},

]
