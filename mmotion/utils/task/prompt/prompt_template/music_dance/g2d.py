from mmotion.utils.task.modality import GENRE_HOLDER, MOTION_HOLDER

G2D_TEMPLATE = [
    {'input': f"Dance {GENRE_HOLDER} for me",
     'output': f"Okay: {MOTION_HOLDER}"},

    {'input': f"Can you perform a {GENRE_HOLDER} dance for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a dance in the {GENRE_HOLDER} style.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Perform a {GENRE_HOLDER} dance.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a dance in the {GENRE_HOLDER} genre.",
     'output': f"Here you go: {MOTION_HOLDER}"},

    {'input': f"Do a dance in the style of {GENRE_HOLDER}.",
     'output': f"Here’s your {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Please perform a {GENRE_HOLDER} dance.",
     'output': f"Okay, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you show me a {GENRE_HOLDER} dance?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a dance performance in {GENRE_HOLDER}.",
     'output': f"Here’s your dance: {MOTION_HOLDER}"},

    {'input': f"Do a {GENRE_HOLDER} dance for me.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you dance {GENRE_HOLDER}?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a dance from the {GENRE_HOLDER} genre.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Perform a {GENRE_HOLDER} dance for me.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a {GENRE_HOLDER} style dance.",
     'output': f"Here you go: {MOTION_HOLDER}"},

    {'input': f"Do a {GENRE_HOLDER} dance performance.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Please show me a {GENRE_HOLDER} dance.",
     'output': f"Okay, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you perform a dance from the {GENRE_HOLDER} genre?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a dance that fits the {GENRE_HOLDER} style.",
     'output': f"Here’s your dance: {MOTION_HOLDER}"},

    {'input': f"Do a {GENRE_HOLDER} style dance for me.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you do a {GENRE_HOLDER} dance for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a performance in the {GENRE_HOLDER} dance style.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Perform a dance in the {GENRE_HOLDER} style.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a {GENRE_HOLDER} genre dance.",
     'output': f"Here you go: {MOTION_HOLDER}"},

    {'input': f"Do a performance in the style of {GENRE_HOLDER}.",
     'output': f"Here’s your {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Please show me a {GENRE_HOLDER} genre dance.",
     'output': f"Okay, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you perform a {GENRE_HOLDER} style dance for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a dance from the {GENRE_HOLDER} genre.",
     'output': f"Here’s your dance: {MOTION_HOLDER}"},

    {'input': f"Do a performance in the {GENRE_HOLDER} dance genre.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you do a {GENRE_HOLDER} dance performance for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a {GENRE_HOLDER} style dance performance.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Perform a {GENRE_HOLDER} genre dance.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a {GENRE_HOLDER} genre dance performance.",
     'output': f"Here you go: {MOTION_HOLDER}"},

    {'input': f"Do a dance in the {GENRE_HOLDER} style for me.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Please show me a dance in the {GENRE_HOLDER} genre.",
     'output': f"Okay, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you perform a {GENRE_HOLDER} genre dance for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a performance of a {GENRE_HOLDER} dance.",
     'output': f"Here’s your dance: {MOTION_HOLDER}"},

    {'input': f"Do a dance performance in the {GENRE_HOLDER} style.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you dance in the {GENRE_HOLDER} genre for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a dance performance from the {GENRE_HOLDER} genre.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Perform a dance in the style of {GENRE_HOLDER}.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a dance from the {GENRE_HOLDER} genre.",
     'output': f"Here you go: {MOTION_HOLDER}"},

    {'input': f"Do a performance of a {GENRE_HOLDER} dance.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Please perform a dance in the {GENRE_HOLDER} style.",
     'output': f"Okay, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you show me a performance in the {GENRE_HOLDER} genre?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a {GENRE_HOLDER} genre dance performance.",
     'output': f"Here’s your dance: {MOTION_HOLDER}"},

    {'input': f"Do a {GENRE_HOLDER} genre dance for me.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you do a performance in the {GENRE_HOLDER} dance style?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a {GENRE_HOLDER} style dance.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Perform a dance in the {GENRE_HOLDER} genre.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a performance of a {GENRE_HOLDER} genre dance.",
     'output': f"Here you go: {MOTION_HOLDER}"},

    {'input': f"Do a performance in the {GENRE_HOLDER} genre for me.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you perform a dance in the {GENRE_HOLDER} style?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a {GENRE_HOLDER} genre dance.",
     'output': f"Here’s your dance: {MOTION_HOLDER}"},

    {'input': f"Do a {GENRE_HOLDER} genre dance performance.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Please perform a {GENRE_HOLDER} dance for me.",
     'output': f"Okay, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you perform a {GENRE_HOLDER} style dance for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a {GENRE_HOLDER} dance performance.",
     'output': f"Here’s your dance: {MOTION_HOLDER}"},

    {'input': f"Do a {GENRE_HOLDER} dance style performance.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you dance a {GENRE_HOLDER} dance for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a dance performance in the {GENRE_HOLDER} genre.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Perform a {GENRE_HOLDER} dance style for me.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a performance of a {GENRE_HOLDER} style dance.",
     'output': f"Here you go: {MOTION_HOLDER}"},

    {'input': f"Do a performance in the {GENRE_HOLDER} style for me.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Please perform a dance in the {GENRE_HOLDER} genre.",
     'output': f"Okay, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you show me a {GENRE_HOLDER} style dance performance?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a {GENRE_HOLDER} style dance performance.",
     'output': f"Here’s your dance: {MOTION_HOLDER}"},

    {'input': f"Do a dance performance in the {GENRE_HOLDER} genre.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you perform a {GENRE_HOLDER} genre dance?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a dance in the {GENRE_HOLDER} genre.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Perform a {GENRE_HOLDER} style dance.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Give me a performance of a {GENRE_HOLDER} dance.",
     'output': f"Here you go: {MOTION_HOLDER}"},

    {'input': f"Do a {GENRE_HOLDER} dance for me.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"},

    {'input': f"Can you do a {GENRE_HOLDER} dance for me?",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Show me a performance in the {GENRE_HOLDER} genre.",
     'output': f"Here’s the {GENRE_HOLDER} dance: {MOTION_HOLDER}"}
]
