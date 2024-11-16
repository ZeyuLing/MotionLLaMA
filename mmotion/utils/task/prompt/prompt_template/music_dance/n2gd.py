from mmotion.utils.task.modality import GENRE_HOLDER, MOTION_HOLDER

N2GD_TEMPLATE = [
    {'input': f"Create a dance for a random genre",
     'output': f"It's a {GENRE_HOLDER}-style dance: {MOTION_HOLDER}"},
    {'input': f"Create a dance for a random genre.",
     'output': f"Here’s an exciting {GENRE_HOLDER} dance just for you: {MOTION_HOLDER}."},

    {'input': f"Generate a dance from a random genre.",
     'output': f"I’ve crafted a wonderful {GENRE_HOLDER} dance sequence for you: {MOTION_HOLDER}."},

    {'input': f"Give me a dance from any random genre.",
     'output': f"This {GENRE_HOLDER} dance is sure to captivate you: {MOTION_HOLDER}."},

    {'input': f"Can you create a random genre dance?",
     'output': f"Absolutely! Here’s a delightful {GENRE_HOLDER} style dance: {MOTION_HOLDER}."},

    {'input': f"Show me a random genre dance performance.",
     'output': f"Here’s a vibrant {GENRE_HOLDER} dance that will surely impress you: {MOTION_HOLDER}."},

    {'input': f"Generate a dance in a random style for me.",
     'output': f"Let me take you on a journey with this {GENRE_HOLDER}-inspired dance: {MOTION_HOLDER}."},

    {'input': f"Give me a random genre dance to enjoy.",
     'output': f"I’ve prepared this energetic {GENRE_HOLDER} dance for you: {MOTION_HOLDER}."},

    {'input': f"Create a dance from any random genre.",
     'output': f"Here’s an artistic {GENRE_HOLDER}-style dance performance: {MOTION_HOLDER}."},

    {'input': f"Generate a dance in a random genre.",
     'output': f"This {GENRE_HOLDER} dance is full of energy and grace: {MOTION_HOLDER}."},

    {'input': f"Make up a dance from a random genre.",
     'output': f"I’ve choreographed a beautiful {GENRE_HOLDER} dance for you: {MOTION_HOLDER}."},

    {'input': f"Can you create a dance performance from a random genre?",
     'output': f"Certainly! This {GENRE_HOLDER} dance will sweep you off your feet: {MOTION_HOLDER}."},

    {'input': f"Show me a dance in any random genre.",
     'output': f"Enjoy this dynamic and expressive {GENRE_HOLDER} dance: {MOTION_HOLDER}."},

    {'input': f"Create a dance routine for a random genre.",
     'output': f"This {GENRE_HOLDER} dance routine is sure to mesmerize you: {MOTION_HOLDER}."},

    {'input': f"Generate a dance in any random style.",
     'output': f"Here’s a {GENRE_HOLDER} dance performance that’s both bold and captivating: {MOTION_HOLDER}."},

    {'input': f"Give me a dance in a randomly selected genre.",
     'output': f"Here’s a lively {GENRE_HOLDER} dance that will leave you amazed: {MOTION_HOLDER}."},

    {'input': f"Can you create a random genre dance for me?",
     'output': f"I’ve crafted an energetic {GENRE_HOLDER} dance just for you: {MOTION_HOLDER}."},

    {'input': f"Create a random genre dance performance.",
     'output': f"Here’s a stunning {GENRE_HOLDER} dance sequence that will impress: {MOTION_HOLDER}."},

    {'input': f"Generate a random style dance for me.",
     'output': f"This {GENRE_HOLDER} dance has a unique rhythm and flow: {MOTION_HOLDER}."},

    {'input': f"Give me a dance from a randomly chosen genre.",
     'output': f"I’ve created a {GENRE_HOLDER} style dance that’s full of life: {MOTION_HOLDER}."},

    {'input': f"Create a random genre and generate a dance for it.",
     'output': f"Here’s a beautiful {GENRE_HOLDER} dance to match your request: {MOTION_HOLDER}."},

    {'input': f"Generate a dance for a random genre.",
     'output': f"This dynamic {GENRE_HOLDER} dance is sure to impress: {MOTION_HOLDER}."},

    {'input': f"Create a dance from any random genre.",
     'output': f"I’ve choreographed a {GENRE_HOLDER} dance that will inspire you: {MOTION_HOLDER}."},

    {'input': f"Give me a random dance performance.",
     'output': f"This dance, inspired by the {GENRE_HOLDER} genre, is a feast for the eyes: {MOTION_HOLDER}."},

    {'input': f"Show me a random genre dance.",
     'output': f"Enjoy this spectacular {GENRE_HOLDER} dance routine: {MOTION_HOLDER}."},

    {'input': f"Generate a dance from a random style.",
     'output': f"I’ve created a lively {GENRE_HOLDER} style dance for you: {MOTION_HOLDER}."},

    {'input': f"Create a random genre and make a dance for it.",
     'output': f"Here’s an expressive {GENRE_HOLDER} dance that will take your breath away: {MOTION_HOLDER}."},

    {'input': f"Give me a dance in a random style.",
     'output': f"This dance in the {GENRE_HOLDER} style will leave you in awe: {MOTION_HOLDER}."},

    {'input': f"Can you create a random genre dance routine?",
     'output': f"I’ve crafted a wonderful {GENRE_HOLDER} dance routine for you: {MOTION_HOLDER}."},

    {'input': f"Generate a dance in a random genre for me.",
     'output': f"This {GENRE_HOLDER} dance is sure to dazzle you: {MOTION_HOLDER}."},

    {'input': f"Show me a dance performance from a random genre.",
     'output': f"Here’s a lively {GENRE_HOLDER} style performance: {MOTION_HOLDER}."},

    {'input': f"Create a random dance routine for me.",
     'output': f"This exciting {GENRE_HOLDER} dance routine will keep you captivated: {MOTION_HOLDER}."},

    {'input': f"Can you create a dance in a randomly selected genre?",
     'output': f"I’ve crafted a vibrant {GENRE_HOLDER} dance: {MOTION_HOLDER}."},

    {'input': f"Give me a random dance and genre.",
     'output': f"Here’s a bold and lively {GENRE_HOLDER} dance: {MOTION_HOLDER}."},

    {'input': f"Generate a dance for a random style.",
     'output': f"Here’s a beautiful {GENRE_HOLDER} dance performance that will leave you amazed: {MOTION_HOLDER}."},

    {'input': f"Create a dance from a random genre for me to watch.",
     'output': f"This {GENRE_HOLDER} dance is full of energy and passion: {MOTION_HOLDER}."},

    {'input': f"Give me a randomly chosen genre dance.",
     'output': f"Here’s a dynamic {GENRE_HOLDER} dance performance: {MOTION_HOLDER}."},

    {'input': f"Show me a random style dance routine.",
     'output': f"This dance, inspired by {GENRE_HOLDER}, will blow you away: {MOTION_HOLDER}."},

    {'input': f"Generate a dance for a random genre.",
     'output': f"I’ve prepared a captivating {GENRE_HOLDER} dance for you: {MOTION_HOLDER}."},

    {'input': f"Create a dance from any random genre.",
     'output': f"Here’s a visually stunning {GENRE_HOLDER} dance routine: {MOTION_HOLDER}."},

    {'input': f"Give me a random genre dance to watch.",
     'output': f"This {GENRE_HOLDER} dance is packed with style and grace: {MOTION_HOLDER}."},

    {'input': f"Can you create a random genre dance performance?",
     'output': f"Here’s a unique and elegant {GENRE_HOLDER} dance performance: {MOTION_HOLDER}."},

    {'input': f"Show me a random genre dance.",
     'output': f"Enjoy this exquisite {GENRE_HOLDER} dance performance: {MOTION_HOLDER}."},

    {'input': f"Generate a random style dance for me.",
     'output': f"This {GENRE_HOLDER} dance will captivate your attention: {MOTION_HOLDER}."},

    {'input': f"Create a random genre and generate a dance.",
     'output': f"I’ve choreographed a creative {GENRE_HOLDER} dance for you: {MOTION_HOLDER}."},

    {'input': f"Give me a random dance and genre to enjoy.",
     'output': f"This {GENRE_HOLDER} dance is full of artistic flair: {MOTION_HOLDER}."},

    {'input': f"Generate a random dance style for me.",
     'output': f"This lively {GENRE_HOLDER} dance is full of movement and excitement: {MOTION_HOLDER}."},

    {'input': f"Make up a dance in a random genre.",
     'output': f"I’ve crafted this {GENRE_HOLDER} dance to delight your senses: {MOTION_HOLDER}."},

    {'input': f"Can you create a dance for a random genre?",
     'output': f"Here’s an exciting {GENRE_HOLDER} dance just for you: {MOTION_HOLDER}."},

    {'input': f"Show me a random genre dance performance.",
     'output': f"This {GENRE_HOLDER} dance is sure to keep you entertained: {MOTION_HOLDER}."},

    {'input': f"Generate a dance for any random genre.",
     'output': f"I’ve created a dazzling {GENRE_HOLDER} dance performance for you: {MOTION_HOLDER}."},

    {'input': f"Create a random genre and make a dance for it.",
     'output': f"This artistic {GENRE_HOLDER} dance is sure to impress: {MOTION_HOLDER}."},

    {'input': f"Give me a random genre dance routine to watch.",
     'output': f"Enjoy this fun and vibrant {GENRE_HOLDER} dance: {MOTION_HOLDER}."},

    {'input': f"Create a random dance routine for me.",
     'output': f"This {GENRE_HOLDER} dance is packed with creativity and energy: {MOTION_HOLDER}."},

    {'input': f"Generate a random style dance performance.",
     'output': f"I’ve choreographed this energetic {GENRE_HOLDER} dance: {MOTION_HOLDER}."},

    {'input': f"Give me a random genre and create a dance.",
     'output': f"Here’s a delightful {GENRE_HOLDER} dance routine for you: {MOTION_HOLDER}."},

    {'input': f"Can you create a dance in a randomly selected genre?",
     'output': f"Here’s a lively {GENRE_HOLDER} dance performance: {MOTION_HOLDER}."},

    {'input': f"Show me a dance from any random genre.",
     'output': f"This {GENRE_HOLDER} dance is a blend of elegance and excitement: {MOTION_HOLDER}."},

    {'input': f"Generate a random genre dance routine.",
     'output': f"I’ve created a stunning {GENRE_HOLDER} dance performance: {MOTION_HOLDER}."},

    {'input': f"Create a dance from a random genre for me to enjoy.",
     'output': f"This {GENRE_HOLDER} dance is sure to leave you amazed: {MOTION_HOLDER}."},

    {'input': f"Give me a randomly chosen genre dance performance.",
     'output': f"Here’s a mesmerizing {GENRE_HOLDER} dance routine: {MOTION_HOLDER}."},

    {'input': f"Generate a dance performance from a random genre.",
     'output': f"This {GENRE_HOLDER} dance will sweep you off your feet: {MOTION_HOLDER}."},

    {'input': f"Create a random dance and genre for me.",
     'output': f"I’ve prepared a creative {GENRE_HOLDER} dance performance: {MOTION_HOLDER}."},

    {'input': f"Show me a random dance style.",
     'output': f"This {GENRE_HOLDER} dance is full of bold moves and energy: {MOTION_HOLDER}."},

    {'input': f"Generate a random style and choreograph a dance.",
     'output': f"Here’s a vibrant {GENRE_HOLDER} dance performance for you: {MOTION_HOLDER}."},

    {'input': f"Create a random genre and generate a dance for it.",
     'output': f"I’ve choreographed a lively {GENRE_HOLDER} dance for your enjoyment: {MOTION_HOLDER}."},

    {'input': f"Give me a random dance routine from any genre.",
     'output': f"This {GENRE_HOLDER} dance is both graceful and exhilarating: {MOTION_HOLDER}."},

    {'input': f"Generate a random dance from any genre.",
     'output': f"This {GENRE_HOLDER} dance routine will keep you on your toes: {MOTION_HOLDER}."},

    {'input': f"Show me a dance routine from a random genre.",
     'output': f"Here’s an energetic {GENRE_HOLDER} dance performance: {MOTION_HOLDER}."},

    {'input': f"Create a random genre and make a dance.",
     'output': f"This {GENRE_HOLDER} dance is full of rhythm and creativity: {MOTION_HOLDER}."},

    {'input': f"Give me a random genre dance to watch.",
     'output': f"Enjoy this spectacular {GENRE_HOLDER} dance routine: {MOTION_HOLDER}."},

    {'input': f"Can you create a random genre dance for me?",
     'output': f"This {GENRE_HOLDER} dance is full of excitement and energy: {MOTION_HOLDER}."},

    {'input': f"Show me a random dance performance.",
     'output': f"Here’s a delightful {GENRE_HOLDER} dance that you will love: {MOTION_HOLDER}."},

    {'input': f"Generate a random style dance performance for me.",
     'output': f"This vibrant {GENRE_HOLDER} dance performance will keep you engaged: {MOTION_HOLDER}."},

    {'input': f"Create a random genre dance for me to enjoy.",
     'output': f"This {GENRE_HOLDER} dance will captivate you with its rhythm and energy: {MOTION_HOLDER}."},

    {'input': f"Give me a random genre and create a dance for it.",
     'output': f"Here’s a dynamic {GENRE_HOLDER} dance just for you: {MOTION_HOLDER}."},

    {'input': f"Show me a random style dance.",
     'output': f"This {GENRE_HOLDER} dance is full of grace and passion: {MOTION_HOLDER}."},

    {'input': f"Generate a random genre and choreograph a dance.",
     'output': f"This {GENRE_HOLDER} dance performance is sure to impress: {MOTION_HOLDER}."},

    {'input': f"Create a dance from any random genre.",
     'output': f"Here’s an amazing {GENRE_HOLDER} dance performance: {MOTION_HOLDER}."}

]
