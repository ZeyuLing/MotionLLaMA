from mmotion.utils.task.modality import MOTION_HOLDER

N2D_TEMPLATE = [
    {'input': f"Randomly generate a dance sequence.",
     'output': f"Okay. {MOTION_HOLDER}"},
    {'input': f"No music, just dance freely!",
     'output': f"Alright, please watch as I dance for you: {MOTION_HOLDER}"},
    {'input': f"Generate a random dance sequence for me.",
     'output': f"Sure! Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Please create a random dance routine.",
     'output': f"Okay, here’s the dance sequence: {MOTION_HOLDER}"},

    {'input': f"Come up with a random dance for me.",
     'output': f"Alright, here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Can you randomly create a dance?",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Make up a random dance sequence.",
     'output': f"Here’s the dance I came up with: {MOTION_HOLDER}"},

    {'input': f"Please randomly generate a dance routine.",
     'output': f"Alright, here’s the dance I created: {MOTION_HOLDER}"},

    {'input': f"Generate a dance randomly for me.",
     'output': f"Okay, here’s a random dance: {MOTION_HOLDER}"},

    {'input': f"Can you make a random dance for me?",
     'output': f"Sure, here’s a random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance sequence.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Please create a random dance, any style.",
     'output': f"Here’s the dance I generated: {MOTION_HOLDER}"},

    {'input': f"Generate a dance sequence, completely random.",
     'output': f"Here’s a random dance for you: {MOTION_HOLDER}"},

    {'input': f"Come up with a dance at random.",
     'output': f"Here’s the random dance routine: {MOTION_HOLDER}"},

    {'input': f"Can you create a random set of dance moves?",
     'output': f"Here’s the dance sequence I made: {MOTION_HOLDER}"},

    {'input': f"Please generate a random dance routine.",
     'output': f"Here’s a dance sequence I created: {MOTION_HOLDER}"},

    {'input': f"Randomly create a dance for me.",
     'output': f"Here’s the random dance I came up with: {MOTION_HOLDER}"},

    {'input': f"Make a random dance sequence.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance, any kind.",
     'output': f"Here’s the random dance routine: {MOTION_HOLDER}"},

    {'input': f"Please come up with a random dance for me.",
     'output': f"Here’s what I created: {MOTION_HOLDER}"},

    {'input': f"Generate a random dance set of moves.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Can you produce a random dance routine for me?",
     'output': f"Sure, here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Create a dance sequence, completely random.",
     'output': f"Here’s a dance sequence I made: {MOTION_HOLDER}"},

    {'input': f"Make up a random dance, please.",
     'output': f"Here’s the random dance I came up with: {MOTION_HOLDER}"},

    {'input': f"Please generate a random set of dance moves.",
     'output': f"Here’s the random dance routine: {MOTION_HOLDER}"},

    {'input': f"Come up with a random dance and show me.",
     'output': f"Here’s the random dance I generated: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance, with no music.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Can you make a dance randomly, without music?",
     'output': f"Here’s the dance sequence I created: {MOTION_HOLDER}"},

    {'input': f"Randomly create a dance, no music involved.",
     'output': f"Here’s the random dance I came up with: {MOTION_HOLDER}"},

    {'input': f"Make a random dance, no background music.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a random dance with no music.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Please create a random dance, no music required.",
     'output': f"Here’s the dance routine: {MOTION_HOLDER}"},

    {'input': f"Come up with a random dance and show it to me.",
     'output': f"Here’s the random dance I created: {MOTION_HOLDER}"},

    {'input': f"Generate a random dance performance.",
     'output': f"Here’s the random dance I made: {MOTION_HOLDER}"},

    {'input': f"Make up a random dance without music.",
     'output': f"Here’s the random dance routine: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance, without background music.",
     'output': f"Here’s the dance I created: {MOTION_HOLDER}"},

    {'input': f"Create a random dance, no music in the background.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Can you come up with a random dance routine, without music?",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Randomly make a dance performance, no music needed.",
     'output': f"Here’s the random dance routine: {MOTION_HOLDER}"},

    {'input': f"Generate a random set of dance moves for me.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance set of moves.",
     'output': f"Here’s the dance routine: {MOTION_HOLDER}"},

    {'input': f"Make up a random dance, without background music.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Please generate a random dance routine with no music.",
     'output': f"Here’s the random dance I made: {MOTION_HOLDER}"},

    {'input': f"Create a random dance sequence for me, with no music involved.",
     'output': f"Here’s the dance sequence I came up with: {MOTION_HOLDER}"},

    {'input': f"Come up with a random dance and perform it for me.",
     'output': f"Here’s the random dance I generated: {MOTION_HOLDER}"},

    {'input': f"Generate a random dance performance, no music required.",
     'output': f"Here’s the random dance I created: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance, no music included.",
     'output': f"Here’s the dance sequence: {MOTION_HOLDER}"},

    {'input': f"Please create a random set of dance moves.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Generate a random set of dance steps for me.",
     'output': f"Here’s the dance routine: {MOTION_HOLDER}"},

    {'input': f"Come up with a random dance set of moves.",
     'output': f"Here’s the random dance I made: {MOTION_HOLDER}"},

    {'input': f"Please make up a random dance, with no music.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a random set of dance moves, without music.",
     'output': f"Here’s the random dance routine: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance, no music background.",
     'output': f"Here’s the dance I came up with: {MOTION_HOLDER}"},

    {'input': f"Make a random dance, no music in the background.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Generate a random dance, without a music track.",
     'output': f"Here’s the dance I created: {MOTION_HOLDER}"},

    {'input': f"Please create a random dance set of moves.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance routine for me.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Make a random dance set of moves, no music required.",
     'output': f"Here’s the dance sequence I made: {MOTION_HOLDER}"},

    {'input': f"Create a random dance sequence, but without music.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Can you come up with a random set of dance moves, no music?",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Make up a random dance, no background music needed.",
     'output': f"Here’s the dance I created: {MOTION_HOLDER}"},

    {'input': f"Generate a random set of dance steps without music.",
     'output': f"Here’s the random dance I made: {MOTION_HOLDER}"},

    {'input': f"Create a random dance routine, with no music.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance, with no music required.",
     'output': f"Here’s the dance sequence I generated: {MOTION_HOLDER}"},

    {'input': f"Make a random set of dance steps, with no background music.",
     'output': f"Here’s the random dance I made: {MOTION_HOLDER}"},

    {'input': f"Come up with a random dance routine, no music necessary.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Generate a random dance sequence with no music.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Please create a random set of dance moves, no music involved.",
     'output': f"Here’s the random dance I made: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance routine for me, without music.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Make up a random dance with no background music.",
     'output': f"Here’s the dance I generated: {MOTION_HOLDER}"},

    {'input': f"Can you generate a random dance for me, no music required?",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Randomly create a set of dance moves, with no music.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Make a random dance routine, no music needed.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Produce a random dance, without any music involved.",
     'output': f"Here’s the random dance routine: {MOTION_HOLDER}"},

    {'input': f"Create a random set of dance steps, with no music.",
     'output': f"Here’s the dance I created: {MOTION_HOLDER}"},

    {'input': f"Generate a random dance sequence for me, no music necessary.",
     'output': f"Here’s the random dance: {MOTION_HOLDER}"},

    {'input': f"Make up a random dance set of moves, without music.",
     'output': f"Here’s the dance: {MOTION_HOLDER}"},

    {'input': f"Please come up with a random set of dance moves, no music involved.",
     'output': f"Here’s the random dance sequence: {MOTION_HOLDER}"},

    {'input': f"Can you produce a random dance routine, without background music?",
     'output': f"Here’s the random dance I made: {MOTION_HOLDER}"},
]
