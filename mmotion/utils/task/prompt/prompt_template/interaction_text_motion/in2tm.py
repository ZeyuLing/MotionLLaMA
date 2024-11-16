from mmotion.utils.task.modality import INTERACTOR_MOTION_HOLDER, MOTION_HOLDER, UNION_CAPTION_HOLDER

IN2TM_TEMPLATE = [
    {'input': f"Randomly generate interaction motion together with its caption.",
     'output': f"Okay, here is the interaction motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, and the caption is : {UNION_CAPTION_HOLDER}"},
    {'input': f"I want an interaction motion of 2 persons and its caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}"},
    {'input': f"Generate a random interaction motion with its caption.",
     'output': f"Alright, here is the motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, and the caption is: {UNION_CAPTION_HOLDER}"},

    {'input': f"Please create an interaction motion sequence along with a caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Produce a random interaction motion of two persons and give me a caption.",
     'output': f"The interaction is: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, and here’s the caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Can you randomly generate an interaction between two people and provide a caption?",
     'output': f"{UNION_CAPTION_HOLDER}: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Randomly create an interaction motion with its corresponding caption.",
     'output': f"Here's the motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate a motion of two interacting people with a caption.",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Please give me an interaction motion and a caption for it.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a random interaction between two people and include a caption.",
     'output': f"Okay, here's the interaction motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Can you generate an interaction motion and a caption together?",
     'output': f"Interaction motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, and the caption is: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate two persons interacting and provide a caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"I want a captioned interaction motion between two people.",
     'output': f"Motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Produce a random sequence of interaction with a caption.",
     'output': f"Here it is: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate an interaction motion along with a description caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Please create a captioned interaction motion of two individuals.",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate an interactive motion and include a caption for it.",
     'output': f"Here’s the motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a motion of two people interacting and provide a caption.",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER} - {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate a captioned interaction between two people.",
     'output': f"Interaction: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Produce a sequence of motion with its caption for two people.",
     'output': f"{UNION_CAPTION_HOLDER} - Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you randomly generate a motion sequence for two persons and its caption?",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, and here’s the caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly create an interaction with its corresponding caption.",
     'output': f"Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate a motion of two interacting individuals with a description.",
     'output': f"Motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Please provide an interaction motion along with a caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a random interactive motion and include its caption.",
     'output': f"Motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Can you generate interaction movements and provide a caption for them?",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, and here’s the caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate movements of two people interacting and their caption.",
     'output': f"Here’s the interaction: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate a sequence of two interacting people and include a caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"I need an interaction motion and a caption describing it.",
     'output': f"Interaction: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Produce a random interactive motion and its caption.",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER} - {UNION_CAPTION_HOLDER}"},

    {'input': f"Create an interaction motion between two people with a caption.",
     'output': f"Alright, motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly create an interaction movement along with its caption.",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate interaction movements for two people and caption it.",
     'output': f"Interaction: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Can you provide a motion of interaction and its description?",
     'output': f"{UNION_CAPTION_HOLDER} - {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a captioned motion sequence of two people interacting.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, and the caption is: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate a motion interaction and give a caption for it.",
     'output': f"Here’s the motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a two-person interaction motion with a caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate a captioned motion of two people interacting.",
     'output': f"Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Can you provide an interaction motion together with its caption?",
     'output': f"Motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Please create a random interaction motion and give me its caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a motion and its caption for two people interacting.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate interaction movements and a caption.",
     'output': f"Here it is: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a captioned motion sequence of interaction between two persons.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate a random interaction and caption it.",
     'output': f"Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Produce an interaction sequence of two people with a caption.",
     'output': f"{UNION_CAPTION_HOLDER} - {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Can you create a motion interaction and its caption?",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER} - {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate a captioned interaction motion.",
     'output': f"Here’s the interaction: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a captioned interaction motion.",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, and here’s the caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate movements of interaction between two people and a caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Produce an interaction of two persons and include its caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create interaction motions with a description caption?",
     'output': f"Motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate an interaction movement and give me a caption.",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a motion of interaction between two people and its caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate an interaction motion and provide a caption.",
     'output': f"Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Please give me an interaction motion of two people with its caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Produce a sequence of interactive motions and a caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, and the caption is: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate a motion of two persons and include a caption.",
     'output': f"Here’s the motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create interaction movements between two people with a caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate a captioned motion sequence for interaction.",
     'output': f"Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Can you randomly create an interaction with its caption?",
     'output': f"{UNION_CAPTION_HOLDER} - {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce an interaction motion and describe it with a caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, and here’s the caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate an interactive motion with a description.",
     'output': f"Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a motion interaction with a caption for two people.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate a sequence of interaction with a caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please generate a random interaction motion and provide a caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Produce a random motion and give me a caption for it.",
     'output': f"{UNION_CAPTION_HOLDER} - Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly generate a motion sequence of interaction with a caption.",
     'output': f"Here’s the motion: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a captioned interactive movement for two persons.",
     'output': f"Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Generate an interaction of two people and provide a caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - {UNION_CAPTION_HOLDER}"},

    {'input': f"Can you provide an interaction and its caption?",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Please create a motion of interaction and a caption.",
     'output': f"{UNION_CAPTION_HOLDER}: {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Produce a random interaction with its caption.",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}, and the caption is: {UNION_CAPTION_HOLDER}"},

    {'input': f"Randomly generate an interaction motion with a caption included.",
     'output': f"Here’s the interaction: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Create a motion with a caption for two people interacting.",
     'output': f"{UNION_CAPTION_HOLDER} - {INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER}"},

    {'input': f"Generate a motion and caption of interaction between two persons.",
     'output': f"Motion: {MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Can you create a captioned motion of two people interacting?",
     'output': f"{INTERACTOR_MOTION_HOLDER}{MOTION_HOLDER} - Caption: {UNION_CAPTION_HOLDER}"},

    {'input': f"Produce an interaction sequence and describe it with a caption.",
     'output': f"{MOTION_HOLDER}{INTERACTOR_MOTION_HOLDER}, and here’s the caption: {UNION_CAPTION_HOLDER}"},
]
