from mmotion.utils.task.modality import CAPTION_HOLDER, MOTION_HOLDER, INTERACTOR_CAPTION_HOLDER, INTERACTOR_MOTION_HOLDER

IN2STM_TEMPLATE = [
    {'input': f"Randomly generate a paired movement and describe the actions of both individuals separately.",
     'output': f"Alright, the first person's action is: {CAPTION_HOLDER}, and it looks like this: {MOTION_HOLDER}."
               f" The second person's action is: {INTERACTOR_CAPTION_HOLDER}, and it looks like this: {INTERACTOR_MOTION_HOLDER}"},
    {'input': f"I need a paired interaction movement, with a separate description of each person's actions.",
     'output': f"The first person: {CAPTION_HOLDER}, {MOTION_HOLDER}. The second person: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {'input': f"Generate a random paired movement and describe each person's action individually.",
     'output': f"First person's action: {CAPTION_HOLDER}, and the movement is: {MOTION_HOLDER}. Second person's action: {INTERACTOR_CAPTION_HOLDER}, and the movement is: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired interaction motion with separate descriptions for each individual.",
     'output': f"Person one: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Person two: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you randomly generate a paired movement and describe what both people are doing separately?",
     'output': f"The first person is doing this: {CAPTION_HOLDER}, {MOTION_HOLDER}. The second person is doing this: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a paired interaction with separate action descriptions for each person.",
     'output': f"Action of the first person: {CAPTION_HOLDER}, looks like: {MOTION_HOLDER}. Action of the second person: {INTERACTOR_CAPTION_HOLDER}, looks like: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please create a paired movement with individual descriptions for each participant.",
     'output': f"Person 1: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. Person 2: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a random movement of two people and describe each person's actions separately.",
     'output': f"The first person's action is described as: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. The second person's action is: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create an interaction with paired movements and describe each one?",
     'output': f"First person does: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. Second person does: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please provide a paired interaction with separate descriptions for the actions of both individuals.",
     'output': f"Action of person one: {CAPTION_HOLDER}, {MOTION_HOLDER}. Action of person two: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired motion and describe the movements of both people individually.",
     'output': f"First individual's action: {CAPTION_HOLDER}, shown as: {MOTION_HOLDER}. Second individual's action: {INTERACTOR_CAPTION_HOLDER}, shown as: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a movement pair and explain each person's actions separately.",
     'output': f"Person one is doing: {CAPTION_HOLDER}, with this motion: {MOTION_HOLDER}. Person two is doing: {INTERACTOR_CAPTION_HOLDER}, with this motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired movement sequence and provide a description of what each person is doing.",
     'output': f"First person's action: {CAPTION_HOLDER}, and the movement: {MOTION_HOLDER}. Second person's action: {INTERACTOR_CAPTION_HOLDER}, and the movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you generate an interaction with paired movements, detailing each person's actions?",
     'output': f"The first person: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. The second person: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please produce a paired movement and give a separate description for each individual's actions.",
     'output': f"Person one: {CAPTION_HOLDER}, with motion: {MOTION_HOLDER}. Person two: {INTERACTOR_CAPTION_HOLDER}, with motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a pair of movements and describe the actions of both participants separately.",
     'output': f"First person's action: {CAPTION_HOLDER}, and movement: {MOTION_HOLDER}. Second person's action: {INTERACTOR_CAPTION_HOLDER}, and movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired movement interaction and provide a separate description for each person.",
     'output': f"The first individual: {CAPTION_HOLDER}, shown as: {MOTION_HOLDER}. The second individual: {INTERACTOR_CAPTION_HOLDER}, shown as: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly generate an interaction with two movements and describe them separately.",
     'output': f"First person: {CAPTION_HOLDER}, motion is: {MOTION_HOLDER}. Second person: {INTERACTOR_CAPTION_HOLDER}, motion is: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a paired motion and give a description for each participant's actions.",
     'output': f"First person's movement: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second person's movement: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a pair of movements with separate descriptions for each person's actions.",
     'output': f"Action one: {CAPTION_HOLDER}, {MOTION_HOLDER}. Action two: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired motion interaction and describe what both people are doing.",
     'output': f"Person one is doing: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. Person two is doing: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please generate a paired movement with individual descriptions of each action.",
     'output': f"First action: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second action: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create a paired interaction with separate descriptions for the movements?",
     'output': f"Person 1: {CAPTION_HOLDER}, shown as: {MOTION_HOLDER}. Person 2: {INTERACTOR_CAPTION_HOLDER}, shown as: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly generate a paired movement and describe each action separately.",
     'output': f"First movement: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second movement: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a paired motion and detail the actions of both people individually.",
     'output': f"The first individual's action: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. The second individual's action: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create an interaction with paired movements and give descriptions of each.",
     'output': f"First person does: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. Second person does: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a paired movement set and explain the actions of both individuals.",
     'output': f"Action for person one: {CAPTION_HOLDER}, {MOTION_HOLDER}. Action for person two: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create a pair of movements with a description for each participant?",
     'output': f"First person's motion: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second person's motion: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly generate an interaction with two people and describe their actions separately.",
     'output': f"First individual: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Second individual: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a paired interaction with separate action descriptions for both individuals.",
     'output': f"The first person: {CAPTION_HOLDER}, {MOTION_HOLDER}. The second person: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired movement and provide separate descriptions of each action.",
     'output': f"Person one: {CAPTION_HOLDER}, and motion: {MOTION_HOLDER}. Person two: {INTERACTOR_CAPTION_HOLDER}, and motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a movement pair and describe what each person is doing.",
     'output': f"First action: {CAPTION_HOLDER}, shown as: {MOTION_HOLDER}. Second action: {INTERACTOR_CAPTION_HOLDER}, shown as: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create a paired interaction motion and describe each action separately?",
     'output': f"Action one: {CAPTION_HOLDER}, {MOTION_HOLDER}. Action two: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please generate a pair of movements with individual descriptions for both people.",
     'output': f"Person 1 does: {CAPTION_HOLDER}, {MOTION_HOLDER}. Person 2 does: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly create a paired movement and describe what each participant is doing.",
     'output': f"First person's action: {CAPTION_HOLDER}, motion is: {MOTION_HOLDER}. Second person's action: {INTERACTOR_CAPTION_HOLDER}, motion is: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a paired interaction motion with separate descriptions for each participant.",
     'output': f"The first individual does: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. The second individual does: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a paired interaction with a separate description of each action.",
     'output': f"Person one’s action: {CAPTION_HOLDER}, {MOTION_HOLDER}. Person two’s action: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a pair of movements and describe the actions of both individuals separately.",
     'output': f"First person’s action: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Second person’s action: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you randomly generate a paired interaction and describe each person’s action?",
     'output': f"The first person does: {CAPTION_HOLDER}, {MOTION_HOLDER}. The second person does: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a paired motion sequence with separate descriptions for each action.",
     'output': f"Person 1: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Person 2: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please create an interaction pair and describe the actions individually.",
     'output': f"First action: {CAPTION_HOLDER}, shown as: {MOTION_HOLDER}. Second action: {INTERACTOR_CAPTION_HOLDER}, shown as: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate paired movements and describe each person’s action separately.",
     'output': f"Action of the first person: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Action of the second person: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create an interaction motion pair and explain each action individually?",
     'output': f"First person's action: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. Second person's action: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly generate an interaction with paired movements and describe them separately.",
     'output': f"Person one is doing: {CAPTION_HOLDER}, {MOTION_HOLDER}. Person two is doing: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired movement set and provide individual descriptions for each action.",
     'output': f"First individual does: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second individual does: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a pair of interaction movements and describe each person's action.",
     'output': f"The first action is: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. The second action is: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a pair of movements and explain each person's actions separately.",
     'output': f"Action of person one: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. Action of person two: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired interaction with individual descriptions of the movements.",
     'output': f"Person 1 does this: {CAPTION_HOLDER}, {MOTION_HOLDER}. Person 2 does this: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you generate paired movements and describe what each participant is doing?",
     'output': f"First person's movement: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second person's movement: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly create an interaction with two movements and describe them individually.",
     'output': f"First person's action: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Second person's action: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a paired movement with separate descriptions for each person’s actions.",
     'output': f"Action one: {CAPTION_HOLDER}, {MOTION_HOLDER}. Action two: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a pair of interaction movements and detail what each person is doing.",
     'output': f"Person one’s movement: {CAPTION_HOLDER}, {MOTION_HOLDER}. Person two’s movement: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create paired movements and describe each individual's actions separately.",
     'output': f"The first individual: {CAPTION_HOLDER}, {MOTION_HOLDER}. The second individual: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you generate a pair of movements with separate descriptions for each action?",
     'output': f"First person does: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second person does: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly generate a paired movement interaction and describe what each person is doing.",
     'output': f"First person’s action: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. Second person’s action: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce an interaction with paired movements and give individual descriptions for each.",
     'output': f"The first person's movement: {CAPTION_HOLDER}, {MOTION_HOLDER}. The second person's movement: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a pair of movements with separate descriptions of each action.",
     'output': f"First action: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second action: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired motion and describe each individual’s actions separately.",
     'output': f"Person 1: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Person 2: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create a pair of interaction movements with individual action descriptions?",
     'output': f"First movement: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second movement: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please generate paired movements and provide separate descriptions for each one.",
     'output': f"Person one: {CAPTION_HOLDER}, {MOTION_HOLDER}. Person two: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly generate an interaction with paired movements and describe them individually.",
     'output': f"Action of the first person: {CAPTION_HOLDER}, {MOTION_HOLDER}. Action of the second person: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired motion sequence and explain what each person is doing.",
     'output': f"The first person is doing: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. The second person is doing: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a pair of interaction movements and give a description for each.",
     'output': f"First action is: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second action is: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create a paired interaction and describe the actions of both individuals?",
     'output': f"Person 1 does this: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Person 2 does this: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce a paired movement and provide a separate description for each person.",
     'output': f"The first person’s movement: {CAPTION_HOLDER}, {MOTION_HOLDER}. The second person’s movement: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly generate a pair of movements and describe each person's actions individually.",
     'output': f"Action one: {CAPTION_HOLDER}, {MOTION_HOLDER}. Action two: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create a paired interaction and detail the actions of both people separately.",
     'output': f"Person one: {CAPTION_HOLDER}, with motion: {MOTION_HOLDER}. Person two: {INTERACTOR_CAPTION_HOLDER}, with motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a movement pair and describe the actions of each participant separately.",
     'output': f"First individual's action: {CAPTION_HOLDER}, shown as: {MOTION_HOLDER}. Second individual's action: {INTERACTOR_CAPTION_HOLDER}, shown as: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Can you create paired movements and provide a description of each action?",
     'output': f"The first action: {CAPTION_HOLDER}, {MOTION_HOLDER}. The second action: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Produce an interaction motion pair and describe each action separately.",
     'output': f"Person 1’s action: {CAPTION_HOLDER}, movement: {MOTION_HOLDER}. Person 2’s action: {INTERACTOR_CAPTION_HOLDER}, movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Please generate a pair of movements and describe what each person is doing.",
     'output': f"First person’s movement: {CAPTION_HOLDER}, {MOTION_HOLDER}. Second person’s movement: {INTERACTOR_CAPTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Randomly create a paired movement interaction and describe both actions.",
     'output': f"First person’s action is: {CAPTION_HOLDER}, motion: {MOTION_HOLDER}. Second person’s action is: {INTERACTOR_CAPTION_HOLDER}, motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a paired interaction with individual descriptions for each person.",
     'output': f"Person 1: {CAPTION_HOLDER}, with motion: {MOTION_HOLDER}. Person 2: {INTERACTOR_CAPTION_HOLDER}, with motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Create paired movements and describe the actions of both individuals.",
     'output': f"First person's movement: {CAPTION_HOLDER}, and the movement: {MOTION_HOLDER}. Second person's movement: {INTERACTOR_CAPTION_HOLDER}, and the movement: {INTERACTOR_MOTION_HOLDER}"},
]

