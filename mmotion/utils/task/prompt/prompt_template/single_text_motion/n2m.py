"""
For humanml3d, motion-x.
Randomly generate a motion. with no description or other instructions
"""
from mmotion.utils.task.modality import MOTION_HOLDER

N2M_TEMPLATE = [
    {'input': "Generate a random sequence of actions for me; any actions will do.",
     "output": f"There is a motion: {MOTION_HOLDER}"},
    {'input': "Create a random action sequence for me, anything is fine.",
     'output': f"Sure, here's a motion: {MOTION_HOLDER}"},
    {'input': "Can you generate some random actions for me? Anything will do.",
     'output': f"Alright, here's a motion: {MOTION_HOLDER}"},
    {'input': "Please produce a random series of actions.", 'output': f"Got it, motion: {MOTION_HOLDER}"},
    {'input': "I'd like a random sequence of actions, any kind.", 'output': f"Okay, motion: {MOTION_HOLDER}"},
    {'input': "Give me any random actions you can think of.", 'output': f"Here's a motion: {MOTION_HOLDER}"},
    {'input': "Generate some random actions, doesn't matter what.",
     'output': f"No problem, here's a motion: {MOTION_HOLDER}"},
    {'input': "Can you make a random list of actions?", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
    {'input': "Create an action sequence at random.", 'output': f"Okay, here it is: {MOTION_HOLDER}"},
    {'input': "Give me a random set of actions.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Produce any random actions.", 'output': f"Here's one: {MOTION_HOLDER}"},
    {'input': "Generate a sequence of random actions.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Can you give me some random actions?", 'output': f"Of course, here's a motion: {MOTION_HOLDER}"},
    {'input': "Make a random series of actions for me.", 'output': f"Okay, motion: {MOTION_HOLDER}"},
    {'input': "Please create some random actions.", 'output': f"Here's a motion: {MOTION_HOLDER}"},
    {'input': "Generate random actions, anything goes.", 'output': f"Alright, here's a motion: {MOTION_HOLDER}"},
    {'input': "I'd like a set of random actions.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Can you make a list of random actions?", 'output': f"Got it, motion: {MOTION_HOLDER}"},
    {'input': "Produce a sequence of actions at random.", 'output': f"Here you go, motion: {MOTION_HOLDER}"},
    {'input': "Give me any actions, generated randomly.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Generate some actions, any will do.", 'output': f"Okay, here's a motion: {MOTION_HOLDER}"},
    {'input': "Create a random action set.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Make any actions you like, randomly.", 'output': f"Alright, here's a motion: {MOTION_HOLDER}"},
    {'input': "Produce a random list of actions.", 'output': f"Here's a motion: {MOTION_HOLDER}"},
    {'input': "Generate a random action.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Please create some actions at random.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "I'd like any random actions.", 'output': f"Got it, motion: {MOTION_HOLDER}"},
    {'input': "Give me a random action sequence.", 'output': f"Okay, here's a motion: {MOTION_HOLDER}"},
    {'input': "Can you generate a random action list?", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
    {'input': "Create any random actions.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Produce some random actions.", 'output': f"Here's a motion: {MOTION_HOLDER}"},
    {'input': "Generate a list of random actions.", 'output': f"Okay, motion: {MOTION_HOLDER}"},
    {'input': "Please make a random sequence of actions.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "I'd like a random set of actions.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Give me any random action sequence.", 'output': f"Got it, motion: {MOTION_HOLDER}"},
    {'input': "Can you produce a list of random actions?", 'output': f"Alright, here's a motion: {MOTION_HOLDER}"},
    {'input': "Create a random series of actions for me.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Generate any random actions.", 'output': f"Here's one: {MOTION_HOLDER}"},
    {'input': "Make a random action for me.", 'output': f"Okay, motion: {MOTION_HOLDER}"},
    {'input': "Produce any actions, randomly.", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
    {'input': "Give me a sequence of random actions.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Can you create some random actions?", 'output': f"Here's a motion: {MOTION_HOLDER}"},
    {'input': "Please generate a random action.", 'output': f"Okay, here's a motion: {MOTION_HOLDER}"},
    {'input': "I'd like any sequence of random actions.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Produce a random set of actions.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Make some random actions.", 'output': f"Got it, motion: {MOTION_HOLDER}"},
    {'input': "Give me a list of random actions.", 'output': f"Alright, here's a motion: {MOTION_HOLDER}"},
    {'input': "Create a random action for me.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Can you produce random actions?", 'output': f"Okay, here's a motion: {MOTION_HOLDER}"},
    {'input': "Generate any actions randomly.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Please make a random action list.", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
    {'input': "I'd like a sequence of random actions.", 'output': f"Here's a motion: {MOTION_HOLDER}"},
    {'input': "Give me a random set of actions.", 'output': f"Okay, motion: {MOTION_HOLDER}"},
    {'input': "Create any random action sequence.", 'output': f"Sure, here's a motion: {MOTION_HOLDER}"},
    {'input': "Can you generate some actions randomly?", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Please produce a random action.", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
    {'input': "Make a list of random actions.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Give me any action, generated randomly.", 'output': f"Okay, here's a motion: {MOTION_HOLDER}"},
    {'input': "Create some random actions for me.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Generate a random sequence for me.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Please make a random set of actions.", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
    {'input': "I'd like a random action series.", 'output': f"Here's a motion: {MOTION_HOLDER}"},
    {'input': "Give me any random action set.", 'output': f"Okay, motion: {MOTION_HOLDER}"},
    {'input': "Create a random list of actions for me.", 'output': f"Sure, here's a motion: {MOTION_HOLDER}"},
    {'input': "Can you produce a random action sequence?", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Generate some actions at random.", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
    {'input': "Please create any random actions.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Make a sequence of random actions for me.", 'output': f"Okay, here's a motion: {MOTION_HOLDER}"},
    {'input': "Give me a random series of actions.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Create an action set randomly.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Produce any random action sequence.", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
    {'input': "Generate a random action set.", 'output': f"Alright, motion: {MOTION_HOLDER}"},
    {'input': "Can you make a random action series?", 'output': f"Here's a motion: {MOTION_HOLDER}"},
    {'input': "Please produce some random actions.", 'output': f"Okay, motion: {MOTION_HOLDER}"},
    {'input': "Create any actions at random.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Make some actions randomly.", 'output': f"Alright, here's a motion: {MOTION_HOLDER}"},
    {'input': "Give me a sequence of random actions.", 'output': f"Okay, motion: {MOTION_HOLDER}"},
    {'input': "Generate a random action list.", 'output': f"Sure, motion: {MOTION_HOLDER}"},
    {'input': "Please create a random action sequence.", 'output': f"Alright, here's a motion: {MOTION_HOLDER}"},
    {'input': "I'd like some random actions.", 'output': f"Got it, motion: {MOTION_HOLDER}"},
    {'input': "Produce a set of random actions.", 'output': f"Sure thing, motion: {MOTION_HOLDER}"},
]
