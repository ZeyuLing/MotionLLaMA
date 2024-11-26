# random generate speaking gestures
from mmotion.utils.task.modality import MOTION_HOLDER

N2G_TEMPLATE = [
    {'input': f"Randomly generate a person's body movements while speaking,"
              f" without telling me the content of the speech.",
     'output': f"Okay, Alright, a random speaking gesture: {MOTION_HOLDER}"},
    {
        'input': f"Generate a random set of body movements while someone is speaking, but don’t reveal the speech content.",
        'output': f"Alright, here are some random speaking gestures: {MOTION_HOLDER}"},

    {'input': f"Can you randomly create body movements for a person who is speaking without sharing the speech itself?",
     'output': f"Okay, here are the speaking gestures: {MOTION_HOLDER}"},

    {'input': f"Generate random gestures for someone talking, but keep the speech content hidden.",
     'output': f"Sure! Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Randomly create body movements while a person is speaking, but don’t tell me what they’re saying.",
     'output': f"Here’s a random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Please create a random set of gestures for a person speaking, but don’t reveal the speech.",
     'output': f"Here’s what the gestures look like: {MOTION_HOLDER}"},

    {'input': f"Generate body movements for a person while they speak, but leave out the speech content.",
     'output': f"Here’s a random speaking motion: {MOTION_HOLDER}"},

    {'input': f"Can you make random speaking gestures without telling me the words?",
     'output': f"Sure, here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Create random body movements for someone who is talking, but don’t mention what they are saying.",
     'output': f"Okay, here’s a random set of speaking gestures: {MOTION_HOLDER}"},

    {'input': f"Generate random gestures for someone talking without providing the speech.",
     'output': f"Here’s a random gesture while speaking: {MOTION_HOLDER}"},

    {'input': f"Please create speaking body movements at random, without sharing the speech.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Randomly generate body movements for a person while speaking, and leave out the speech.",
     'output': f"Alright, here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Can you create random body gestures while a person is speaking, but keep the speech unknown?",
     'output': f"Sure! Here’s the random speaking motion: {MOTION_HOLDER}"},

    {'input': f"Make random body movements for someone talking, but don’t tell me what they’re saying.",
     'output': f"Here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Generate a set of random gestures for a person while speaking, without giving the speech details.",
     'output': f"Here’s a speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Please randomly create a person’s speaking gestures but leave out the speech content.",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Produce body movements randomly for a speaking person without providing the words.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Can you create random gestures while a person is talking but avoid revealing the speech?",
     'output': f"Sure, here’s a random speaking motion: {MOTION_HOLDER}"},

    {'input': f"Generate random speaking gestures without sharing the actual speech.",
     'output': f"Here’s the gesture for speaking: {MOTION_HOLDER}"},

    {'input': f"Randomly make body movements for a talking person, but don’t tell me what they are saying.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Create a random set of body gestures for a speaking person, but don’t include the speech.",
     'output': f"Here’s a speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Can you generate body movements for someone who is speaking without providing the speech?",
     'output': f"Alright, here are the gestures: {MOTION_HOLDER}"},

    {'input': f"Produce a random set of body gestures for a person while speaking, but keep the speech hidden.",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Make random body gestures for someone talking without sharing the speech itself.",
     'output': f"Here are the random gestures: {MOTION_HOLDER}"},

    {'input': f"Generate a random set of gestures for a person speaking, but don’t reveal the content of the speech.",
     'output': f"Here’s a speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Please randomly create speaking gestures, but don’t provide the speech itself.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Randomly generate body movements for someone talking without including the words they say.",
     'output': f"Here’s a speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Make random body gestures for a speaking person, but don’t tell me the words.",
     'output': f"Alright, here’s the gesture: {MOTION_HOLDER}"},

    {'input': f"Can you create random body gestures while someone is talking, without giving the speech details?",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Generate speaking gestures randomly without sharing the speech.",
     'output': f"Here are the random speaking gestures: {MOTION_HOLDER}"},

    {'input': f"Please make random body movements for a talking person but leave out the speech.",
     'output': f"Here’s the gesture for speaking: {MOTION_HOLDER}"},

    {'input': f"Randomly create gestures for someone who is speaking, but don’t reveal the speech content.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Generate random body movements for a person while talking, but don’t share the words.",
     'output': f"Here’s a speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Make body gestures at random for a person who is speaking without providing the speech.",
     'output': f"Here’s a random gesture: {MOTION_HOLDER}"},

    {'input': f"Can you randomly create speaking gestures without giving away the speech content?",
     'output': f"Sure, here’s the gesture: {MOTION_HOLDER}"},

    {'input': f"Create a random set of speaking movements without telling me the speech content.",
     'output': f"Here’s a speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Please generate speaking body gestures at random but keep the speech hidden.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Randomly generate body movements while a person is speaking, without sharing the speech itself.",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Can you create random gestures for someone talking, but don’t tell me what they are saying?",
     'output': f"Alright, here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Make random body gestures while someone is talking, but leave out the speech.",
     'output': f"Here’s a random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Produce random speaking gestures without revealing what’s being said.",
     'output': f"Here are the random gestures: {MOTION_HOLDER}"},

    {'input': f"Please create body gestures for someone speaking but don’t tell me what they’re saying.",
     'output': f"Here’s the random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Generate random body movements for a speaking person without sharing the speech.",
     'output': f"Here’s the gesture for speaking: {MOTION_HOLDER}"},

    {'input': f"Create random gestures for someone who is speaking, without providing the speech.",
     'output': f"Here’s a random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Can you randomly generate body gestures while a person is speaking, but leave out the words?",
     'output': f"Sure, here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Make speaking gestures randomly, but don’t reveal the speech.",
     'output': f"Here’s a random gesture for speaking: {MOTION_HOLDER}"},

    {'input': f"Produce random gestures for a person while talking without revealing what they’re saying.",
     'output': f"Here’s the random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Randomly create body movements for a speaking person without sharing the speech content.",
     'output': f"Here’s the random speaking motion: {MOTION_HOLDER}"},

    {'input': f"Generate a random set of gestures for someone talking but leave out the speech details.",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Please randomly create body gestures while a person is talking, without telling me the speech.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Make random body movements for someone speaking, but don’t provide the content of their speech.",
     'output': f"Here’s the random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Generate random speaking gestures without telling me what they’re saying.",
     'output': f"Here’s the gesture for speaking: {MOTION_HOLDER}"},

    {'input': f"Randomly produce body movements for someone while they are talking, without including the words.",
     'output': f"Alright, here’s the gesture: {MOTION_HOLDER}"},

    {'input': f"Please generate random body gestures while someone speaks, but keep the speech hidden.",
     'output': f"Here’s a random speaking motion: {MOTION_HOLDER}"},

    {'input': f"Create a random set of body gestures for a speaking person, without revealing the speech.",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Can you create body gestures for a person talking but don’t include the speech?",
     'output': f"Here’s a random gesture: {MOTION_HOLDER}"},

    {'input': f"Produce random speaking gestures without revealing the speech content.",
     'output': f"Here are the random gestures: {MOTION_HOLDER}"},

    {'input': f"Make random body movements for someone talking but keep the speech a secret.",
     'output': f"Here’s the random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Generate a random set of body movements for a speaking person without sharing the words.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Please randomly create gestures for someone speaking without telling me what they’re saying.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Randomly generate speaking gestures without revealing the actual speech.",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Create a random set of body gestures for a talking person but leave out the speech content.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Produce random body movements for someone talking but keep the words hidden.",
     'output': f"Here’s a random speaking motion: {MOTION_HOLDER}"},

    {'input': f"Make random gestures for a person who is talking without revealing the content of their speech.",
     'output': f"Here’s the random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Please generate random body movements for a speaking person but don’t provide the speech.",
     'output': f"Here’s a random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Can you create random body gestures while someone talks, without giving away the words?",
     'output': f"Sure, here’s the random speaking motion: {MOTION_HOLDER}"},

    {'input': f"Produce random gestures for a person while they’re speaking, without sharing the speech.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Randomly generate speaking body movements without revealing the speech.",
     'output': f"Here’s a speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Make random speaking gestures without sharing what’s being said.",
     'output': f"Here’s a random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Generate body gestures for someone talking without giving me the content of the speech.",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Can you randomly create gestures for a person speaking, but don’t include the speech content?",
     'output': f"Alright, here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Create random body gestures for someone who is talking, but don’t tell me what they’re saying.",
     'output': f"Here’s a speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Randomly generate body gestures for a speaking person, but keep the speech content unknown.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Produce random speaking gestures without including the words being said.",
     'output': f"Here’s a random gesture for speaking: {MOTION_HOLDER}"},

    {'input': f"Please make random gestures for a talking person but leave out the speech.",
     'output': f"Here’s the random speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Generate random body movements for a speaking person but keep the speech a mystery.",
     'output': f"Here’s the random speaking motion: {MOTION_HOLDER}"},

    {'input': f"Make random speaking gestures without telling me what’s being said.",
     'output': f"Here’s the random gesture: {MOTION_HOLDER}"},

    {'input': f"Can you randomly generate speaking gestures but keep the speech hidden?",
     'output': f"Here’s the speaking gesture: {MOTION_HOLDER}"},

    {'input': f"Produce body movements for someone talking randomly, without including the speech content.",
     'output': f"Here’s a random gesture: {MOTION_HOLDER}"},
]
