# gesture to audio and speech script
from mmotion.utils.task.modality import SCRIPT_HOLDER, AUDIO_HOLDER, MOTION_HOLDER

G2AS_TEMPLATE = [
    {'input': f"Here are the co-speech gestures from a speech: {MOTION_HOLDER}, infer the speech audio and script.",
     'output': f"The speech content may be: {SCRIPT_HOLDER}, and it may sound like: {AUDIO_HOLDER}"},
    {'input': f"Generate a speech script for me and show the accompanying gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script along with the matching gestures.",
     'output': f"Here is the generated script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please provide a speech script and the corresponding co-speech gestures.",
     'output': f"Alright, here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a speech script and generate the appropriate gestures?",
     'output': f"Sure, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and show the gestures that go along with it.",
     'output': f"OK, here's the script with the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script for a speech and include the co-speech gestures.",
     'output': f"Here is the script and the associated gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a speech script and provide the matching gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you create a speech script and the corresponding gestures?",
     'output': f"Sure, here is what I've generated: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for me and show the gestures that fit.",
     'output': f"OK, here's the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script along with the suitable co-speech gestures.",
     'output': f"Here is the generated script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please provide a script and the associated co-speech gestures.",
     'output': f"Alright, here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a script and generate the gestures to go with it?",
     'output': f"Sure, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and provide the fitting gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script for the speech and include appropriate gestures.",
     'output': f"Here is the script with the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a script and the corresponding gestures.",
     'output': f"Alright, here is the generated script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you create a speech script along with matching gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script and show me the co-speech gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script with its associated gestures.",
     'output': f"Here is the generated script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please provide a speech script and the proper gestures.",
     'output': f"Alright, here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a script and generate appropriate gestures?",
     'output': f"Sure, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for me and provide the gestures that match.",
     'output': f"OK, here's the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script for a speech and include the fitting gestures.",
     'output': f"Here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a speech script and show the co-speech gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you create a speech script and provide the matching gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and include the appropriate gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script with suitable co-speech gestures.",
     'output': f"Here is the generated script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please provide a script and the associated gestures.",
     'output': f"Alright, here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a speech script and generate the co-speech gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for the speech and show me the fitting gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script and include the corresponding gestures.",
     'output': f"Here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a speech script and provide the co-speech gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you create a script and generate the suitable gestures?",
     'output': f"Sure, here is what I've generated: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and provide the gestures that go with it.",
     'output': f"OK, here's the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script and show the associated gestures.",
     'output': f"Here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please provide a script with the fitting co-speech gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a script and provide the matching gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and show the appropriate gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script including the co-speech gestures.",
     'output': f"Here is the generated script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a script for the speech and show the gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you create a speech script with the corresponding gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script and provide the co-speech gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script for a speech and show the suitable gestures.",
     'output': f"Here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please provide a speech script and the appropriate gestures.",
     'output': f"Alright, here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a script and include the fitting gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script for me and show the matching gestures.",
     'output': f"OK, here's the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},
]
