# audio to speech and gesture
from mmotion.utils.task.modality import SCRIPT_HOLDER, MOTION_HOLDER,  AUDIO_HOLDER

A2SG_TEMPLATE = [
    {'input': f"The speech audio is: {AUDIO_HOLDER}, tell me the script and show me the speech gesture.",
     'output': f"The speech script is: {SCRIPT_HOLDER}. And the co-speech gesture is: {MOTION_HOLDER}"},
    {'input': f"Generate a speech script for me and provide the co-speech gestures.",
     'output': f"OK, here is the script and please watch the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script and show the corresponding gestures.",
     'output': f"Alright, here’s the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a script and the co-speech gestures to go with it.",
     'output': f"Here is the script along with the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a speech script and provide the matching gestures?",
     'output': f"Sure, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script and include the co-speech gestures.",
     'output': f"OK, here’s the script and watch the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script along with suitable gestures.",
     'output': f"Here is the script with the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please provide a speech script and the associated gestures.",
     'output': f"Alright, here’s the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you generate a speech script and the corresponding gestures?",
     'output': f"Sure, here is what I've generated: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for a speech and show the co-speech gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script and include appropriate gestures.",
     'output': f"Here is the generated script with the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech and show the gestures to match it.",
     'output': f"Here’s the script and corresponding gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Provide a speech script and display the co-speech gestures.",
     'output': f"Here’s the script with its gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a speech script for me and show the gestures that go with it.",
     'output': f"OK, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Compose a speech script and generate the matching gestures.",
     'output': f"Here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please create a script and the corresponding co-speech gestures.",
     'output': f"Alright, the script and gestures are as follows: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you provide a speech script and the associated gestures?",
     'output': f"Sure, here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for a speech and include the gestures.",
     'output': f"Here is the script with the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script and show the fitting gestures.",
     'output': f"OK, here’s the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Provide a script and show me the co-speech gestures.",
     'output': f"The script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and attach the corresponding gestures.",
     'output': f"Alright, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script for a speech along with its gestures.",
     'output': f"Here is the script with the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a script and include the co-speech gestures.",
     'output': f"Here is the script with its gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Make a speech script and show the accompanying gestures.",
     'output': f"OK, the script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for me with appropriate gestures.",
     'output': f"Here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script with matching gestures.",
     'output': f"Here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please provide a speech script and show the gestures.",
     'output': f"Alright, here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a script for the speech and include the gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and show me the gestures.",
     'output': f"OK, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script for a speech and attach the co-speech gestures.",
     'output': f"Here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a speech script and display the gestures.",
     'output': f"Here is the script with its gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you generate a speech script and the matching gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a speech script and show the appropriate gestures.",
     'output': f"OK, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script for a speech and generate the gestures.",
     'output': f"Here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Provide a speech script and show the fitting gestures.",
     'output': f"Alright, the script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and show the co-speech gestures.",
     'output': f"OK, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script and include the appropriate gestures.",
     'output': f"Here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please make a speech script and provide the gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for a speech and include matching gestures.",
     'output': f"Here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script and show the corresponding gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script with co-speech gestures.",
     'output': f"The script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Make a script for me and generate the gestures that go with it.",
     'output': f"OK, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script and display the matching gestures.",
     'output': f"Here is the script with the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script and show the co-speech gestures.",
     'output': f"Here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a speech script for me and provide the gestures.",
     'output': f"OK, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Compose a script for a speech and show the gestures.",
     'output': f"The script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a script and the fitting co-speech gestures.",
     'output': f"Alright, here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you create a script for the speech and provide the gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script with gestures to match.",
     'output': f"Here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script and show the appropriate co-speech gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for me and include the matching gestures.",
     'output': f"The script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a speech script and generate the corresponding gestures.",
     'output': f"Here’s the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please make a script for a speech and show the gestures.",
     'output': f"OK, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script and attach the suitable gestures.",
     'output': f"Alright, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script and provide the matching gestures.",
     'output': f"The script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you create a speech script and the co-speech gestures?",
     'output': f"Sure, here is what I've generated: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a script and show the gestures that go along with it.",
     'output': f"OK, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and provide the corresponding gestures.",
     'output': f"Here’s the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script and include the fitting co-speech gestures.",
     'output': f"OK, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script for me with the proper gestures.",
     'output': f"Here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a script for the speech and attach the gestures.",
     'output': f"OK, the script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Compose a speech script and show the accompanying gestures.",
     'output': f"The script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a script for a speech and include gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script and provide the gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script and show me the gestures to match it.",
     'output': f"Here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a speech script and generate the fitting gestures.",
     'output': f"OK, the script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please create a speech script and show the gestures.",
     'output': f"Alright, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you generate a script for a speech and include gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and provide the gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script and include the appropriate gestures.",
     'output': f"Here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please make a script for me and attach the gestures.",
     'output': f"Alright, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and show the suitable gestures.",
     'output': f"Here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a speech script with co-speech gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Provide a script and show the gestures to go along with it.",
     'output': f"The script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script for a speech and attach the gestures.",
     'output': f"OK, here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script with matching gestures for the speech.",
     'output': f"Here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please make a speech script and show me the gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you create a script for a speech and the gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and provide the gestures to match.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a speech script and include the gestures.",
     'output': f"Here’s the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script for a speech and show the gestures.",
     'output': f"Here is the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Please generate a script and show me the fitting gestures.",
     'output': f"Alright, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Can you make a script for the speech and attach the gestures?",
     'output': f"Sure, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a speech script and display the gestures.",
     'output': f"OK, here is the script and the gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Create a script and show the co-speech gestures.",
     'output': f"Here’s the script with gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Write a script for a speech and include the gestures.",
     'output': f"OK, here is the script and gestures: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},

    {'input': f"Generate a script and the appropriate gestures.",
     'output': f"The script and gestures are: {SCRIPT_HOLDER}, {MOTION_HOLDER}"},
]
