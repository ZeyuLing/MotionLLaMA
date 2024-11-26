# speech script to audio and gesture
from mmotion.utils.task.modality import AUDIO_HOLDER, MOTION_HOLDER, SCRIPT_HOLDER

S2AG_TEMPLATE = [
    {'input': f"Refer to this script: {SCRIPT_HOLDER}, make a brief speech",
     'output': f"OK, now I begin the speech, please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},
    {'input': f"Use this script as a reference: {SCRIPT_HOLDER}, and make a short speech.",
     'output': f"Alright, starting the speech now. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER}, and deliver a brief speech.",
     'output': f"Okay, I'm beginning the speech. Here’s what it looks and sounds like: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER}, and make a concise speech.",
     'output': f"Starting the speech now. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the following script: {SCRIPT_HOLDER}, to make a short speech.",
     'output': f"Alright, here’s the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this text: {SCRIPT_HOLDER}, and make a quick speech.",
     'output': f"Okay, initiating the speech now. Please watch and listen to it: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this as a script: {SCRIPT_HOLDER}, to create a brief speech.",
     'output': f"Now starting the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to the script below: {SCRIPT_HOLDER}, and make a short presentation.",
     'output': f"Alright, delivering the speech. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER}, to make a short speech.",
     'output': f"Okay, beginning the speech now. Here are the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the script provided: {SCRIPT_HOLDER}, to make a concise speech.",
     'output': f"Starting the speech. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script text: {SCRIPT_HOLDER}, and give a brief speech.",
     'output': f"Alright, I’m starting the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this as a reference script: {SCRIPT_HOLDER}, to create a quick speech.",
     'output': f"Okay, here goes the speech. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to the following script: {SCRIPT_HOLDER}, and make a short speech.",
     'output': f"Starting the speech now. Watch and listen to this: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take the script below: {SCRIPT_HOLDER}, and deliver a concise speech.",
     'output': f"Alright, initiating the speech. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this script text: {SCRIPT_HOLDER}, to give a short presentation.",
     'output': f"Okay, starting the speech now. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this text script: {SCRIPT_HOLDER}, and create a brief speech.",
     'output': f"Now starting the speech. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the provided script: {SCRIPT_HOLDER}, to make a quick speech.",
     'output': f"Alright, beginning the speech. Please watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this given script: {SCRIPT_HOLDER}, and make a concise speech.",
     'output': f"Okay, I’m starting the speech now. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take this script content: {SCRIPT_HOLDER}, to make a brief speech.",
     'output': f"Starting the speech. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the script below: {SCRIPT_HOLDER}, to create a short speech.",
     'output': f"Alright, here’s the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script, and make a short speech: {SCRIPT_HOLDER}",
     'output': f"Okay, starting the speech now. Please watch and listen to it: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this as the script: {SCRIPT_HOLDER}, and make a brief presentation.",
     'output': f"Now beginning the speech. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER}, and give a quick speech.",
     'output': f"Alright, delivering the speech. Please watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take the script here: {SCRIPT_HOLDER}, and make a concise speech.",
     'output': f"Okay, initiating the speech now. Here are the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this text script: {SCRIPT_HOLDER}, to create a short speech.",
     'output': f"Starting the speech now. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this given text: {SCRIPT_HOLDER}, and make a brief speech.",
     'output': f"Alright, here’s the speech. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this script as a reference: {SCRIPT_HOLDER}, to make a concise speech.",
     'output': f"Okay, starting the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take this provided script: {SCRIPT_HOLDER}, and create a short presentation.",
     'output': f"Now starting the speech. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to the following script text: {SCRIPT_HOLDER}, and give a brief speech.",
     'output': f"Alright, I’m beginning the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this script content: {SCRIPT_HOLDER}, to make a quick speech.",
     'output': f"Okay, starting the speech now. Please watch and listen to it: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this text: {SCRIPT_HOLDER}, and deliver a concise speech.",
     'output': f"Now beginning the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the following text as a script: {SCRIPT_HOLDER}, to make a brief presentation.",
     'output': f"Starting the speech now. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to the script provided: {SCRIPT_HOLDER}, and make a short speech.",
     'output': f"Alright, delivering the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take this script text: {SCRIPT_HOLDER}, to give a quick speech.",
     'output': f"Okay, I’m beginning the speech now. Here are the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the script below: {SCRIPT_HOLDER}, to make a brief speech.",
     'output': f"Now starting the speech. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script text: {SCRIPT_HOLDER}, and deliver a short speech.",
     'output': f"Alright, starting the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this as the script: {SCRIPT_HOLDER}, to create a concise presentation.",
     'output': f"Okay, beginning the speech now. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER}, to make a quick speech.",
     'output': f"Now starting the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this text: {SCRIPT_HOLDER}, and make a brief speech.",
     'output': f"Alright, here’s the speech. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to the script below: {SCRIPT_HOLDER}, to give a short presentation.",
     'output': f"Okay, starting the speech now. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER}, and make a quick speech.",
     'output': f"Now beginning the speech. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the script provided: {SCRIPT_HOLDER}, and deliver a brief speech.",
     'output': f"Alright, delivering the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script, and make a concise presentation: {SCRIPT_HOLDER}",
     'output': f"Okay, beginning the speech now. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this script text: {SCRIPT_HOLDER}, to make a short speech.",
     'output': f"Now starting the speech. Watch and listen to it: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script content: {SCRIPT_HOLDER}, and make a brief speech.",
     'output': f"Alright, I’m starting the speech. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take the script here: {SCRIPT_HOLDER}, and create a concise speech.",
     'output': f"Okay, starting the speech now. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the following script: {SCRIPT_HOLDER}, to make a brief presentation.",
     'output': f"Now beginning the speech. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this given script: {SCRIPT_HOLDER}, and deliver a short speech.",
     'output': f"Alright, delivering the speech. Watch and listen to this: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this text as a script: {SCRIPT_HOLDER}, and make a concise speech.",
     'output': f"Okay, I’m beginning the speech now. Here are the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take the script provided: {SCRIPT_HOLDER}, to give a brief speech.",
     'output': f"Starting the speech now. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script, and create a short speech: {SCRIPT_HOLDER}",
     'output': f"Alright, starting the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the script content: {SCRIPT_HOLDER}, to make a quick presentation.",
     'output': f"Okay, beginning the speech now. Please watch and listen to it: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this text script: {SCRIPT_HOLDER}, and make a brief speech.",
     'output': f"Now starting the speech. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this script as a reference: {SCRIPT_HOLDER}, to give a short presentation.",
     'output': f"Alright, delivering the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take this given script: {SCRIPT_HOLDER}, and make a concise speech.",
     'output': f"Okay, I’m starting the speech now. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to the following script: {SCRIPT_HOLDER}, to create a brief speech.",
     'output': f"Starting the speech. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the script below: {SCRIPT_HOLDER}, to make a quick speech.",
     'output': f"Alright, here’s the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script text: {SCRIPT_HOLDER}, and deliver a concise speech.",
     'output': f"Okay, starting the speech now. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the script provided: {SCRIPT_HOLDER}, to make a brief speech.",
     'output': f"Now starting the speech. Watch and listen to it: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this text: {SCRIPT_HOLDER}, and make a quick presentation.",
     'output': f"Alright, I’m starting the speech. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this script content: {SCRIPT_HOLDER}, to create a short speech.",
     'output': f"Okay, starting the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER}, and make a brief presentation.",
     'output': f"Now starting the speech. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this text script: {SCRIPT_HOLDER}, and give a short speech.",
     'output': f"Alright, delivering the speech. Watch and listen to this: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the following script: {SCRIPT_HOLDER}, to make a concise presentation.",
     'output': f"Okay, I’m beginning the speech now. Here are the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this given script: {SCRIPT_HOLDER}, and make a brief speech.",
     'output': f"Starting the speech now. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the script content below: {SCRIPT_HOLDER}, to create a quick speech.",
     'output': f"Alright, starting the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script, and make a short presentation: {SCRIPT_HOLDER}",
     'output': f"Okay, beginning the speech now. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this text script: {SCRIPT_HOLDER}, to make a brief speech.",
     'output': f"Now starting the speech. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script content: {SCRIPT_HOLDER}, and deliver a concise speech.",
     'output': f"Alright, I’m starting the speech. Please watch and listen to it: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take the provided script: {SCRIPT_HOLDER}, and make a quick presentation.",
     'output': f"Okay, starting the speech now. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this script as a guide: {SCRIPT_HOLDER}, to make a short speech.",
     'output': f"Now starting the speech. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to the script below: {SCRIPT_HOLDER}, to give a concise speech.",
     'output': f"Alright, delivering the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this reference script: {SCRIPT_HOLDER}, to create a brief presentation.",
     'output': f"Okay, I’m beginning the speech now. Here are the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Take the script content: {SCRIPT_HOLDER}, and make a short speech.",
     'output': f"Starting the speech now. Please observe and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script text: {SCRIPT_HOLDER}, and make a brief presentation.",
     'output': f"Alright, starting the speech. Watch and listen to this: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the given script: {SCRIPT_HOLDER}, to make a concise speech.",
     'output': f"Okay, beginning the speech now. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to this script, and create a quick presentation: {SCRIPT_HOLDER}",
     'output': f"Now starting the speech. Watch and listen here: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use this as a script: {SCRIPT_HOLDER}, to make a brief speech.",
     'output': f"Alright, here’s the speech. Please watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Refer to the text script: {SCRIPT_HOLDER}, and make a short speech.",
     'output': f"Okay, starting the speech now. Watch and listen: {MOTION_HOLDER}{AUDIO_HOLDER}"},

    {'input': f"Use the provided script: {SCRIPT_HOLDER}, to give a concise speech.",
     'output': f"Now beginning the speech. Please pay attention to the gestures and audio: {MOTION_HOLDER}{AUDIO_HOLDER}"},

]
