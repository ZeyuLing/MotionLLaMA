from mmotion.utils.task.modality import DURATION_HOLDER, MOTION_HOLDER, CAPTION_HOLDER

L2TM_TEMPLATE = [
    {'input': f"Give me a motion with {DURATION_HOLDER} seconds, and tell me what's the motion about.",
     'output': f"Here is the generated motion. {MOTION_HOLDER}, the content of the motion is: {CAPTION_HOLDER}"},
    {'input': f"Show me a {DURATION_HOLDER}-second motion, tell me the content of the motion.",
     'output': f"Okay, here is such a motion: {CAPTION_HOLDER}, the results can be visualized as: {MOTION_HOLDER}"},
    {'input': f"Create a motion that lasts {DURATION_HOLDER} seconds and describe it.",
     'output': f"This is the motion: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Generate a motion lasting {DURATION_HOLDER} seconds and explain the motion.",
     'output': f"Here is the motion: {MOTION_HOLDER}, the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Produce a {DURATION_HOLDER}-second motion and provide its description.",
     'output': f"This motion is {MOTION_HOLDER}, and the description is: {CAPTION_HOLDER}"},
    {'input': f"Can you create a motion of {DURATION_HOLDER} seconds and tell me what it is?",
     'output': f"Sure, the motion is {MOTION_HOLDER}, and it is about: {CAPTION_HOLDER}"},
    {'input': f"Construct a {DURATION_HOLDER}-second motion and describe its content.",
     'output': f"Alright, here is the motion: {MOTION_HOLDER}, the content is: {CAPTION_HOLDER}"},
    {'input': f"Design a motion for {DURATION_HOLDER} seconds and explain it.",
     'output': f"This is the motion: {MOTION_HOLDER}, and it can be explained as: {CAPTION_HOLDER}"},
    {'input': f"Give me a {DURATION_HOLDER}-second motion and describe its purpose.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and its purpose is: {CAPTION_HOLDER}"},
    {'input': f"Show me a motion that lasts {DURATION_HOLDER} seconds and tell me what it depicts.",
     'output': f"Okay, the motion is: {MOTION_HOLDER}, and it depicts: {CAPTION_HOLDER}"},
    {'input': f"Create a {DURATION_HOLDER}-second motion and give a description of it.",
     'output': f"This is the motion: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Generate a motion for {DURATION_HOLDER} seconds and explain what it is about.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it is about: {CAPTION_HOLDER}"},
    {'input': f"Produce a motion sequence lasting {DURATION_HOLDER} seconds and tell me what it shows.",
     'output': f"This motion sequence is {MOTION_HOLDER}, and it shows: {CAPTION_HOLDER}"},
    {'input': f"Can you create a {DURATION_HOLDER}-second motion and describe its content?",
     'output': f"Sure, the motion is: {MOTION_HOLDER}, and it contains: {CAPTION_HOLDER}"},
    {'input': f"Construct a motion lasting {DURATION_HOLDER} seconds and explain its purpose.",
     'output': f"Alright, the motion is: {MOTION_HOLDER}, and its purpose is: {CAPTION_HOLDER}"},
    {'input': f"Design a motion sequence for {DURATION_HOLDER} seconds and tell me about it.",
     'output': f"This motion sequence is: {MOTION_HOLDER}, and it is about: {CAPTION_HOLDER}"},
    {'input': f"Give me a motion that lasts {DURATION_HOLDER} seconds and explain it.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Show me a {DURATION_HOLDER}-second motion and describe its content.",
     'output': f"Okay, the motion is: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Create a motion lasting {DURATION_HOLDER} seconds and tell me what it is.",
     'output': f"This is the motion: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Generate a {DURATION_HOLDER}-second motion and provide an explanation.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Produce a motion of {DURATION_HOLDER} seconds and describe it.",
     'output': f"This motion is: {MOTION_HOLDER}, and it is described as: {CAPTION_HOLDER}"},
    {'input': f"Can you create a motion lasting {DURATION_HOLDER} seconds and explain its content?",
     'output': f"Sure, the motion is: {MOTION_HOLDER}, and it is about: {CAPTION_HOLDER}"},
    {'input': f"Construct a {DURATION_HOLDER}-second motion and provide its description.",
     'output': f"Alright, the motion is: {MOTION_HOLDER}, and the description is: {CAPTION_HOLDER}"},
    {'input': f"Design a motion sequence that lasts {DURATION_HOLDER} seconds and explain it.",
     'output': f"This is the motion: {MOTION_HOLDER}, and it can be explained as: {CAPTION_HOLDER}"},
    {'input': f"Give me a {DURATION_HOLDER}-second motion and describe its content.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it is about: {CAPTION_HOLDER}"},
    {'input': f"Show me a motion lasting {DURATION_HOLDER} seconds and tell me what it is.",
     'output': f"Okay, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Create a motion that lasts {DURATION_HOLDER} seconds and explain it.",
     'output': f"This is the motion: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Generate a {DURATION_HOLDER}-second motion and describe what it shows.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it shows: {CAPTION_HOLDER}"},
    {'input': f"Produce a motion lasting {DURATION_HOLDER} seconds and describe its content.",
     'output': f"This motion is: {MOTION_HOLDER}, and the content is: {CAPTION_HOLDER}"},
    {'input': f"Can you create a {DURATION_HOLDER}-second motion and explain what it is?",
     'output': f"Sure, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Construct a motion for {DURATION_HOLDER} seconds and describe its purpose.",
     'output': f"Alright, the motion is: {MOTION_HOLDER}, and its purpose is: {CAPTION_HOLDER}"},
    {'input': f"Design a {DURATION_HOLDER}-second motion and explain what it depicts.",
     'output': f"This motion is: {MOTION_HOLDER}, and it depicts: {CAPTION_HOLDER}"},
    {'input': f"Give me a motion lasting {DURATION_HOLDER} seconds and describe it.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it is described as: {CAPTION_HOLDER}"},
    {'input': f"Show me a motion that is {DURATION_HOLDER} seconds long and explain it.",
     'output': f"Okay, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Create a {DURATION_HOLDER}-second motion and describe its purpose.",
     'output': f"This is the motion: {MOTION_HOLDER}, and its purpose is: {CAPTION_HOLDER}"},
    {'input': f"Generate a motion sequence of {DURATION_HOLDER} seconds and explain it.",
     'output': f"Here is the motion sequence: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Produce a {DURATION_HOLDER}-second motion and describe what it shows.",
     'output': f"This motion is: {MOTION_HOLDER}, and it shows: {CAPTION_HOLDER}"},
    {'input': f"Can you create a motion that lasts {DURATION_HOLDER} seconds and explain it?",
     'output': f"Sure, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Construct a motion sequence of {DURATION_HOLDER} seconds and describe it.",
     'output': f"Alright, the motion sequence is: {MOTION_HOLDER}, and it is described as: {CAPTION_HOLDER}"},
    {'input': f"Design a motion for {DURATION_HOLDER} seconds and explain its content.",
     'output': f"This motion is: {MOTION_HOLDER}, and the content is: {CAPTION_HOLDER}"},
    {'input': f"Give me a motion sequence that lasts {DURATION_HOLDER} seconds and explain it.",
     'output': f"Here is the motion sequence: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Show me a {DURATION_HOLDER}-second motion sequence and describe it.",
     'output': f"Okay, the motion sequence is: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Create a motion lasting {DURATION_HOLDER} seconds and tell me about it.",
     'output': f"This motion is: {MOTION_HOLDER}, and it is about: {CAPTION_HOLDER}"},
    {'input': f"Generate a motion of {DURATION_HOLDER} seconds and explain it.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Produce a sequence lasting {DURATION_HOLDER} seconds and describe it.",
     'output': f"This sequence is: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Can you create a {DURATION_HOLDER}-second sequence and tell me what it is?",
     'output': f"Sure, the sequence is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Construct a motion of {DURATION_HOLDER} seconds and explain its content.",
     'output': f"Alright, the motion is: {MOTION_HOLDER}, and it is about: {CAPTION_HOLDER}"},
    {'input': f"Design a {DURATION_HOLDER}-second motion sequence and describe it.",
     'output': f"This motion sequence is: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Give me a motion that is {DURATION_HOLDER} seconds long and explain it.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Show me a motion lasting {DURATION_HOLDER} seconds and tell me what it is.",
     'output': f"Okay, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Create a motion sequence of {DURATION_HOLDER} seconds and describe it.",
     'output': f"This motion sequence is: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Generate a {DURATION_HOLDER}-second motion and explain its content.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and the content is: {CAPTION_HOLDER}"},
    {'input': f"Produce a motion sequence lasting {DURATION_HOLDER} seconds and tell me what it is.",
     'output': f"This motion sequence is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Can you create a motion of {DURATION_HOLDER} seconds and describe it?",
     'output': f"Sure, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Construct a {DURATION_HOLDER}-second motion sequence and explain it.",
     'output': f"Alright, the motion sequence is: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Design a motion lasting {DURATION_HOLDER} seconds and describe its content.",
     'output': f"This motion is: {MOTION_HOLDER}, and the content is: {CAPTION_HOLDER}"},
    {'input': f"Give me a {DURATION_HOLDER}-second motion and explain it.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Show me a motion sequence that lasts {DURATION_HOLDER} seconds and describe it.",
     'output': f"Okay, the motion sequence is: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Create a motion of {DURATION_HOLDER} seconds and tell me what it is.",
     'output': f"This motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Generate a sequence lasting {DURATION_HOLDER} seconds and describe it.",
     'output': f"Here is the sequence: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Produce a {DURATION_HOLDER}-second motion sequence and explain it.",
     'output': f"This motion sequence is: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Can you create a {DURATION_HOLDER}-second motion and explain its content?",
     'output': f"Sure, the motion is: {MOTION_HOLDER}, and the content is: {CAPTION_HOLDER}"},
    {'input': f"Construct a motion lasting {DURATION_HOLDER} seconds and describe it.",
     'output': f"Alright, the motion is: {MOTION_HOLDER}, and it is described as: {CAPTION_HOLDER}"},
    {'input': f"Design a {DURATION_HOLDER}-second motion and explain it.",
     'output': f"This motion is: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Give me a motion that lasts {DURATION_HOLDER} seconds and describe its content.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Show me a {DURATION_HOLDER}-second motion and tell me what it is.",
     'output': f"Okay, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Create a {DURATION_HOLDER}-second motion sequence and describe it.",
     'output': f"This motion sequence is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Generate a motion lasting {DURATION_HOLDER} seconds and explain it.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Produce a motion of {DURATION_HOLDER} seconds and describe what it shows.",
     'output': f"This motion is: {MOTION_HOLDER}, and it shows: {CAPTION_HOLDER}"},
    {'input': f"Can you create a motion sequence lasting {DURATION_HOLDER} seconds and describe it?",
     'output': f"Sure, the motion sequence is: {MOTION_HOLDER}, and it describes: {CAPTION_HOLDER}"},
    {'input': f"Construct a {DURATION_HOLDER}-second motion and explain what it is.",
     'output': f"Alright, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Design a motion sequence that lasts {DURATION_HOLDER} seconds and explain its content.",
     'output': f"This motion sequence is: {MOTION_HOLDER}, and the content is: {CAPTION_HOLDER}"},
    {'input': f"Give me a motion of {DURATION_HOLDER} seconds and describe it.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it is described as: {CAPTION_HOLDER}"},
    {'input': f"Show me a motion that is {DURATION_HOLDER} seconds long and tell me what it is.",
     'output': f"Okay, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Create a motion lasting {DURATION_HOLDER} seconds and explain it.",
     'output': f"This motion is: {MOTION_HOLDER}, and the explanation is: {CAPTION_HOLDER}"},
    {'input': f"Generate a {DURATION_HOLDER}-second motion and describe what it shows.",
     'output': f"Here is the motion: {MOTION_HOLDER}, and it shows: {CAPTION_HOLDER}"},
    {'input': f"Produce a motion lasting {DURATION_HOLDER} seconds and explain its content.",
     'output': f"This motion is: {MOTION_HOLDER}, and the content is: {CAPTION_HOLDER}"},
    {'input': f"Can you create a motion of {DURATION_HOLDER} seconds and tell me what it is?",
     'output': f"Sure, the motion is: {MOTION_HOLDER}, and it is: {CAPTION_HOLDER}"},
    {'input': f"Construct a motion that lasts {DURATION_HOLDER} seconds and describe it.",
     'output': f"Alright, the motion is: {MOTION_HOLDER}, and it is described as: {CAPTION_HOLDER}"},
    {'input': f"Design a {DURATION_HOLDER}-second motion and explain what it depicts.",
     'output': f"This motion is: {MOTION_HOLDER}, and it depicts: {CAPTION_HOLDER}"},
]