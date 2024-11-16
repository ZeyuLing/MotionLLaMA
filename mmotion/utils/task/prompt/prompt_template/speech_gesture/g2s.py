# gesture to speech script
from mmotion.utils.task.modality import MOTION_HOLDER, SCRIPT_HOLDER

G2S_TEMPLATE = [
    {'input': f"Infer the speech script from the gestures: {MOTION_HOLDER}",
     'output': f"A possible answer is: {SCRIPT_HOLDER}"},
    {'input': f"Infer the speech script from the given gestures: {MOTION_HOLDER}",
     'output': f"A possible answer is: {SCRIPT_HOLDER}"},

    {'input': f"Generate a speech script based on these gestures: {MOTION_HOLDER}",
     'output': f"Here’s a possible script: {SCRIPT_HOLDER}"},

    {'input': f"What might be the speech script for these gestures: {MOTION_HOLDER}?",
     'output': f"One possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Based on the gestures provided: {MOTION_HOLDER}, infer the speech script.",
     'output': f"A potential script could be: {SCRIPT_HOLDER}"},

    {'input': f"Can you deduce the speech script from these gestures: {MOTION_HOLDER}?",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"From these gestures: {MOTION_HOLDER}, what would the speech script be?",
     'output': f"A likely answer is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the possible speech script using the gestures: {MOTION_HOLDER}",
     'output': f"One possible answer: {SCRIPT_HOLDER}"},

    {'input': f"Based on these gestures: {MOTION_HOLDER}, generate a speech script.",
     'output': f"The inferred script is: {SCRIPT_HOLDER}"},

    {'input': f"Guess the speech script from the gestures: {MOTION_HOLDER}",
     'output': f"One possible script might be: {SCRIPT_HOLDER}"},

    {'input': f"Given the gestures: {MOTION_HOLDER}, infer the corresponding speech script.",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Can you suggest a speech script from these gestures: {MOTION_HOLDER}?",
     'output': f"A likely script could be: {SCRIPT_HOLDER}"},

    {'input': f"Generate a possible script using the gestures: {MOTION_HOLDER}",
     'output': f"The possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the script that matches these gestures: {MOTION_HOLDER}",
     'output': f"One possible match is: {SCRIPT_HOLDER}"},

    {'input': f"What speech script would these gestures: {MOTION_HOLDER} suggest?",
     'output': f"A possible answer might be: {SCRIPT_HOLDER}"},

    {'input': f"Based on the gestures: {MOTION_HOLDER}, what could be the speech script?",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"From these gestures: {MOTION_HOLDER}, infer the possible script.",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Given the following gestures: {MOTION_HOLDER}, suggest a speech script.",
     'output': f"The likely script is: {SCRIPT_HOLDER}"},

    {'input': f"Can you derive a speech script from these gestures: {MOTION_HOLDER}?",
     'output': f"One possible script: {SCRIPT_HOLDER}"},

    {'input': f"Infer the possible speech script for these gestures: {MOTION_HOLDER}",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Generate a script based on the gestures: {MOTION_HOLDER}",
     'output': f"A potential answer is: {SCRIPT_HOLDER}"},

    {'input': f"What could be the speech script for these gestures: {MOTION_HOLDER}?",
     'output': f"One possible script is: {SCRIPT_HOLDER}"},

    {'input': f"From the gestures: {MOTION_HOLDER}, infer the matching speech script.",
     'output': f"The inferred script is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the speech script that aligns with these gestures: {MOTION_HOLDER}",
     'output': f"A possible script could be: {SCRIPT_HOLDER}"},

    {'input': f"Given the gestures: {MOTION_HOLDER}, deduce the speech script.",
     'output': f"One possible answer: {SCRIPT_HOLDER}"},

    {'input': f"What script might correspond to these gestures: {MOTION_HOLDER}?",
     'output': f"A likely script is: {SCRIPT_HOLDER}"},

    {'input': f"Can you infer a speech script using the gestures: {MOTION_HOLDER}?",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Based on these gestures: {MOTION_HOLDER}, guess the speech script.",
     'output': f"A possible answer is: {SCRIPT_HOLDER}"},

    {'input': f"Generate a possible script from the gestures: {MOTION_HOLDER}",
     'output': f"The likely script could be: {SCRIPT_HOLDER}"},

    {'input': f"Suggest a speech script for these gestures: {MOTION_HOLDER}",
     'output': f"One possible match is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the speech script from these gestures: {MOTION_HOLDER}",
     'output': f"A potential answer could be: {SCRIPT_HOLDER}"},

    {'input': f"What would the speech script be for these gestures: {MOTION_HOLDER}?",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"From the gestures: {MOTION_HOLDER}, what script might be inferred?",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Can you generate a speech script using these gestures: {MOTION_HOLDER}?",
     'output': f"The inferred script might be: {SCRIPT_HOLDER}"},

    {'input': f"Infer the possible speech script from the gestures: {MOTION_HOLDER}",
     'output': f"One possible script: {SCRIPT_HOLDER}"},

    {'input': f"Based on the gestures: {MOTION_HOLDER}, create a matching script.",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Generate a speech script that fits these gestures: {MOTION_HOLDER}",
     'output': f"The possible script could be: {SCRIPT_HOLDER}"},

    {'input': f"Suggest what the speech script might be for these gestures: {MOTION_HOLDER}",
     'output': f"One possible answer is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the script for the given gestures: {MOTION_HOLDER}",
     'output': f"A potential script is: {SCRIPT_HOLDER}"},

    {'input': f"What script might these gestures: {MOTION_HOLDER} represent?",
     'output': f"A possible script could be: {SCRIPT_HOLDER}"},

    {'input': f"From the gestures: {MOTION_HOLDER}, what is the likely script?",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Can you infer the speech script from these gestures: {MOTION_HOLDER}?",
     'output': f"A likely script is: {SCRIPT_HOLDER}"},

    {'input': f"Generate a possible speech script using the gestures: {MOTION_HOLDER}",
     'output': f"The inferred script is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the script that corresponds to these gestures: {MOTION_HOLDER}",
     'output': f"One possible script might be: {SCRIPT_HOLDER}"},

    {'input': f"Based on these gestures: {MOTION_HOLDER}, what could be the speech script?",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"From the gestures: {MOTION_HOLDER}, generate a possible script.",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Can you suggest a script that fits these gestures: {MOTION_HOLDER}?",
     'output': f"The likely script is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the possible script from the gestures: {MOTION_HOLDER}",
     'output': f"One possible answer is: {SCRIPT_HOLDER}"},

    {'input': f"Generate a script using these gestures: {MOTION_HOLDER}",
     'output': f"A potential script is: {SCRIPT_HOLDER}"},

    {'input': f"What might be the speech script for the gestures: {MOTION_HOLDER}?",
     'output': f"One possible script could be: {SCRIPT_HOLDER}"},

    {'input': f"Based on the gestures: {MOTION_HOLDER}, infer the script that matches.",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"From these gestures: {MOTION_HOLDER}, what speech script could be inferred?",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Can you create a script based on these gestures: {MOTION_HOLDER}?",
     'output': f"The inferred script is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the speech script that could match these gestures: {MOTION_HOLDER}",
     'output': f"A possible answer: {SCRIPT_HOLDER}"},

    {'input': f"Generate a possible speech script from these gestures: {MOTION_HOLDER}",
     'output': f"One possible script: {SCRIPT_HOLDER}"},

    {'input': f"Given the gestures: {MOTION_HOLDER}, what might the script be?",
     'output': f"A possible script could be: {SCRIPT_HOLDER}"},

    {'input': f"Infer a possible script using the gestures: {MOTION_HOLDER}",
     'output': f"The script is likely: {SCRIPT_HOLDER}"},

    {'input': f"What script might be inferred from these gestures: {MOTION_HOLDER}?",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Based on these gestures: {MOTION_HOLDER}, generate a matching speech script.",
     'output': f"The likely script: {SCRIPT_HOLDER}"},

    {'input': f"From the gestures: {MOTION_HOLDER}, infer a possible speech script.",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Can you guess the script from these gestures: {MOTION_HOLDER}?",
     'output': f"One possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the possible script based on the gestures: {MOTION_HOLDER}",
     'output': f"A potential script is: {SCRIPT_HOLDER}"},

    {'input': f"Generate a speech script that fits with these gestures: {MOTION_HOLDER}",
     'output': f"A possible answer could be: {SCRIPT_HOLDER}"},

    {'input': f"What would the speech script be that matches these gestures: {MOTION_HOLDER}?",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"From the gestures: {MOTION_HOLDER}, what could be the inferred script?",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Can you derive a script from the gestures: {MOTION_HOLDER}?",
     'output': f"The inferred script might be: {SCRIPT_HOLDER}"},

    {'input': f"Infer a script based on these gestures: {MOTION_HOLDER}",
     'output': f"One possible answer: {SCRIPT_HOLDER}"},

    {'input': f"Generate a possible script for these gestures: {MOTION_HOLDER}",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Based on the gestures: {MOTION_HOLDER}, what script could it be?",
     'output': f"A likely script is: {SCRIPT_HOLDER}"},

    {'input': f"What script might these gestures: {MOTION_HOLDER} suggest?",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"From the gestures: {MOTION_HOLDER}, what might the script be?",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"Can you infer a possible script using these gestures: {MOTION_HOLDER}?",
     'output': f"One possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Infer the possible speech script for the gestures: {MOTION_HOLDER}",
     'output': f"The inferred script is: {SCRIPT_HOLDER}"},

    {'input': f"Generate a script that matches these gestures: {MOTION_HOLDER}",
     'output': f"A possible script could be: {SCRIPT_HOLDER}"},

    {'input': f"What would the speech script be based on these gestures: {MOTION_HOLDER}?",
     'output': f"A possible script is: {SCRIPT_HOLDER}"},

    {'input': f"Based on the gestures: {MOTION_HOLDER}, infer what the script might be.",
     'output': f"The script could be: {SCRIPT_HOLDER}"},

    {'input': f"From the gestures: {MOTION_HOLDER}, generate a possible speech script.",
     'output': f"A potential script is: {SCRIPT_HOLDER}"},

    {'input': f"Can you guess the script that corresponds with these gestures: {MOTION_HOLDER}?",
     'output': f"One possible script: {SCRIPT_HOLDER}"},

    {'input': f"Infer the possible script using the gestures: {MOTION_HOLDER}",
     'output': f"The likely script is: {SCRIPT_HOLDER}"},

    {'input': f"Generate a speech script that fits the gestures: {MOTION_HOLDER}",
     'output': f"A possible answer might be: {SCRIPT_HOLDER}"}
]
