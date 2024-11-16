# script to gesture
from mmotion.utils.task.modality import SCRIPT_HOLDER, MOTION_HOLDER

S2G_TEMPLATE = [
    {'input': f"Based on this lecture script: {SCRIPT_HOLDER}, generate an appropriate speech gesture.",
     'output': f"You can accompany it with following gestures: {MOTION_HOLDER}"},
    {'input': f"Based on this lecture script: {SCRIPT_HOLDER}, generate an appropriate speech gesture.",
     'output': f"You can accompany it with the following gestures: {MOTION_HOLDER}"},

    {'input': f"From the lecture script: {SCRIPT_HOLDER}, suggest suitable gestures for the speech.",
     'output': f"The appropriate gestures to accompany it are: {MOTION_HOLDER}"},

    {'input': f"Using this script: {SCRIPT_HOLDER}, generate suitable gestures for the speech.",
     'output': f"You may use the following gestures: {MOTION_HOLDER}"},

    {'input': f"Given the lecture script: {SCRIPT_HOLDER}, what gestures would enhance the speech?",
     'output': f"Here are some suggested gestures: {MOTION_HOLDER}"},

    {'input': f"Based on the provided script: {SCRIPT_HOLDER}, suggest gestures for the speech.",
     'output': f"The following gestures would be appropriate: {MOTION_HOLDER}"},

    {'input': f"From this lecture script: {SCRIPT_HOLDER}, generate fitting speech gestures.",
     'output': f"You can use these gestures: {MOTION_HOLDER}"},

    {'input': f"Considering the script: {SCRIPT_HOLDER}, what gestures should accompany the speech?",
     'output': f"Recommended gestures are: {MOTION_HOLDER}"},

    {'input': f"Using the lecture script: {SCRIPT_HOLDER}, generate appropriate gestures.",
     'output': f"Here are the gestures to complement it: {MOTION_HOLDER}"},

    {'input': f"From this script: {SCRIPT_HOLDER}, suggest gestures to enhance the delivery.",
     'output': f"The following gestures are suggested: {MOTION_HOLDER}"},

    {'input': f"Given the lecture script: {SCRIPT_HOLDER}, what gestures would be fitting?",
     'output': f"Here are the suitable gestures: {MOTION_HOLDER}"},

    {'input': f"Based on this script: {SCRIPT_HOLDER}, generate speech gestures that match.",
     'output': f"Suggested gestures include: {MOTION_HOLDER}"},

    {'input': f"Using the lecture script: {SCRIPT_HOLDER}, what gestures can enhance the presentation?",
     'output': f"Consider these gestures: {MOTION_HOLDER}"},

    {'input': f"From this script: {SCRIPT_HOLDER}, suggest some gestures for effective delivery.",
     'output': f"The recommended gestures are: {MOTION_HOLDER}"},

    {'input': f"Considering the script: {SCRIPT_HOLDER}, what gestures would suit the speech?",
     'output': f"Here are the gestures that could work well: {MOTION_HOLDER}"},

    {'input': f"Based on the provided script: {SCRIPT_HOLDER}, generate corresponding speech gestures.",
     'output': f"The following gestures could enhance it: {MOTION_HOLDER}"},

    {'input': f"From this lecture script: {SCRIPT_HOLDER}, what gestures would fit the speech?",
     'output': f"Suggested gestures are: {MOTION_HOLDER}"},

    {'input': f"Using this script: {SCRIPT_HOLDER}, what gestures would best accompany the speech?",
     'output': f"You can use the following gestures: {MOTION_HOLDER}"},

    {'input': f"Given the lecture script: {SCRIPT_HOLDER}, generate gestures that fit the context.",
     'output': f"Here are the gestures that would complement it: {MOTION_HOLDER}"},

    {'input': f"From the script: {SCRIPT_HOLDER}, suggest suitable gestures for effective communication.",
     'output': f"The following gestures would be appropriate: {MOTION_HOLDER}"},

    {'input': f"Based on this lecture script: {SCRIPT_HOLDER}, generate suitable speech gestures.",
     'output': f"You may consider the following gestures: {MOTION_HOLDER}"},

    {'input': f"Using the provided script: {SCRIPT_HOLDER}, what gestures should accompany the speech?",
     'output': f"Recommended gestures are: {MOTION_HOLDER}"},

    {'input': f"From this lecture script: {SCRIPT_HOLDER}, generate appropriate gestures.",
     'output': f"You can enhance it with the following gestures: {MOTION_HOLDER}"},

    {'input': f"Considering the script: {SCRIPT_HOLDER}, what gestures would enhance the presentation?",
     'output': f"Here are some suggested gestures: {MOTION_HOLDER}"},

    {'input': f"Using the lecture script: {SCRIPT_HOLDER}, what gestures can enhance the delivery?",
     'output': f"Suggested gestures include: {MOTION_HOLDER}"},

    {'input': f"From this script: {SCRIPT_HOLDER}, suggest gestures to enhance the delivery.",
     'output': f"The following gestures are suggested: {MOTION_HOLDER}"},

    {'input': f"Given the lecture script: {SCRIPT_HOLDER}, what gestures would be fitting?",
     'output': f"Here are the suitable gestures: {MOTION_HOLDER}"},

    {'input': f"Based on this script: {SCRIPT_HOLDER}, generate speech gestures that match.",
     'output': f"Consider these gestures: {MOTION_HOLDER}"},

    {'input': f"Using the lecture script: {SCRIPT_HOLDER}, what gestures would suit the speech?",
     'output': f"Here are the gestures that could work well: {MOTION_HOLDER}"},

    {'input': f"From this script: {SCRIPT_HOLDER}, suggest some gestures for effective delivery.",
     'output': f"The recommended gestures are: {MOTION_HOLDER}"},

    {'input': f"Considering the script: {SCRIPT_HOLDER}, what gestures would enhance the speech?",
     'output': f"Suggested gestures are: {MOTION_HOLDER}"},

    {'input': f"Using this lecture script: {SCRIPT_HOLDER}, generate appropriate speech gestures.",
     'output': f"You can accompany it with the following gestures: {MOTION_HOLDER}"},

    {'input': f"From the script: {SCRIPT_HOLDER}, suggest gestures to match the speech.",
     'output': f"Here are the gestures that would complement it: {MOTION_HOLDER}"},

    {'input': f"Given this script: {SCRIPT_HOLDER}, what gestures would fit well?",
     'output': f"Consider these gestures: {MOTION_HOLDER}"},

    {'input': f"Based on the lecture script: {SCRIPT_HOLDER}, generate appropriate gestures.",
     'output': f"The following gestures could enhance it: {MOTION_HOLDER}"},

    {'input': f"Using the provided script: {SCRIPT_HOLDER}, what gestures would best suit the speech?",
     'output': f"Suggested gestures are: {MOTION_HOLDER}"},

    {'input': f"From this lecture script: {SCRIPT_HOLDER}, generate fitting gestures.",
     'output': f"You may consider the following gestures: {MOTION_HOLDER}"},

    {'input': f"Considering the script: {SCRIPT_HOLDER}, what gestures would enhance the communication?",
     'output': f"Here are the gestures that could work well: {MOTION_HOLDER}"},

    {'input': f"Using the lecture script: {SCRIPT_HOLDER}, generate appropriate gestures.",
     'output': f"Recommended gestures include: {MOTION_HOLDER}"},

    {'input': f"From this script: {SCRIPT_HOLDER}, what gestures should accompany the speech?",
     'output': f"The following gestures would be appropriate: {MOTION_HOLDER}"},

    {'input': f"Based on this lecture script: {SCRIPT_HOLDER}, suggest fitting speech gestures.",
     'output': f"You can use these gestures: {MOTION_HOLDER}"},

    {'input': f"Given this script: {SCRIPT_HOLDER}, what gestures would you suggest?",
     'output': f"Here are some suggested gestures: {MOTION_HOLDER}"},

    {'input': f"Using the provided script: {SCRIPT_HOLDER}, what gestures would enhance the delivery?",
     'output': f"Consider these gestures: {MOTION_HOLDER}"},

    {'input': f"From this lecture script: {SCRIPT_HOLDER}, generate gestures to match the speech.",
     'output': f"The appropriate gestures to accompany it are: {MOTION_HOLDER}"},

    {'input': f"Considering the script: {SCRIPT_HOLDER}, what gestures can enhance the communication?",
     'output': f"Here are the gestures that could work well: {MOTION_HOLDER}"},

    {'input': f"Using the lecture script: {SCRIPT_HOLDER}, generate appropriate gestures.",
     'output': f"Suggested gestures include: {MOTION_HOLDER}"},

    {'input': f"From this script: {SCRIPT_HOLDER}, suggest gestures to enhance the delivery.",
     'output': f"The recommended gestures are: {MOTION_HOLDER}"},

    {'input': f"Given the lecture script: {SCRIPT_HOLDER}, what gestures would be fitting?",
     'output': f"Here are the suitable gestures: {MOTION_HOLDER}"},

    {'input': f"Based on this script: {SCRIPT_HOLDER}, generate speech gestures that match.",
     'output': f"You may consider the following gestures: {MOTION_HOLDER}"},

    {'input': f"Using the lecture script: {SCRIPT_HOLDER}, what gestures would suit the speech?",
     'output': f"Here are the gestures that could work well: {MOTION_HOLDER}"},

    {'input': f"From this script: {SCRIPT_HOLDER}, suggest some gestures for effective delivery.",
     'output': f"The recommended gestures are: {MOTION_HOLDER}"},

    {'input': f"Considering the script: {SCRIPT_HOLDER}, what gestures would enhance the speech?",
     'output': f"Suggested gestures are: {MOTION_HOLDER}"},

    {'input': f"Using this lecture script: {SCRIPT_HOLDER}, generate appropriate speech gestures.",
     'output': f"You can accompany it with the following gestures: {MOTION_HOLDER}"},

    {'input': f"From the script: {SCRIPT_HOLDER}, suggest gestures to match the speech.",
     'output': f"Here are the gestures that would complement it: {MOTION_HOLDER}"},

    {'input': f"Given this script: {SCRIPT_HOLDER}, what gestures would fit well?",
     'output': f"Consider these gestures: {MOTION_HOLDER}"},

    {'input': f"Based on the lecture script: {SCRIPT_HOLDER}, generate appropriate gestures.",
     'output': f"The following gestures could enhance it: {MOTION_HOLDER}"},

    {'input': f"Using the provided script: {SCRIPT_HOLDER}, what gestures would best suit the speech?",
     'output': f"Suggested gestures are: {MOTION_HOLDER}"},

    {'input': f"From this lecture script: {SCRIPT_HOLDER}, generate fitting gestures.",
     'output': f"You may consider the following gestures: {MOTION_HOLDER}"},

    {'input': f"Considering the script: {SCRIPT_HOLDER}, what gestures would enhance the communication?",
     'output': f"Here are the gestures that could work well: {MOTION_HOLDER}"},

    {'input': f"Using the lecture script: {SCRIPT_HOLDER}, generate appropriate gestures.",
     'output': f"Recommended gestures include: {MOTION_HOLDER}"},

    {'input': f"From this script: {SCRIPT_HOLDER}, what gestures should accompany the speech?",
     'output': f"The following gestures would be appropriate: {MOTION_HOLDER}"},

    {'input': f"Create a gesture sequence based on this lecture script: {SCRIPT_HOLDER}.",
     'output': f"You may enhance it with the following gestures: {MOTION_HOLDER}"}
]
