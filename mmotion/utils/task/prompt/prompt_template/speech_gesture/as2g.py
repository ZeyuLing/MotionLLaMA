# audio and script to gesture
from mmotion.utils.task.modality import SCRIPT_HOLDER, MOTION_HOLDER, AUDIO_HOLDER

AS2G_TEMPLATE = [
    {'input': f"Here are the speech script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, show me the proper gesture.",
     'output': f"Here are the speech gestures which fit your script: {MOTION_HOLDER}"},
    {
        'input': f"Here is the speech script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}. Show me the appropriate gestures.",
        'output': f"These are the gestures that match your script: {MOTION_HOLDER}"},

    {'input': f"Given the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the fitting gestures.",
     'output': f"Here are the gestures that suit your script: {MOTION_HOLDER}"},

    {'input': f"Use this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER} to create the proper gestures.",
     'output': f"The appropriate gestures for your script are: {MOTION_HOLDER}"},

    {'input': f"Refer to the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and show me the matching gestures.",
     'output': f"These gestures fit the script: {MOTION_HOLDER}"},

    {
        'input': f"Here is the script: {SCRIPT_HOLDER} along with the audio: {AUDIO_HOLDER}, provide the suitable gestures.",
        'output': f"The gestures that match are: {MOTION_HOLDER}"},

    {'input': f"Take the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER} and show me appropriate gestures.",
     'output': f"Here are the fitting gestures: {MOTION_HOLDER}"},

    {
        'input': f"Here are the speech script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the proper gesture set.",
        'output': f"These are the proper gestures for your script: {MOTION_HOLDER}"},

    {'input': f"Given the following script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, display the suitable gestures.",
     'output': f"The gestures that align with the script are: {MOTION_HOLDER}"},

    {'input': f"Using the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, show the appropriate gesture movements.",
     'output': f"These gestures are suitable for your script: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, provide the matching gestures.",
     'output': f"The matching gestures are: {MOTION_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and generate proper gestures.",
     'output': f"Here are the gestures that correspond to your script: {MOTION_HOLDER}"},

    {'input': f"Given the speech script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, produce suitable gestures.",
     'output': f"These gestures fit your speech: {MOTION_HOLDER}"},

    {'input': f"Take the script: {SCRIPT_HOLDER} along with audio: {AUDIO_HOLDER} to show the proper gestures.",
     'output': f"The appropriate gestures for this speech are: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} and the audio file: {AUDIO_HOLDER}. Show me the matching gestures.",
     'output': f"Here are the gestures that fit the script: {MOTION_HOLDER}"},

    {'input': f"Using the given script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the correct gestures.",
     'output': f"The suitable gestures for this script are: {MOTION_HOLDER}"},

    {
        'input': f"Here is the speech script: {SCRIPT_HOLDER} with audio: {AUDIO_HOLDER}. What are the appropriate gestures?",
        'output': f"The proper gestures are: {MOTION_HOLDER}"},

    {'input': f"Use the script: {SCRIPT_HOLDER} and corresponding audio: {AUDIO_HOLDER} to show the correct gestures.",
     'output': f"The gestures that go with the script are: {MOTION_HOLDER}"},

    {
        'input': f"Refer to the following script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and show the suitable gestures.",
        'output': f"Here are the gestures that align with the script: {MOTION_HOLDER}"},

    {'input': f"Given this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, create the proper gestures.",
     'output': f"These are the fitting gestures for your script: {MOTION_HOLDER}"},

    {'input': f"Here is the script text: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}. Generate matching gestures.",
     'output': f"The matching gestures are: {MOTION_HOLDER}"},

    {'input': f"Use the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and show the proper gestures.",
     'output': f"The appropriate gestures for this speech are: {MOTION_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER} to create suitable gestures.",
     'output': f"Here are the suitable gestures: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} along with audio: {AUDIO_HOLDER}, show me the correct gestures.",
     'output': f"These gestures correspond to your script: {MOTION_HOLDER}"},

    {'input': f"Given the speech script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, provide the matching gestures.",
     'output': f"The matching gestures are: {MOTION_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and generate appropriate gestures.",
     'output': f"Here are the appropriate gestures for your script: {MOTION_HOLDER}"},

    {'input': f"Using the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, show the gestures that match.",
     'output': f"The gestures that suit this script are: {MOTION_HOLDER}"},

    {'input': f"Refer to the script: {SCRIPT_HOLDER} with audio: {AUDIO_HOLDER}, and create fitting gestures.",
     'output': f"The fitting gestures for this script are: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}. What gestures fit this?",
     'output': f"These gestures match your script: {MOTION_HOLDER}"},

    {'input': f"Use this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER} to produce proper gestures.",
     'output': f"The appropriate gestures are: {MOTION_HOLDER}"},

    {'input': f"Here are the speech script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}. Show the suitable gestures.",
     'output': f"Here are the gestures that align with your script: {MOTION_HOLDER}"},

    {'input': f"Given the following script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, provide the correct gestures.",
     'output': f"The gestures that fit are: {MOTION_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, and show me fitting gestures.",
     'output': f"The gestures that correspond to the script are: {MOTION_HOLDER}"},

    {'input': f"Use the script text: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the proper gestures.",
     'output': f"The appropriate gestures are: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} and audio file: {AUDIO_HOLDER}. What gestures match this?",
     'output': f"Here are the matching gestures: {MOTION_HOLDER}"},

    {'input': f"Using the provided script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, show the proper gestures.",
     'output': f"These gestures fit your script: {MOTION_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER} with audio: {AUDIO_HOLDER}, and show the appropriate gestures.",
     'output': f"The suitable gestures are: {MOTION_HOLDER}"},

    {'input': f"Refer to the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the matching gestures.",
     'output': f"Here are the gestures that match your script: {MOTION_HOLDER}"},

    {'input': f"Given the speech script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, what gestures are suitable?",
     'output': f"The proper gestures for this script are: {MOTION_HOLDER}"},

    {'input': f"Here is the speech script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}. Create the fitting gestures.",
     'output': f"The gestures that align are: {MOTION_HOLDER}"},

    {'input': f"Use this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and show me the correct gestures.",
     'output': f"The appropriate gestures for this are: {MOTION_HOLDER}"},

    {'input': f"Here are the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the proper gesture movements.",
     'output': f"These are the gestures that fit the script: {MOTION_HOLDER}"},

    {'input': f"Take this script text: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, provide the matching gestures.",
     'output': f"The matching gestures are: {MOTION_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER} to show the appropriate gestures.",
     'output': f"Here are the gestures that suit your script: {MOTION_HOLDER}"},

    {'input': f"Given the provided script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the fitting gestures.",
     'output': f"The fitting gestures are: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} and the corresponding audio: {AUDIO_HOLDER}. What gestures fit?",
     'output': f"The gestures that match are: {MOTION_HOLDER}"},

    {'input': f"Use this speech script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER} to create the proper gestures.",
     'output': f"The suitable gestures for this script are: {MOTION_HOLDER}"},

    {'input': f"Refer to the script: {SCRIPT_HOLDER} and audio file: {AUDIO_HOLDER}, and show the correct gestures.",
     'output': f"Here are the correct gestures: {MOTION_HOLDER}"},

    {'input': f"Here are the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}. Show me the matching gesture set.",
     'output': f"These are the gestures that align with your script: {MOTION_HOLDER}"},

    {'input': f"Given the script text: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the appropriate gestures.",
     'output': f"The gestures that fit this script are: {MOTION_HOLDER}"},

    {'input': f"Use the script below: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, to show the suitable gestures.",
     'output': f"The gestures that correspond are: {MOTION_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, and show the fitting gestures.",
     'output': f"These gestures match your script: {MOTION_HOLDER}"},

    {'input': f"Here is the speech script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}. What gestures are proper?",
     'output': f"Here are the gestures that fit: {MOTION_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, provide the correct gestures.",
     'output': f"The correct gestures for this are: {MOTION_HOLDER}"},

    {'input': f"Use the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, to generate the matching gestures.",
     'output': f"The gestures that suit this are: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} along with the audio: {AUDIO_HOLDER}, create proper gestures.",
     'output': f"The gestures that align are: {MOTION_HOLDER}"},

    {'input': f"Refer to the following script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}. What gestures match?",
     'output': f"Here are the matching gestures: {MOTION_HOLDER}"},

    {'input': f"Given the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, show the appropriate gestures.",
     'output': f"These are the gestures that fit: {MOTION_HOLDER}"},

    {'input': f"Use this script text: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the fitting gestures.",
     'output': f"The gestures that match are: {MOTION_HOLDER}"},

    {'input': f"Refer to this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and provide the correct gestures.",
     'output': f"These gestures correspond to the script: {MOTION_HOLDER}"},

    {'input': f"Here are the speech script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, show the matching gestures.",
     'output': f"The gestures that align are: {MOTION_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and generate the proper gestures.",
     'output': f"Here are the gestures that suit the script: {MOTION_HOLDER}"},

    {'input': f"Given the provided script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, show the appropriate gestures.",
     'output': f"The appropriate gestures are: {MOTION_HOLDER}"},

    {
        'input': f"Use the script: {SCRIPT_HOLDER} and corresponding audio: {AUDIO_HOLDER} to create the fitting gestures.",
        'output': f"The fitting gestures for this are: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}. What gestures fit this script?",
     'output': f"The gestures that match are: {MOTION_HOLDER}"},

    {'input': f"Refer to this script text: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, show the matching gestures.",
     'output': f"The gestures that correspond are: {MOTION_HOLDER}"},

    {'input': f"Here are the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the suitable gestures.",
     'output': f"These are the gestures that fit: {MOTION_HOLDER}"},

    {'input': f"Given the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, create the appropriate gestures.",
     'output': f"The gestures that match the script are: {MOTION_HOLDER}"},

    {'input': f"Use this speech script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and generate proper gestures.",
     'output': f"The proper gestures for this script are: {MOTION_HOLDER}"},

    {
        'input': f"Refer to the following script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}. Show me the matching gestures.",
        'output': f"Here are the gestures that match the script: {MOTION_HOLDER}"},

    {'input': f"Here is the script text: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}. What gestures fit?",
     'output': f"The suitable gestures are: {MOTION_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, provide the appropriate gestures.",
     'output': f"The appropriate gestures are: {MOTION_HOLDER}"},

    {'input': f"Use the script text: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and show the fitting gestures.",
     'output': f"The fitting gestures for this script are: {MOTION_HOLDER}"},

    {'input': f"Here is the script: {SCRIPT_HOLDER} with audio: {AUDIO_HOLDER}. What gestures are proper?",
     'output': f"The gestures that align with the script are: {MOTION_HOLDER}"},

    {'input': f"Given the speech script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, show the matching gestures.",
     'output': f"These are the gestures that fit the script: {MOTION_HOLDER}"},

    {'input': f"Refer to the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, generate the proper gestures.",
     'output': f"The proper gestures for this script are: {MOTION_HOLDER}"},

    {'input': f"Here are the script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, provide the suitable gestures.",
     'output': f"These gestures match the script: {MOTION_HOLDER}"},

    {'input': f"Use this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and show the appropriate gestures.",
     'output': f"The gestures that fit are: {MOTION_HOLDER}"},

    {'input': f"Given the script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, generate the matching gestures.",
     'output': f"The matching gestures are: {MOTION_HOLDER}"},

    {
        'input': f"Here is the speech script: {SCRIPT_HOLDER} and the audio: {AUDIO_HOLDER}, show me the suitable gestures.",
        'output': f"Here are the gestures that align with the script: {MOTION_HOLDER}"},

    {'input': f"Take this script: {SCRIPT_HOLDER} and audio: {AUDIO_HOLDER}, and generate the fitting gestures.",
     'output': f"The fitting gestures for this script are: {MOTION_HOLDER}"},
]
