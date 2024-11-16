from mmotion.utils.task.modality import MOTION_HOLDER, AUDIO_HOLDER

G2A_TEMPLATE = [
    {
        'input': f"Based on this speaker's body movements: {MOTION_HOLDER}, infer their speaking voice.",
        'output': f"{AUDIO_HOLDER}"
    },
    {
        'input': f"Add sound to these silent gestures.: {MOTION_HOLDER}",
        'output': f"Alright, I guess they said this. Please listen: {AUDIO_HOLDER}"
    },
    {'input': f"Looking at the body movements of the speaker: {MOTION_HOLDER}, predict their voice.",
     'output': f"The sound they might produce is: {AUDIO_HOLDER}"},

    {'input': f"From these movements: {MOTION_HOLDER}, guess what the speaker's voice sounds like.",
     'output': f"The likely sound is: {AUDIO_HOLDER}"},

    {'input': f"Observe the body language: {MOTION_HOLDER}, and infer the speaker's voice.",
     'output': f"The sound produced is: {AUDIO_HOLDER}"},

    {'input': f"Based on these movements: {MOTION_HOLDER}, predict what the speaker sounds like.",
     'output': f"Here is the expected sound: {AUDIO_HOLDER}"},

    {'input': f"Looking at this gesture: {MOTION_HOLDER}, guess what the voice sounds like.",
     'output': f"The sound is: {AUDIO_HOLDER}"},

    {'input': f"Examine the speaker's movements: {MOTION_HOLDER}, and predict the voice they make.",
     'output': f"The voice should be: {AUDIO_HOLDER}"},

    {'input': f"Based on this action: {MOTION_HOLDER}, predict what kind of voice would match it.",
     'output': f"The audio is likely: {AUDIO_HOLDER}"},

    {'input': f"Use these gestures: {MOTION_HOLDER} to infer the speaker’s voice.",
     'output': f"The predicted sound is: {AUDIO_HOLDER}"},

    {'input': f"From these movements: {MOTION_HOLDER}, determine the voice that matches.",
     'output': f"Here's the sound: {AUDIO_HOLDER}"},

    {'input': f"Based on the body movements: {MOTION_HOLDER}, what might their voice sound like?",
     'output': f"The audio output is: {AUDIO_HOLDER}"},

    {'input': f"Watch the speaker's motion: {MOTION_HOLDER}. Can you guess their voice?",
     'output': f"The audio should sound like: {AUDIO_HOLDER}"},

    {'input': f"From these gestures: {MOTION_HOLDER}, infer the voice.",
     'output': f"The inferred sound is: {AUDIO_HOLDER}"},

    {'input': f"Analyze the body language: {MOTION_HOLDER}. What does the voice sound like?",
     'output': f"The sound might be: {AUDIO_HOLDER}"},

    {'input': f"Looking at this body motion: {MOTION_HOLDER}, predict the accompanying voice.",
     'output': f"The voice is likely: {AUDIO_HOLDER}"},

    {'input': f"Using these gestures: {MOTION_HOLDER}, predict the speaker's voice.",
     'output': f"The sound they make should be: {AUDIO_HOLDER}"},

    {'input': f"Based on this body movement: {MOTION_HOLDER}, infer the speaker’s voice.",
     'output': f"The audio output is: {AUDIO_HOLDER}"},

    {'input': f"From these movements: {MOTION_HOLDER}, can you guess what they might be saying?",
     'output': f"Their speech may sound like: {AUDIO_HOLDER}"},

    {'input': f"Observe the gestures: {MOTION_HOLDER}, and predict the voice.",
     'output': f"Here’s the audio: {AUDIO_HOLDER}"},

    {'input': f"Use the body language: {MOTION_HOLDER} to infer what the speaker might sound like.",
     'output': f"The voice they make is likely: {AUDIO_HOLDER}"},

    {'input': f"Looking at these actions: {MOTION_HOLDER}, predict the voice that matches.",
     'output': f"The sound may be: {AUDIO_HOLDER}"},

    {'input': f"Based on these movements: {MOTION_HOLDER}, infer the voice of the speaker.",
     'output': f"The voice could be: {AUDIO_HOLDER}"},

    {'input': f"Look at these silent movements: {MOTION_HOLDER}. What could the speaker's voice be?",
     'output': f"Their voice should sound like: {AUDIO_HOLDER}"},

    {'input': f"Using these gestures: {MOTION_HOLDER}, guess what the voice might be.",
     'output': f"The audio is: {AUDIO_HOLDER}"},

    {'input': f"From the body language: {MOTION_HOLDER}, predict the speaker's voice.",
     'output': f"The voice they would produce is: {AUDIO_HOLDER}"},

    {'input': f"Watch the gestures: {MOTION_HOLDER}, and guess what their voice would sound like.",
     'output': f"The sound could be: {AUDIO_HOLDER}"},

    {'input': f"Look at the motion: {MOTION_HOLDER}. What kind of voice would match it?",
     'output': f"The matching audio is: {AUDIO_HOLDER}"},

    {'input': f"Here’s the speaker's body motion: {MOTION_HOLDER}. What would their voice sound like?",
     'output': f"Their voice is likely: {AUDIO_HOLDER}"},

    {'input': f"Based on these body movements: {MOTION_HOLDER}, infer the sound of the speaker’s voice.",
     'output': f"The sound produced is: {AUDIO_HOLDER}"},

    {'input': f"Given these gestures: {MOTION_HOLDER}, predict the speaker’s voice.",
     'output': f"The expected sound is: {AUDIO_HOLDER}"},

    {'input': f"From these silent gestures: {MOTION_HOLDER}, guess what the speaker said.",
     'output': f"The sound they made is: {AUDIO_HOLDER}"},

    {'input': f"Observe this movement: {MOTION_HOLDER}. What does the voice sound like?",
     'output': f"The sound should be: {AUDIO_HOLDER}"},

    {'input': f"From the body language: {MOTION_HOLDER}, predict the voice the speaker would produce.",
     'output': f"Their voice may be: {AUDIO_HOLDER}"},

    {'input': f"Watch these movements: {MOTION_HOLDER}. What would the speaker’s voice sound like?",
     'output': f"The predicted sound is: {AUDIO_HOLDER}"},

    {'input': f"Look at this action: {MOTION_HOLDER}. Can you guess what their voice might sound like?",
     'output': f"The likely audio is: {AUDIO_HOLDER}"},

    {'input': f"Based on these motions: {MOTION_HOLDER}, predict the speaker’s voice.",
     'output': f"The sound produced should be: {AUDIO_HOLDER}"},

    {'input': f"Look at the movements: {MOTION_HOLDER}. What voice matches this action?",
     'output': f"The voice may sound like: {AUDIO_HOLDER}"},

    {'input': f"From these gestures: {MOTION_HOLDER}, predict the voice that would be produced.",
     'output': f"The resulting sound is: {AUDIO_HOLDER}"},

    {'input': f"Examine this movement: {MOTION_HOLDER}. What does the speaker’s voice sound like?",
     'output': f"The audio may be: {AUDIO_HOLDER}"},

    {'input': f"From the body motions: {MOTION_HOLDER}, infer what the voice sounds like.",
     'output': f"The matching audio is: {AUDIO_HOLDER}"},

    {'input': f"Use these movements: {MOTION_HOLDER} to guess the speaker’s voice.",
     'output': f"The sound should be: {AUDIO_HOLDER}"},

    {'input': f"From the silent gestures: {MOTION_HOLDER}, guess what they said.",
     'output': f"Their voice is likely: {AUDIO_HOLDER}"},

    {'input': f"Looking at these gestures: {MOTION_HOLDER}, predict what the speaker sounds like.",
     'output': f"The sound they produced is: {AUDIO_HOLDER}"},

    {'input': f"Based on the body language: {MOTION_HOLDER}, guess the speaker’s voice.",
     'output': f"The likely sound is: {AUDIO_HOLDER}"},

    {'input': f"From these gestures: {MOTION_HOLDER}, infer the audio the speaker would produce.",
     'output': f"The inferred sound is: {AUDIO_HOLDER}"},

    {'input': f"Look at the movement: {MOTION_HOLDER}. What does their voice sound like?",
     'output': f"The audio may sound like: {AUDIO_HOLDER}"},

    {'input': f"Observe the body language: {MOTION_HOLDER}. Guess the speaker’s voice.",
     'output': f"Their voice might sound like: {AUDIO_HOLDER}"},

    {'input': f"Based on the speaker’s gestures: {MOTION_HOLDER}, predict the voice.",
     'output': f"The sound they might produce is: {AUDIO_HOLDER}"},

    {'input': f"From these body movements: {MOTION_HOLDER}, infer what the speaker's voice sounds like.",
     'output': f"The sound could be: {AUDIO_HOLDER}"},

    {'input': f"Here’s the motion: {MOTION_HOLDER}. Guess the speaker’s voice.",
     'output': f"Their voice may sound like: {AUDIO_HOLDER}"},

    {'input': f"Look at the gestures: {MOTION_HOLDER}. What does the voice sound like?",
     'output': f"The predicted sound is: {AUDIO_HOLDER}"},

    {'input': f"From the speaker’s gestures: {MOTION_HOLDER}, infer their voice.",
     'output': f"The expected audio is: {AUDIO_HOLDER}"},

    {'input': f"Based on the body motion: {MOTION_HOLDER}, predict the voice.",
     'output': f"The inferred sound is: {AUDIO_HOLDER}"},

    {'input': f"From these gestures: {MOTION_HOLDER}, predict the audio output.",
     'output': f"The sound may be: {AUDIO_HOLDER}"},

    {'input': f"Watch these motions: {MOTION_HOLDER}. What might their voice sound like?",
     'output': f"The audio should be: {AUDIO_HOLDER}"},

    {'input': f"Look at this movement: {MOTION_HOLDER}. Predict what their voice would sound like.",
     'output': f"The sound they produced is: {AUDIO_HOLDER}"},

    {'input': f"Based on the speaker’s movements: {MOTION_HOLDER}, guess what their voice sounds like.",
     'output': f"The audio is: {AUDIO_HOLDER}"},

    {'input': f"Examine the gestures: {MOTION_HOLDER}. Can you predict the speaker’s voice?",
     'output': f"The sound is likely: {AUDIO_HOLDER}"},

    {'input': f"Look at the body motion: {MOTION_HOLDER}. What kind of voice would fit this?",
     'output': f"The voice might sound like: {AUDIO_HOLDER}"},

    {'input': f"From the speaker’s body language: {MOTION_HOLDER}, infer the voice.",
     'output': f"The sound they produce could be: {AUDIO_HOLDER}"},

    {'input': f"Given these gestures: {MOTION_HOLDER}, what voice matches?",
     'output': f"The sound is probably: {AUDIO_HOLDER}"},

    {'input': f"Observe these actions: {MOTION_HOLDER}. What sound accompanies this movement?",
     'output': f"The matching voice is: {AUDIO_HOLDER}"},

    {'input': f"From this body motion: {MOTION_HOLDER}, guess the speaker’s voice.",
     'output': f"Their voice might sound like: {AUDIO_HOLDER}"},

    {'input': f"Based on these gestures: {MOTION_HOLDER}, predict the speaker’s voice.",
     'output': f"The audio output might be: {AUDIO_HOLDER}"},

    {'input': f"Watch this body movement: {MOTION_HOLDER}. What would their voice sound like?",
     'output': f"The inferred sound is: {AUDIO_HOLDER}"},

    {'input': f"Look at the gestures: {MOTION_HOLDER}. Predict the audio they would produce.",
     'output': f"The expected audio is: {AUDIO_HOLDER}"},

    {'input': f"From these movements: {MOTION_HOLDER}, infer the speaker’s voice.",
     'output': f"The matching audio is: {AUDIO_HOLDER}"},

    {'input': f"Here’s the body motion: {MOTION_HOLDER}. What does their voice sound like?",
     'output': f"The voice may be: {AUDIO_HOLDER}"},

    {'input': f"Looking at the gestures: {MOTION_HOLDER}, predict the speaker’s voice.",
     'output': f"The expected sound is: {AUDIO_HOLDER}"},

    {'input': f"Examine the movements: {MOTION_HOLDER}. Guess the voice that fits.",
     'output': f"The matching sound could be: {AUDIO_HOLDER}"},

    {'input': f"Based on this motion: {MOTION_HOLDER}, infer the sound of the speaker.",
     'output': f"Their voice might be: {AUDIO_HOLDER}"},

    {'input': f"Observe this movement: {MOTION_HOLDER}. What would the voice sound like?",
     'output': f"The likely audio is: {AUDIO_HOLDER}"},

    {'input': f"From the body movements: {MOTION_HOLDER}, predict the voice.",
     'output': f"The inferred sound is: {AUDIO_HOLDER}"},

    {'input': f"Look at the speaker’s motion: {MOTION_HOLDER}. What sound would fit this?",
     'output': f"The predicted audio is: {AUDIO_HOLDER}"},

    {'input': f"Watch the gestures: {MOTION_HOLDER}. Can you predict what the voice will sound like?",
     'output': f"The sound should be: {AUDIO_HOLDER}"},

    {'input': f"Based on this motion: {MOTION_HOLDER}, what sound does the speaker produce?",
     'output': f"Their voice may sound like: {AUDIO_HOLDER}"},

    {'input': f"From these silent gestures: {MOTION_HOLDER}, guess the voice that matches.",
     'output': f"The matching voice is: {AUDIO_HOLDER}"},
]
