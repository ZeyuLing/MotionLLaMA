from mmotion.utils.task.modality import UNION_CAPTION_HOLDER, MOTION_HOLDER, INTERACTOR_MOTION_HOLDER, CAPTION_HOLDER, \
    INTERACTOR_CAPTION_HOLDER

IN2USTM_TEMPLATE = [
    {
        'input': f"Randomly generate a paired movement. First, provide an overall description of the movement, then describe each person's actions separately.",
        'output': f"The motion is: {UNION_CAPTION_HOLDER}. The first person is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},
    {
        'input': f"Generate a paired movement. First, tell me what each person’s actions are separately, then show me what the movement looks like and provide an overall description.",
        'output': f"The first person's action is: {CAPTION_HOLDER} and it looks like this: {MOTION_HOLDER}. "
                  f"The second person's action is: {INTERACTOR_CAPTION_HOLDER} and it looks like this: {INTERACTOR_MOTION_HOLDER}."
                  f" This movement can be described overall as: {UNION_CAPTION_HOLDER}"},
    {
        'input': f"Randomly generate a paired movement. First, provide an overall description of the movement, then describe each person's actions separately.",
        'output': f"The motion is: {UNION_CAPTION_HOLDER}. The first person is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a paired movement. First, tell me what each person’s actions are separately, then show me what the movement looks like and provide an overall description.",
        'output': f"The first person's action is: {CAPTION_HOLDER} and it looks like this: {MOTION_HOLDER}. "
                  f"The second person's action is: {INTERACTOR_CAPTION_HOLDER} and it looks like this: {INTERACTOR_MOTION_HOLDER}."
                  f" This movement can be described overall as: {UNION_CAPTION_HOLDER}"},

    {
        'input': f"Create a paired movement. Describe the overall action first, then detail each person's movements separately.",
        'output': f"This action is: {UNION_CAPTION_HOLDER}. First person's action: {CAPTION_HOLDER} {MOTION_HOLDER}. Second person's action: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a paired movement description. Start with an overall description, then specify individual actions.",
        'output': f"Overall, the motion is: {UNION_CAPTION_HOLDER}. The first individual does: {CAPTION_HOLDER} {MOTION_HOLDER}. The second individual does: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly create a paired movement. Provide an overall motion description followed by individual actions.",
        'output': f"The complete motion is: {UNION_CAPTION_HOLDER}. For the first person, it’s: {CAPTION_HOLDER} {MOTION_HOLDER}. For the second person, it’s: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a paired movement. First, give an overall description, then describe each person's movements individually.",
        'output': f"The action can be summarized as: {UNION_CAPTION_HOLDER}. The first person's action involves: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person acts as: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Create a paired movement. Start with a description of the overall motion and detail each person's actions.",
        'output': f"Overall motion description: {UNION_CAPTION_HOLDER}. First person action: {CAPTION_HOLDER} {MOTION_HOLDER}. Second person action: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly generate a paired movement. First, describe the overall movement, then detail each participant's actions.",
        'output': f"The complete movement is: {UNION_CAPTION_HOLDER}. The first participant's action is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second participant's action is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a description of paired movements. Start with an overall description, then specify individual actions.",
        'output': f"This motion is: {UNION_CAPTION_HOLDER}. The action of the first person is: {CAPTION_HOLDER} {MOTION_HOLDER}. The action of the second person is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Create a paired movement. Provide an overall description first, then explain each individual's actions.",
        'output': f"The movement can be described as: {UNION_CAPTION_HOLDER}. First individual: {CAPTION_HOLDER} {MOTION_HOLDER}. Second individual: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly create a paired movement. First, give an overall motion description, then detail the actions of each person.",
        'output': f"Overall, the motion is: {UNION_CAPTION_HOLDER}. The first person's action: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person's action: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a paired movement. First, summarize the overall action, then describe each person's actions.",
     'output': f"The motion is: {UNION_CAPTION_HOLDER}. The first person does: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person performs: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Create a paired movement description. Start with an overall action summary, then detail individual movements.",
        'output': f"Overall action: {UNION_CAPTION_HOLDER}. First person action: {CAPTION_HOLDER} {MOTION_HOLDER}. Second person action: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly generate a description of a paired movement. Begin with an overall description, then detail actions separately.",
        'output': f"The complete action is: {UNION_CAPTION_HOLDER}. The first participant's action is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second participant's action is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a description of paired movements. Start with an overall summary, then specify individual actions.",
        'output': f"This motion can be described as: {UNION_CAPTION_HOLDER}. The first person's action: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person's action: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Create a paired movement description. Provide an overall action summary first, then detail each person's actions.",
        'output': f"The movement is: {UNION_CAPTION_HOLDER}. The first individual acts: {CAPTION_HOLDER} {MOTION_HOLDER}. The second individual acts: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly create a paired movement. First, describe the overall movement, then detail each participant's actions separately.",
        'output': f"The overall motion is: {UNION_CAPTION_HOLDER}. The first participant performs: {CAPTION_HOLDER} {MOTION_HOLDER}. The second participant performs: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a paired movement description. Start with an overall action summary, then specify individual actions.",
        'output': f"This action is: {UNION_CAPTION_HOLDER}. The first person's action is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person's action is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Create a paired movement. Begin with a description of the overall action, then detail the actions of each participant.",
        'output': f"Overall, the action is: {UNION_CAPTION_HOLDER}. The first person does: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person does: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly generate a paired movement. Provide an overall description first, then describe each person's actions separately.",
        'output': f"The motion is: {UNION_CAPTION_HOLDER}. The first person is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a paired movement. First, tell me what each person’s actions are separately, then show me what the movement looks like and provide an overall description.",
        'output': f"The first person's action is: {CAPTION_HOLDER} and it looks like this: {MOTION_HOLDER}. "
                  f"The second person's action is: {INTERACTOR_CAPTION_HOLDER} and it looks like this: {INTERACTOR_MOTION_HOLDER}."
                  f" This movement can be described overall as: {UNION_CAPTION_HOLDER}"},

    {
        'input': f"Create a paired movement. Provide an overall description, then detail each participant's actions separately.",
        'output': f"This action is: {UNION_CAPTION_HOLDER}. First person's action: {CAPTION_HOLDER} {MOTION_HOLDER}. Second person's action: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a paired movement description. Start with an overall description, then specify individual actions.",
        'output': f"Overall, the motion is: {UNION_CAPTION_HOLDER}. The first individual does: {CAPTION_HOLDER} {MOTION_HOLDER}. The second individual does: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly create a paired movement. First, give an overall motion description, then detail the actions of each person.",
        'output': f"The complete motion is: {UNION_CAPTION_HOLDER}. For the first person, it’s: {CAPTION_HOLDER} {MOTION_HOLDER}. For the second person, it’s: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Generate a paired movement. First, summarize the overall action, then describe each person's actions.",
     'output': f"The motion is: {UNION_CAPTION_HOLDER}. The first person does: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person performs: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Create a paired movement description. Start with an overall action summary, then detail individual movements.",
        'output': f"Overall action: {UNION_CAPTION_HOLDER}. First person action: {CAPTION_HOLDER} {MOTION_HOLDER}. Second person action: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly generate a description of a paired movement. Begin with an overall description, then detail actions separately.",
        'output': f"The complete action is: {UNION_CAPTION_HOLDER}. The first participant's action is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second participant's action is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a description of paired movements. Start with an overall summary, then specify individual actions.",
        'output': f"This motion can be described as: {UNION_CAPTION_HOLDER}. The first person's action: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person's action: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Create a paired movement description. Provide an overall action summary first, then detail each person's actions.",
        'output': f"The movement is: {UNION_CAPTION_HOLDER}. The first individual acts: {CAPTION_HOLDER} {MOTION_HOLDER}. The second individual acts: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly create a paired movement. First, describe the overall movement, then detail each participant's actions separately.",
        'output': f"The overall motion is: {UNION_CAPTION_HOLDER}. The first participant performs: {CAPTION_HOLDER} {MOTION_HOLDER}. The second participant performs: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a paired movement description. Start with an overall action summary, then specify individual actions.",
        'output': f"This action is: {UNION_CAPTION_HOLDER}. The first person's action is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person's action is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Create a paired movement. Begin with a description of the overall action, then detail the actions of each participant.",
        'output': f"Overall, the action is: {UNION_CAPTION_HOLDER}. The first person does: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person does: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Randomly generate a paired movement. First, provide an overall description of the movement, then describe each person's actions separately.",
        'output': f"The motion is: {UNION_CAPTION_HOLDER}. The first person is: {CAPTION_HOLDER} {MOTION_HOLDER}. The second person is: {INTERACTOR_CAPTION_HOLDER} {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Generate a paired movement. First, tell me what each person’s actions are separately, then show me what the movement looks like and provide an overall description.",
        'output': f"The first person's action is: {CAPTION_HOLDER} and it looks like this: {MOTION_HOLDER}. "
                  f"The second person's action is: {INTERACTOR_CAPTION_HOLDER} and it looks like this: {INTERACTOR_MOTION_HOLDER}."
                  f" This movement can be described overall as: {UNION_CAPTION_HOLDER}"}

]
