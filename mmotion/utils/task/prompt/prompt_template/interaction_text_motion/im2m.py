from mmotion.utils.task.modality import MOTION_HOLDER, INTERACTOR_MOTION_HOLDER

IM2M_TEMPLATE = [
    {'input': f"The following is the action of one person in a two-person interaction."
              f" {MOTION_HOLDER} infer the action of the other person.",
     'output': f"Okay, the other person may react like this: {INTERACTOR_MOTION_HOLDER}",
     },
    {'input': f"This is the motion of one person in an interaction: {MOTION_HOLDER},"
              f" What would the other person's motion be like?",
     'output': f"The other person might perform the following action: {INTERACTOR_MOTION_HOLDER}"},
    {
        'input': f"Here's one person's movement during an interaction: {MOTION_HOLDER}. Can you guess the other person's reaction?",
        'output': f"The other person is likely to respond with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Observe this action in a two-person scenario: {MOTION_HOLDER}. How would the other person react?",
     'output': f"The second individual may respond like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One person does this in an interaction: {MOTION_HOLDER}. What do you think the other person will do?",
     'output': f"Possibly, the other individual will react like: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The first person moves like this: {MOTION_HOLDER}. How do you think the second person will respond?",
     'output': f"The second person might respond with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is the first person’s action: {MOTION_HOLDER}. What could the second person do?",
     'output': f"The other person may take the following action: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"During an interaction, one individual does this: {MOTION_HOLDER}. Can you predict the other person’s action?",
        'output': f"The second person is expected to perform: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Given this action: {MOTION_HOLDER}, how might the other person respond?",
     'output': f"The other participant might react with: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In this interaction, one participant moves like this: {MOTION_HOLDER}. What could the other participant do?",
        'output': f"The second person could respond like: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s the movement of one person: {MOTION_HOLDER}. How do you think the other person would react?",
     'output': f"The other person may respond with the following action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Watch this movement from one participant: {MOTION_HOLDER}. How will the other person move?",
     'output': f"The other individual could act in this way: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Consider this action: {MOTION_HOLDER}. How would the other person in the interaction behave?",
     'output': f"The other participant may react as follows: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here's an interaction where one person does this: {MOTION_HOLDER}. What would the other do?",
     'output': f"The other participant might take this action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The first individual moves like this: {MOTION_HOLDER}. How might the second person react?",
     'output': f"The second person may react like: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One person’s action in an interaction is: {MOTION_HOLDER}. What would the other person do?",
     'output': f"The other person might act in the following manner: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s what one participant does: {MOTION_HOLDER}. What will the other participant do in response?",
     'output': f"Most likely, the other participant will respond with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The following action occurs: {MOTION_HOLDER}. Can you predict how the other person will respond?",
     'output': f"The other person might take this approach: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is the first person's motion: {MOTION_HOLDER}. How do you imagine the other person would move?",
     'output': f"The other individual could move in the following way: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One person performs this movement: {MOTION_HOLDER}. What will the second person do?",
     'output': f"The second individual may respond with: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Observe this motion from one participant: {MOTION_HOLDER}. What would the other participant's response be?",
        'output': f"The other individual might react with this movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is what one person does: {MOTION_HOLDER}. What could the other person’s reaction be?",
     'output': f"The second person might react as follows: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here is the action performed by one person: {MOTION_HOLDER}. What might the other person do?",
     'output': f"The other individual may respond by doing: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One participant performs this motion: {MOTION_HOLDER}. What will the other person do in response?",
     'output': f"The second person might act like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Consider the action of the first person: {MOTION_HOLDER}. How would the second person react?",
     'output': f"The second participant could react as follows: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s one person’s action: {MOTION_HOLDER}. How do you expect the second person to respond?",
     'output': f"The other participant might respond with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Given this person’s movement: {MOTION_HOLDER}, what will the other person do?",
     'output': f"The other individual is likely to react in this way: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In this scenario, one person moves like this: {MOTION_HOLDER}. What would the other person’s reaction be?",
        'output': f"The second individual may react like: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is the first person’s motion: {MOTION_HOLDER}. What action might the second person take?",
     'output': f"The other person may respond like: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Watch this action from one person: {MOTION_HOLDER}. What do you think the other person will do?",
     'output': f"The second person might react in this way: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One individual does this: {MOTION_HOLDER}. What could the other person do in response?",
     'output': f"The second person is likely to respond like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is the movement of one participant: {MOTION_HOLDER}. How would the other participant react?",
     'output': f"The second individual might take this action: {INTERACTOR_MOTION_HOLDER}"},
    {
        'input': f"The following is the action of one person in a two-person interaction. {MOTION_HOLDER} infer the action of the other person.",
        'output': f"Okay, the other person may react like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This is the motion of one person in an interaction: {MOTION_HOLDER}, What would the other person's motion be like?",
        'output': f"The other person might perform the following action: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here is the motion of one participant in an interaction: {MOTION_HOLDER}. What could the other person do in response?",
        'output': f"The second person could take the following action: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"The action performed by one individual in this interaction is: {MOTION_HOLDER}. Can you predict the response from the second individual?",
        'output': f"The second person might respond with this movement: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In this interaction, one individual moves like this: {MOTION_HOLDER}. How will the other person respond to it?",
        'output': f"The other person might respond with the following action: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This motion is performed by one person during an interaction: {MOTION_HOLDER}. What action might the other person take?",
        'output': f"The second person could react like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here is an action in a two-person interaction: {MOTION_HOLDER}. What would the second person do in return?",
        'output': f"The second individual may respond with the following motion: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Observe the movement of one person in this interaction: {MOTION_HOLDER}. What might the other person’s response look like?",
        'output': f"The other participant could respond like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In a two-person interaction, one person does this: {MOTION_HOLDER}. What is the probable reaction of the other individual?",
        'output': f"The second person might perform this action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is one person’s motion: {MOTION_HOLDER}. What do you think the other person will do in response?",
     'output': f"The other person may react like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here is the motion of the first person in this interaction: {MOTION_HOLDER}. What do you think the other individual would do?",
        'output': f"The second person may respond in this way: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here is the movement from one person: {MOTION_HOLDER}. How do you think the second person will respond?",
        'output': f"The second person might react with the following action: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Observe this movement: {MOTION_HOLDER}. How do you think the other person will react in this interaction?",
        'output': f"The second individual may perform the following motion: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This is how the first person moves in an interaction: {MOTION_HOLDER}. Can you guess the other person’s motion?",
        'output': f"The other person might act like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here’s the action of the first individual: {MOTION_HOLDER}. What could the second person’s reaction be?",
        'output': f"The other person might respond like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Watch the movement of one participant: {MOTION_HOLDER}. How do you think the other person will move?",
     'output': f"The second individual may react with this action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is what one person does: {MOTION_HOLDER}. What do you think the other person will do?",
     'output': f"The other individual might perform the following motion: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here’s an action by one person in an interaction: {MOTION_HOLDER}. How would the other individual react?",
        'output': f"The other person may respond like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The first person’s motion is as follows: {MOTION_HOLDER}. What could the second person’s action be?",
     'output': f"The other participant may react with: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Observe this action from the first person: {MOTION_HOLDER}. How do you think the second person will respond?",
        'output': f"The other individual might act in the following way: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In this situation, one individual performs this action: {MOTION_HOLDER}. What might the other person’s response look like?",
        'output': f"The second person could act like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is the motion of one person: {MOTION_HOLDER}. How do you think the other individual will respond?",
     'output': f"The other person may respond with the following motion: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"The first person in this interaction moves like this: {MOTION_HOLDER}. How would the second person react?",
        'output': f"The second person could react with: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here’s the movement of one individual in an interaction: {MOTION_HOLDER}. What could the other person’s response be?",
        'output': f"The second individual may react like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"One person moves like this during an interaction: {MOTION_HOLDER}. How do you think the second person will respond?",
        'output': f"The other person might respond with: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"During this interaction, one person performs this motion: {MOTION_HOLDER}. What is the probable response from the other person?",
        'output': f"The second individual may act like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is how the first person acts: {MOTION_HOLDER}. What would the second person do?",
     'output': f"The second person might respond in the following way: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"In this interaction, one person does this: {MOTION_HOLDER}. How might the other person react?",
     'output': f"The other participant might perform the following motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The action of one person is: {MOTION_HOLDER}. How would the other individual respond?",
     'output': f"The second person might take this action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The following movement is done by one participant: {MOTION_HOLDER}. What will the second person do?",
     'output': f"The other individual may perform this motion: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here’s what one person does in an interaction: {MOTION_HOLDER}. What could the second person’s response be?",
        'output': f"The other person may act like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"One person moves like this in a two-person interaction: {MOTION_HOLDER}. How would the other person react?",
        'output': f"The other person might perform this motion: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This is the movement of the first person in an interaction: {MOTION_HOLDER}. What is the response of the other individual?",
        'output': f"The other person could act like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One individual performs this movement: {MOTION_HOLDER}. What might the second person do?",
     'output': f"The second individual could respond with this motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s how the first person moves: {MOTION_HOLDER}. How would the second person react to that?",
     'output': f"The other person may respond like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here’s an action taken by one person: {MOTION_HOLDER}. How might the other person in the interaction respond?",
        'output': f"The second person might perform the following movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The first individual performs this movement: {MOTION_HOLDER}. How will the second person react?",
     'output': f"The other person might react with: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This is the motion of one person: {MOTION_HOLDER}. How will the second person in the interaction respond?",
        'output': f"The other individual may perform this action: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here’s the action performed by one person: {MOTION_HOLDER}. How do you think the other person will react?",
        'output': f"The other individual might respond with this action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One person performs this motion: {MOTION_HOLDER}. How will the other person react?",
     'output': f"The second individual may act like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The first person’s motion is as follows: {MOTION_HOLDER}. How will the second person respond?",
     'output': f"The other person could respond with this action: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In a two-person interaction, one person moves like this: {MOTION_HOLDER}. What will the other person do?",
        'output': f"The second individual may perform the following action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is the motion of one participant: {MOTION_HOLDER}. How will the other participant respond?",
     'output': f"The other person might react like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In this interaction, one person does the following motion: {MOTION_HOLDER}. How will the other person respond?",
        'output': f"The second individual may react with this action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s the movement performed by one person: {MOTION_HOLDER}. What might the other person do?",
     'output': f"The other individual may react in this way: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This is one participant’s motion in a two-person interaction: {MOTION_HOLDER}. What could the other person do?",
        'output': f"The second person might respond with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One individual moves like this: {MOTION_HOLDER}. What is the possible reaction from the other person?",
     'output': f"The other participant may respond with: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In this situation, one person performs this motion: {MOTION_HOLDER}. What will the second person do in response?",
        'output': f"The other person might act like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This is the motion of one individual: {MOTION_HOLDER}. How will the other person in the interaction respond?",
        'output': f"The other person may perform this action: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Observe this motion performed by one person: {MOTION_HOLDER}. What is the likely reaction of the other person?",
        'output': f"The other person could act like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The first person’s action is as follows: {MOTION_HOLDER}. How will the second person respond?",
     'output': f"The second person may perform the following motion: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This movement is performed by one individual: {MOTION_HOLDER}. What might the other person’s response be?",
        'output': f"The other individual could respond with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"In this interaction, one person acts like this: {MOTION_HOLDER}. What will the other person do?",
     'output': f"The second person may react in this way: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s the motion of one participant: {MOTION_HOLDER}. What could the second person do in response?",
     'output': f"The other person might perform the following motion: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is how one person moves: {MOTION_HOLDER}. How will the second person react to this?",
     'output': f"The other person might act like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"One participant performs this movement: {MOTION_HOLDER}. What is the expected response from the other person?",
        'output': f"The second person could perform the following action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s how one person moves: {MOTION_HOLDER}. How will the other person respond in this interaction?",
     'output': f"The second participant may react in this way: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here is the movement of one person: {MOTION_HOLDER}. What will the other individual’s response be?",
     'output': f"The second person might act like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Watch this action by one person: {MOTION_HOLDER}. What will the other person do?",
     'output': f"The second person may respond with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Observe the motion of one participant: {MOTION_HOLDER}. How will the other person react?",
     'output': f"The second individual might react like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is the action performed by one person: {MOTION_HOLDER}. What will the other person do?",
     'output': f"The second participant may perform the following action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"In this interaction, one person moves like this: {MOTION_HOLDER}. What will the other participant do?",
     'output': f"The other individual could react like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The following action is performed by one person: {MOTION_HOLDER}. How will the other person respond?",
     'output': f"The other person may react with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s the motion of one individual: {MOTION_HOLDER}. How will the second person respond to this?",
     'output': f"The second person could respond with: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"This movement is done by one person: {MOTION_HOLDER}. What is the likely response from the second person?",
        'output': f"The other individual may act like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"One person performs this motion: {MOTION_HOLDER}. How will the second person react?",
     'output': f"The second person may respond with: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Here’s what one person does: {MOTION_HOLDER}. What will the other person’s response be?",
     'output': f"The other individual may react with the following action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is how the first individual moves: {MOTION_HOLDER}. How will the second person react?",
     'output': f"The other person might perform the following action: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"In this interaction, one person moves like this: {MOTION_HOLDER}. What will the other person do?",
     'output': f"The second individual may respond with this movement: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"This is the action performed by one individual: {MOTION_HOLDER}. What will the other person do?",
     'output': f"The other person might act like this: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"One person performs the following movement: {MOTION_HOLDER}. What might the second person’s reaction be?",
        'output': f"The second person may react in the following way: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"Here’s the action of one person in an interaction: {MOTION_HOLDER}. How do you think the other individual will respond?",
        'output': f"The second participant could act in this way: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"The following action is done by one individual: {MOTION_HOLDER}. How will the second person respond?",
     'output': f"The other person might respond like this: {INTERACTOR_MOTION_HOLDER}"},

    {'input': f"Observe this action by one person: {MOTION_HOLDER}. What is the response from the other individual?",
     'output': f"The second individual may perform this motion: {INTERACTOR_MOTION_HOLDER}"},

    {
        'input': f"In this interaction, one person performs this motion: {MOTION_HOLDER}. What will the other participant do?",
        'output': f"The other person could act in this way: {INTERACTOR_MOTION_HOLDER}"},
]
