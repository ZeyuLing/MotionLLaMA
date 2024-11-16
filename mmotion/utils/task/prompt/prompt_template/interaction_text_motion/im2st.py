from mmotion.utils.task.modality import MOTION_HOLDER, INTERACTOR_MOTION_HOLDER, CAPTION_HOLDER, INTERACTOR_CAPTION_HOLDER

IM2ST_TEMPLATE=[
    {"input": f"This motion involves two people. The first person moves like this: {MOTION_HOLDER},"
              f" and the second person moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Please describe the motions of both individuals separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this activity, two individuals are involved. The first person performs: {MOTION_HOLDER},"
              f" and the second person performs: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each individual's actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people are part of this action. The first one moves in this way: {MOTION_HOLDER},"
              f" and the second one in this way: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves two people moving. The first person does: {MOTION_HOLDER},"
              f" and the second person does: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions of their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two individuals perform this motion. The first individual's motion is: {MOTION_HOLDER},"
              f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}."
              f" Please describe each person's motion separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here, two people are involved in a motion. The first person acts like this: {MOTION_HOLDER},"
              f" and the second person acts like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each person's actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two individuals are part of this activity. The first one's motion is: {MOTION_HOLDER},"
              f" and the second one's motion is: {INTERACTOR_MOTION_HOLDER}."
              f" Provide a separate description of each motion.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This scenario involves two people. The first moves like this: {MOTION_HOLDER},"
              f" and the second like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this motion, two individuals are engaged. The first person moves as follows: {MOTION_HOLDER},"
              f" and the second person as follows: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each action separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion consists of two people. The first person does: {MOTION_HOLDER},"
              f" and the second person does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe the movements of both individuals separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two participants are involved in this motion. The first participant does: {MOTION_HOLDER},"
              f" and the second participant does: {INTERACTOR_MOTION_HOLDER}."
              f" Please provide separate descriptions of their motions.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This activity features two people. The first one moves like this: {MOTION_HOLDER},"
              f" and the second one like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here we have two people involved in a motion. The first performs: {MOTION_HOLDER},"
              f" and the second performs: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people are engaged in this action. The first person moves in this way: {MOTION_HOLDER},"
              f" and the second person in this way: {INTERACTOR_MOTION_HOLDER}."
              f" Please describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves two individuals in motion. The first person's action is: {MOTION_HOLDER},"
              f" and the second person's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each motion separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this activity, there are two people. The first person moves like this: {MOTION_HOLDER},"
              f" and the second person like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion includes two participants. The first participant does: {MOTION_HOLDER},"
              f" and the second participant does: {INTERACTOR_MOTION_HOLDER}."
              f" Please describe their motions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here, two individuals are engaged in a motion. The first person moves as follows: {MOTION_HOLDER},"
              f" and the second person as follows: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each action separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This activity involves two people. The first one performs: {MOTION_HOLDER},"
              f" and the second one performs: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two individuals take part in this motion. The first person's movement is: {MOTION_HOLDER},"
              f" and the second person's movement is: {INTERACTOR_MOTION_HOLDER}."
              f" Please describe each movement separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This action features two participants. The first participant acts like this: {MOTION_HOLDER},"
              f" and the second participant acts like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each participant's actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion is done by two people. The first person's action is: {MOTION_HOLDER},"
              f" and the second person's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here, two individuals are part of this motion. The first individual moves like this: {MOTION_HOLDER},"
              f" and the second individual moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Please describe each individual's motion separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves a two-person action. The first person's movement is: {MOTION_HOLDER},"
              f" and the second person's movement is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each movement separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people are engaged in this motion. The first moves like this: {MOTION_HOLDER},"
              f" and the second moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this motion, two participants are involved. The first participant performs: {MOTION_HOLDER},"
              f" and the second participant performs: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each participant's movements separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here are two individuals in a motion. The first one does: {MOTION_HOLDER},"
              f" and the second one does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This activity involves two people. The first person performs: {MOTION_HOLDER},"
              f" and the second person performs: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion features two individuals. The first individual moves like this: {MOTION_HOLDER},"
              f" and the second individual moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each individual's motion separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people are performing this motion. The first one's action is: {MOTION_HOLDER},"
              f" and the second one's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves two individuals performing a motion. The first individual's movement is: {MOTION_HOLDER},"
              f" and the second individual's movement is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here, two people are part of this motion. The first person does: {MOTION_HOLDER},"
              f" and the second person does: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions of their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion involves a duo. The first person moves like this: {MOTION_HOLDER},"
              f" and the second person moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each person's actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this scenario, two people are involved. The first individual's motion is: {MOTION_HOLDER},"
              f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each individual's actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion includes two participants. The first one's action is: {MOTION_HOLDER},"
              f" and the second one's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This activity involves two individuals. The first person does: {MOTION_HOLDER},"
              f" and the second person does: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions of their actions.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people are in this motion. The first person performs: {MOTION_HOLDER},"
              f" and the second person performs: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion consists of two people. The first individual moves like this: {MOTION_HOLDER},"
              f" and the second individual moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this activity, two participants are involved. The first person's movement is: {MOTION_HOLDER},"
              f" and the second person's movement is: {INTERACTOR_MOTION_HOLDER}."
              f" Please describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion involves two individuals. The first person moves like this: {MOTION_HOLDER},"
              f" and the second person moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here are two people in a motion. The first individual does: {MOTION_HOLDER},"
              f" and the second individual does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion involves two participants. The first one's action is: {MOTION_HOLDER},"
              f" and the second one's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this scenario, two people are part of a motion. The first person performs: {MOTION_HOLDER},"
              f" and the second person performs: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their actions.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two individuals are involved in this motion. The first one does: {MOTION_HOLDER},"
              f" and the second one does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves two people performing a motion. The first individual's action is: {MOTION_HOLDER},"
              f" and the second individual's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each person's action separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people take part in this activity. The first person moves like this: {MOTION_HOLDER},"
              f" and the second person moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion consists of two participants. The first one's motion is: {MOTION_HOLDER},"
              f" and the second one's motion is: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their motions.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here, two individuals are involved in a motion. The first individual performs: {MOTION_HOLDER},"
              f" and the second individual performs: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each action separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion involves two people. The first person does: {MOTION_HOLDER},"
              f" and the second person does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this activity, two participants are engaged. The first one's action is: {MOTION_HOLDER},"
              f" and the second one's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their actions.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves two people in motion. The first individual moves like this: {MOTION_HOLDER},"
              f" and the second individual moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here are two participants in a motion. The first participant does: {MOTION_HOLDER},"
              f" and the second participant does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each participant's actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two individuals are part of this action. The first person acts like this: {MOTION_HOLDER},"
              f" and the second person acts like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion involves two people. The first individual's motion is: {MOTION_HOLDER},"
              f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions of their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this scenario, two participants are involved. The first participant performs: {MOTION_HOLDER},"
              f" and the second participant performs: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each participant's movements separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people are engaged in this motion. The first one's action is: {MOTION_HOLDER},"
              f" and the second one's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves two individuals in a motion. The first individual moves like this: {MOTION_HOLDER},"
              f" and the second individual moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here, two people are part of this motion. The first person performs: {MOTION_HOLDER},"
              f" and the second person performs: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two participants are engaged in this motion. The first participant acts like this: {MOTION_HOLDER},"
              f" and the second participant acts like this: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each participant's actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion includes two people. The first person's action is: {MOTION_HOLDER},"
              f" and the second person's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"In this activity, two people are involved. The first person moves like this: {MOTION_HOLDER},"
              f" and the second person moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion involves two participants. The first participant does: {MOTION_HOLDER},"
              f" and the second participant does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each participant's actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves a motion with two people. The first individual's action is: {MOTION_HOLDER},"
              f" and the second individual's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here are two individuals involved in a motion. The first person moves like this: {MOTION_HOLDER},"
              f" and the second person moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions of their actions.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people are part of this action. The first person's movement is: {MOTION_HOLDER},"
              f" and the second person's movement is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each person's action separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This activity includes two participants. The first participant does: {MOTION_HOLDER},"
              f" and the second participant does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion involves two individuals. The first one moves like this: {MOTION_HOLDER},"
              f" and the second one moves like this: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their actions.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here, two participants are involved in a motion. The first participant's motion is: {MOTION_HOLDER},"
              f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe each participant's actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This motion includes two people. The first one's action is: {MOTION_HOLDER},"
              f" and the second one's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Two people are involved in this activity. The first person performs: {MOTION_HOLDER},"
              f" and the second person performs: {INTERACTOR_MOTION_HOLDER}."
              f" Provide separate descriptions for their movements.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"This involves a two-person motion. The first participant's action is: {MOTION_HOLDER},"
              f" and the second participant's action is: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their actions separately.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},

    {"input": f"Here, two individuals are part of this motion. The first person does: {MOTION_HOLDER},"
              f" and the second person does: {INTERACTOR_MOTION_HOLDER}."
              f" Describe their movements individually.",
     "output": f"The first person: {CAPTION_HOLDER}, while the second person: {INTERACTOR_CAPTION_HOLDER}"},


]