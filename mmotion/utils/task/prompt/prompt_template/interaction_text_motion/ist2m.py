from mmotion.utils.task.modality import CAPTION_HOLDER, INTERACTOR_CAPTION_HOLDER, MOTION_HOLDER, INTERACTOR_MOTION_HOLDER

IST2M_TEMPLATE=[
    {"input": f"This is a two-person action. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this two-person action for me.",
     "output": f"Ok, the first person's action is: {MOTION_HOLDER},"
               f" and the second person's action is: {INTERACTOR_MOTION_HOLDER}"},
     {"input": f"This is a two-person motion. The first person is: {CAPTION_HOLDER},"
               f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
               f" Please visualize this two-person motion for me.",
      "output": f"Ok, the first person's motion is: {MOTION_HOLDER},"
                f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"There are two people involved in this action. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this two-person action.",
     "output": f"Understood, the first person's action is: {MOTION_HOLDER},"
               f" and the second person's action is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This involves a duo. The first individual is: {CAPTION_HOLDER},"
              f" while the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please provide a visualization for this duo action.",
     "output": f"Alright, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a pair's activity. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this pair's activity.",
     "output": f"Noted, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Here is a scenario involving two people. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Visualize this scenario with two people.",
     "output": f"Sure, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a joint motion. The first actor is: {CAPTION_HOLDER},"
              f" and the second actor is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this joint motion?",
     "output": f"Certainly, the first actor's motion is: {MOTION_HOLDER},"
               f" and the second actor's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion involves two people. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please create a visualization for this two-person motion.",
     "output": f"Got it, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"There are two individuals in this motion. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly provide a visualization of this motion.",
     "output": f"Okay, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is an interaction between two people. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this interaction?",
     "output": f"Right, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"A pair is involved in this motion. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this pair's motion for me.",
     "output": f"Sure thing, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion involves a pair. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Could you visualize this pair motion?",
     "output": f"Of course, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Here is a two-person interaction. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this two-person interaction.",
     "output": f"Yes, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a collaborative action. The first actor is: {CAPTION_HOLDER},"
              f" and the second actor is: {INTERACTOR_CAPTION_HOLDER}."
              f" Could you provide a visualization for this collaborative action?",
     "output": f"Alright, the first actor's motion is: {MOTION_HOLDER},"
               f" and the second actor's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion is carried out by two people. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this motion.",
     "output": f"Understood, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two individuals are performing this action. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please create a visualization for this action.",
     "output": f"Got it, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is a paired motion. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this paired motion for me?",
     "output": f"Okay, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"There are two participants in this action. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this two-participant action.",
     "output": f"Yes, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This action involves a duo. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this duo action?",
     "output": f"Sure, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a two-person activity. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please create a visualization of this two-person activity.",
     "output": f"Certainly, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two people are performing this motion. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this motion.",
     "output": f"Right, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is a description of a two-person motion. The first actor is: {CAPTION_HOLDER},"
              f" and the second actor is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this motion for me?",
     "output": f"Sure thing, the first actor's motion is: {MOTION_HOLDER},"
               f" and the second actor's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is an interaction involving two individuals. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this interaction.",
     "output": f"Of course, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion involves a duo. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Could you provide a visualization for this duo motion?",
     "output": f"Alright, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Here is a scenario with two people. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this scenario.",
     "output": f"Understood, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is an action involving a pair. The first actor is: {CAPTION_HOLDER},"
              f" and the second actor is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this pair action.",
     "output": f"Noted, the first actor's motion is: {MOTION_HOLDER},"
               f" and the second actor's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"There are two people in this motion. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this motion.",
     "output": f"Okay, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two individuals are carrying out this action. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Could you create a visualization for this action?",
     "output": f"Certainly, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a motion done by a pair. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this pair's motion.",
     "output": f"Yes, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This involves two people in motion. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this motion.",
     "output": f"Sure, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Here is a description of a two-person activity. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this activity.",
     "output": f"Got it, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This scenario involves two individuals. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this two-person scenario?",
     "output": f"Alright, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This action involves a pair. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please create a visualization of this action.",
     "output": f"Of course, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a collaborative motion. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this collaborative motion.",
     "output": f"Understood, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two people are involved in this motion. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this two-person motion.",
     "output": f"Yes, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is a scenario with two participants. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this scenario for me?",
     "output": f"Certainly, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"There are two actors in this motion. The first actor is: {CAPTION_HOLDER},"
              f" and the second actor is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please create a visualization for this motion.",
     "output": f"Got it, the first actor's motion is: {MOTION_HOLDER},"
               f" and the second actor's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion involves two participants. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this motion.",
     "output": f"Alright, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a two-person interaction. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this two-person interaction.",
     "output": f"Okay, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two people are performing this motion. The first actor is: {CAPTION_HOLDER},"
              f" and the second actor is: {INTERACTOR_CAPTION_HOLDER}."
              f" Could you visualize this action?",
     "output": f"Sure thing, the first actor's motion is: {MOTION_HOLDER},"
               f" and the second actor's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is an activity involving two individuals. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this activity.",
     "output": f"Of course, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is a two-person scenario. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this scenario.",
     "output": f"Certainly, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Here is a motion involving two participants. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this motion.",
     "output": f"Noted, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is a description of a two-person motion. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you create a visualization for this motion?",
     "output": f"Yes, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two people are involved in this action. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this action.",
     "output": f"Alright, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is an interaction between two participants. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this interaction.",
     "output": f"Sure, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion involves a pair of individuals. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Could you create a visualization for this motion?",
     "output": f"Got it, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"There are two people in this scenario. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this scenario.",
     "output": f"Certainly, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is a motion performed by two people. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this motion.",
     "output": f"Of course, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Here is a scenario with two participants. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please create a visualization for this scenario.",
     "output": f"Yes, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two individuals are carrying out this motion. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this two-person motion?",
     "output": f"Sure thing, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a motion involving a pair. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this pair motion.",
     "output": f"Alright, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two people are described in this motion. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this two-person motion.",
     "output": f"Got it, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This is an action involving two people. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you create a visualization for this action?",
     "output": f"Certainly, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Here is a two-person activity. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this activity.",
     "output": f"Noted, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This involves a motion with two participants. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this motion.",
     "output": f"Yes, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion is done by two individuals. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you create a visualization for this motion?",
     "output": f"Alright, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Two people are engaged in this action. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this action.",
     "output": f"Of course, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes an activity between two people. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this activity.",
     "output": f"Sure, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"There are two participants in this motion. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this two-person motion.",
     "output": f"Right, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion involves two individuals. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this two-person motion.",
     "output": f"Certainly, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Here is an action involving two participants. The first participant is: {CAPTION_HOLDER},"
              f" and the second participant is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this action.",
     "output": f"Okay, the first participant's motion is: {MOTION_HOLDER},"
               f" and the second participant's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This describes a pair's motion. The first person is: {CAPTION_HOLDER},"
              f" and the second person is: {INTERACTOR_CAPTION_HOLDER}."
              f" Can you visualize this pair motion?",
     "output": f"Sure thing, the first person's motion is: {MOTION_HOLDER},"
               f" and the second person's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"There are two people in this activity. The first individual is: {CAPTION_HOLDER},"
              f" and the second individual is: {INTERACTOR_CAPTION_HOLDER}."
              f" Please visualize this activity.",
     "output": f"Got it, the first individual's motion is: {MOTION_HOLDER},"
               f" and the second individual's motion is: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"This motion involves a pair. The first actor is: {CAPTION_HOLDER},"
              f" and the second actor is: {INTERACTOR_CAPTION_HOLDER}."
              f" Kindly visualize this motion.",
     "output": f"Understood, the first actor's motion is: {MOTION_HOLDER},"
               f" and the second actor's motion is: {INTERACTOR_MOTION_HOLDER}"},
]