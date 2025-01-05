from mmotion.utils.task.modality import UNION_CAPTION_HOLDER, MOTION_HOLDER, INTERACTOR_MOTION_HOLDER

IT2M_TEMPLATE = [
    {
        "input": f"Generate a sequence of actions that meets the following description: {UNION_CAPTION_HOLDER} This action may involve two people.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What does this sequence of actions look like?",
     "output": f"This action might look like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Describe the actions according to the following: {UNION_CAPTION_HOLDER} There might be two individuals involved.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions appear?",
     "output": f"It would look something like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Create actions based on the following description: {UNION_CAPTION_HOLDER}. Two people may be part of these actions.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} Can you show how these actions would unfold?",
     "output": f"The actions might unfold like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Formulate a sequence of actions as described: {UNION_CAPTION_HOLDER}. There could be involvement from two people.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What do these actions look like in practice?",
     "output": f"In practice, it might look like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Outline a set of actions based on the following: {UNION_CAPTION_HOLDER}, possibly involving two people.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} Describe the visual representation of these actions.",
     "output": f"The visual representation might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Devise a sequence of actions according to: {UNION_CAPTION_HOLDER}. Two people might participate.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions be depicted?",
     "output": f"These actions would be depicted as follows: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Generate an action sequence that aligns with the description: {UNION_CAPTION_HOLDER}. Two people might be involved.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What is the appearance of these actions?",
     "output": f"The appearance might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Design a sequence of actions as per: {UNION_CAPTION_HOLDER}, potentially involving two individuals.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How might these actions manifest?",
     "output": f"These actions might manifest as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Compose actions based on: {UNION_CAPTION_HOLDER}. There could be two people engaged.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What form would these actions take?",
     "output": f"They might take the form of: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Create a sequence of actions from the following: {UNION_CAPTION_HOLDER}, possibly involving two people.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions be visualized?",
     "output": f"They would be visualized as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Invent a series of actions that match the following: {UNION_CAPTION_HOLDER}. Two individuals might be involved.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What does this series of actions look like?",
     "output": f"This series might look like: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Detail a sequence of actions as follows: {UNION_CAPTION_HOLDER}. There may be two participants.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} Describe these actions visually.",
     "output": f"Visually, they might appear as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Construct an action sequence from: {UNION_CAPTION_HOLDER}, with possible involvement of two people.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would this action sequence appear?",
     "output": f"This action sequence might appear as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Draft actions based on the following: {UNION_CAPTION_HOLDER}. Two people may be engaged in these actions.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What is the visual appearance of these actions?",
     "output": f"The visual appearance might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Sketch a series of actions according to: {UNION_CAPTION_HOLDER}. There might be two individuals participating.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions look in detail?",
     "output": f"In detail, they might look like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Generate an action sequence that matches: {UNION_CAPTION_HOLDER}, with the possibility of involving two people.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How might these actions be illustrated?",
     "output": f"They might be illustrated as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Formulate a set of actions described as: {UNION_CAPTION_HOLDER}. Two people might be part of this set.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What do these actions look like in sequence?",
     "output": f"In sequence, they might look like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Construct actions according to: {UNION_CAPTION_HOLDER}. There may be involvement of two people.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} Describe how these actions would appear.",
     "output": f"They would appear as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Create an action sequence based on: {UNION_CAPTION_HOLDER}. Two people could be involved.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What is the sequence of these actions?",
     "output": f"The sequence might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Devise actions as per the following description: {UNION_CAPTION_HOLDER}. This may involve two individuals.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How are these actions depicted?",
     "output": f"They are depicted as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Generate a sequence of actions from the description: {UNION_CAPTION_HOLDER}, potentially involving two people.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would this action sequence manifest?",
     "output": f"This sequence might manifest as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Describe actions based on: {UNION_CAPTION_HOLDER}. Two people might be part of these actions.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What form do these actions take?",
     "output": f"They might take the form of: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Compose a sequence of actions from: {UNION_CAPTION_HOLDER}, with possible involvement from two people.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions be portrayed?",
     "output": f"They would be portrayed as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Create actions following the description: {UNION_CAPTION_HOLDER}. Two individuals might be engaged.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What do these actions look like?",
     "output": f"They might look like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Draft a sequence of actions from: {UNION_CAPTION_HOLDER}. There could be two participants.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How are these actions visually represented?",
     "output": f"Visually, they might be represented as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Generate an action sequence based on: {UNION_CAPTION_HOLDER}, possibly involving two people.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What is the depiction of these actions?",
     "output": f"The depiction might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Outline actions described by: {UNION_CAPTION_HOLDER}. There may be two individuals involved.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} Describe how these actions unfold.",
     "output": f"They might unfold as follows: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Invent a sequence of actions from: {UNION_CAPTION_HOLDER}, with the potential involvement of two people.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would this action be presented?",
     "output": f"This action might be presented as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Design actions based on: {UNION_CAPTION_HOLDER}. Two people might be participating.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What is the sequence of events for these actions?",
     "output": f"The sequence might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Generate a series of actions from: {UNION_CAPTION_HOLDER}, potentially involving two people.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions be depicted?",
     "output": f"They would be depicted as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Formulate actions described as: {UNION_CAPTION_HOLDER}. Two individuals might be part of this.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How do these actions look?",
     "output": f"They might look like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Generate an action sequence that follows: {UNION_CAPTION_HOLDER}, involving two people.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What is the visual sequence of these actions?",
     "output": f"The visual sequence might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Create actions described in: {UNION_CAPTION_HOLDER}. There may be involvement of two individuals.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} Describe the sequence of these actions.",
     "output": f"They might sequence as follows: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Formulate a series of actions from: {UNION_CAPTION_HOLDER}, possibly involving two people.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions be portrayed?",
     "output": f"They would be portrayed as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Generate a sequence of actions according to: {UNION_CAPTION_HOLDER}. Two individuals might take part.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What is the visual depiction of these actions?",
     "output": f"The visual depiction might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Describe actions based on: {UNION_CAPTION_HOLDER}. Two people could be engaged in these.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How do these actions appear?",
     "output": f"They might appear as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Generate actions from the following: {UNION_CAPTION_HOLDER}. Two individuals might be involved.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What is the sequence representation of these actions?",
     "output": f"The sequence representation might be: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Create a sequence of actions as per: {UNION_CAPTION_HOLDER}, potentially involving two people.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions be described?",
     "output": f"They might be described as: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Outline actions following: {UNION_CAPTION_HOLDER}. Two people could be participants.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} Describe how these actions look in sequence.",
     "output": f"In sequence, they might look like: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {
        "input": f"Devise a series of actions described as: {UNION_CAPTION_HOLDER}. Two individuals might be part of this.",
        "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} What do these actions resemble?",
     "output": f"They might resemble: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"Formulate actions based on: {UNION_CAPTION_HOLDER}. Two people may engage in these actions.",
     "output": f"{MOTION_HOLDER}, {INTERACTOR_MOTION_HOLDER}"},
    {"input": f"{UNION_CAPTION_HOLDER} How would these actions look in detail?",
     "output": f"In detail, they might look like this: First person: {MOTION_HOLDER}. Second person: {INTERACTOR_MOTION_HOLDER}"},

]