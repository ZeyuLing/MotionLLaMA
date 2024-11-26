from mmotion.utils.task.modality import PAST_MOTION_HOLDER, FUTURE_MOTION_HOLDER, MIDDLE_MOTION_HOLDER, \
    MOTION_HOLDER

INBETWEEN_TEMPLATE = [
    {'input': f"Fill in the missing segment of motion between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated entire motion may be {MOTION_HOLDER}"},
{'input': f"Fill in the missing segment of motion between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated entire motion may be {MOTION_HOLDER}."},

    {'input': f"Generate the missing motion segment to connect {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The completed motion is {MOTION_HOLDER}."},

    {'input': f"Complete the motion sequence between these segments: {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion is {MOTION_HOLDER}."},

    {'input': f"Provide the missing motion that bridges {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The result is {MOTION_HOLDER}."},

    {'input': f"Can you interpolate the motion segment between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}?",
     'output': f"The completed motion might be {MOTION_HOLDER}."},

    {'input': f"Generate the missing link between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion is likely {MOTION_HOLDER}."},

    {'input': f"Fill in the gap in the motion sequence between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated sequence is {MOTION_HOLDER}."},

    {'input': f"Bridge the gap between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER} with a motion sequence.",
     'output': f"The completed motion sequence is {MOTION_HOLDER}."},

    {'input': f"Complete the motion by adding the missing part between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The final motion could be {MOTION_HOLDER}."},

    {'input': f"Interpolate the motion sequence that connects {PAST_MOTION_HOLDER} to {FUTURE_MOTION_HOLDER}.",
     'output': f"The result is {MOTION_HOLDER}."},

    {'input': f"Fill in the missing motion segment for a smooth transition from {PAST_MOTION_HOLDER} to {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion is {MOTION_HOLDER}."},

    {'input': f"Create the connecting motion segment between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The entire motion may be {MOTION_HOLDER}."},

    {'input': f"Add the missing motion that links {PAST_MOTION_HOLDER} with {FUTURE_MOTION_HOLDER}.",
     'output': f"The completed sequence is {MOTION_HOLDER}."},

    {'input': f"Provide the transition motion between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion is {MOTION_HOLDER}."},

    {'input': f"Can you generate the missing motion connecting {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}?",
     'output': f"The entire motion is {MOTION_HOLDER}."},

    {'input': f"Complete the motion sequence by interpolating between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The result is {MOTION_HOLDER}."},

    {'input': f"Find the missing motion segment to transition from {PAST_MOTION_HOLDER} to {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion could be {MOTION_HOLDER}."},

    {'input': f"Fill the gap in the motion by connecting {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion sequence is {MOTION_HOLDER}."},

    {'input': f"Generate a motion segment to fill the gap between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The entire motion might look like {MOTION_HOLDER}."},

    {'input': f"Interpolate the missing motion to smoothly connect {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The resulting motion is {MOTION_HOLDER}."},

    {'input': f"Can you fill in the missing motion sequence between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}?",
     'output': f"The interpolated motion could be {MOTION_HOLDER}."},

    {'input': f"Bridge the motion gap between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The completed motion sequence is {MOTION_HOLDER}."},

    {'input': f"Generate the motion transition from {PAST_MOTION_HOLDER} to {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion is {MOTION_HOLDER}."},

    {'input': f"Fill in the missing motion to complete the sequence between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The entire motion may be {MOTION_HOLDER}."},

    {'input': f"Provide the missing motion that fits between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The resulting motion sequence is {MOTION_HOLDER}."},

    {'input': f"Complete the sequence by interpolating the motion segment between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The final motion might look like {MOTION_HOLDER}."},

    {'input': f"Generate the motion to fill in the gap between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion sequence is {MOTION_HOLDER}."},

    {'input': f"Create the missing motion segment that transitions from {PAST_MOTION_HOLDER} to {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion is likely {MOTION_HOLDER}."},

    {'input': f"Interpolate the motion gap between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The completed sequence is {MOTION_HOLDER}."},

    {'input': f"Fill in the missing portion of the motion from {PAST_MOTION_HOLDER} to {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion is {MOTION_HOLDER}."},

    {'input': f"Provide the motion sequence that completes the gap between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The result is {MOTION_HOLDER}."},

    {'input': f"Can you interpolate the missing motion between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}?",
     'output': f"The entire motion could be {MOTION_HOLDER}."},

    {'input': f"Generate the connecting motion between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion sequence is {MOTION_HOLDER}."},

    {'input': f"Add the missing motion segment to transition from {PAST_MOTION_HOLDER} to {FUTURE_MOTION_HOLDER}.",
     'output': f"The result is {MOTION_HOLDER}."},

    {'input': f"Create a motion segment to complete the gap between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion might be {MOTION_HOLDER}."},

    {'input': f"Complete the missing part of the motion sequence between {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The entire motion is {MOTION_HOLDER}."},

    {'input': f"Fill in the motion gap by connecting {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}.",
     'output': f"The interpolated motion could be {MOTION_HOLDER}."},

    {'input': f"Can you generate a motion sequence that bridges {PAST_MOTION_HOLDER} and {FUTURE_MOTION_HOLDER}?",
     'output': f"The result might be {MOTION_HOLDER}."},

    {'input': f"Generate the missing motion segment to transition from {PAST_MOTION_HOLDER} to {FUTURE_MOTION_HOLDER}.",
     'output': f"The completed motion sequence is {MOTION_HOLDER}."},

]

