from mmotion.utils.task.modality import PAST_MOTION_HOLDER, FUTURE_MOTION_HOLDER, MOTION_HOLDER

PRED_TEMPLATE = [
    {
        'input': f"Having observed the first half of the action: {PAST_MOTION_HOLDER}, please predict the entire motion sequence.",
        'output': f"Completed entire motion sequence: {MOTION_HOLDER}"},
    {
        'input': f"Having observed the first half of the action: {PAST_MOTION_HOLDER}, please predict the entire motion sequence.",
        'output': f"Completed entire motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Using the initial motion sequence: {PAST_MOTION_HOLDER}, predict how the full motion unfolds.",
        'output': f"The predicted complete motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"From the observed first part: {PAST_MOTION_HOLDER}, generate the entire motion.",
        'output': f"Here is the full motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Given the first half of the motion: {PAST_MOTION_HOLDER}, predict the rest and complete the sequence.",
        'output': f"The complete motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Based on the initial action: {PAST_MOTION_HOLDER}, generate the full motion sequence.",
        'output': f"The full predicted motion: {MOTION_HOLDER}"
    },
    {
        'input': f"Observe the first half of the action: {PAST_MOTION_HOLDER} and predict the full motion.",
        'output': f"The completed motion is: {MOTION_HOLDER}"
    },
    {
        'input': f"From the partial motion: {PAST_MOTION_HOLDER}, complete the entire sequence.",
        'output': f"The full motion sequence generated is: {MOTION_HOLDER}"
    },
    {
        'input': f"Given this starting motion: {PAST_MOTION_HOLDER}, predict the full sequence of actions.",
        'output': f"The predicted full motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Predict the complete motion based on the initial segment: {PAST_MOTION_HOLDER}.",
        'output': f"The full motion prediction: {MOTION_HOLDER}"
    },
    {
        'input': f"Using the observed first half: {PAST_MOTION_HOLDER}, generate the entire action sequence.",
        'output': f"The completed motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"With the motion start provided: {PAST_MOTION_HOLDER}, predict the rest of the sequence.",
        'output': f"The full predicted sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Predict the full motion starting from: {PAST_MOTION_HOLDER}.",
        'output': f"The entire motion sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Given the first part: {PAST_MOTION_HOLDER}, complete the full motion trajectory.",
        'output': f"The full motion generated is: {MOTION_HOLDER}"
    },
    {
        'input': f"Use the first segment: {PAST_MOTION_HOLDER} to infer the entire motion.",
        'output': f"The completed sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Having the initial motion: {PAST_MOTION_HOLDER}, predict the entire motion path.",
        'output': f"The predicted motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Starting from {PAST_MOTION_HOLDER}, generate the full sequence of motion.",
        'output': f"The full motion trajectory: {MOTION_HOLDER}"
    },
    {
        'input': f"Observe the initial part: {PAST_MOTION_HOLDER}, then predict the entire sequence.",
        'output': f"The predicted full sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Using the observed start: {PAST_MOTION_HOLDER}, create the entire motion sequence.",
        'output': f"The completed motion sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Given the initial motion details: {PAST_MOTION_HOLDER}, predict how it concludes.",
        'output': f"The full motion sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Starting from the observed segment: {PAST_MOTION_HOLDER}, complete the motion.",
        'output': f"The predicted complete motion is: {MOTION_HOLDER}"
    },
    {
        'input': f"Observe the first part of the motion: {PAST_MOTION_HOLDER} and predict the full trajectory.",
        'output': f"The completed sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Based on the initial action {PAST_MOTION_HOLDER}, generate the rest of the motion.",
        'output': f"The full motion sequence generated is: {MOTION_HOLDER}"
    },
    {
        'input': f"Predict the rest of the sequence from the observed motion: {PAST_MOTION_HOLDER}.",
        'output': f"The full motion trajectory is: {MOTION_HOLDER}"
    },
    {
        'input': f"Use the first motion sequence: {PAST_MOTION_HOLDER} to complete the entire motion.",
        'output': f"The resulting full motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Predict the full sequence from the partial motion: {PAST_MOTION_HOLDER}.",
        'output': f"The completed motion sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Given the start of the motion: {PAST_MOTION_HOLDER}, create the entire action sequence.",
        'output': f"The complete motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Using the observed initial part: {PAST_MOTION_HOLDER}, predict the entire sequence.",
        'output': f"The full motion generated is: {MOTION_HOLDER}"
    },
    {
        'input': f"From the first motion segment: {PAST_MOTION_HOLDER}, predict the remaining action.",
        'output': f"The full motion prediction is: {MOTION_HOLDER}"
    },
    {
        'input': f"Given the initial sequence {PAST_MOTION_HOLDER}, complete the entire motion.",
        'output': f"The predicted motion sequence: {MOTION_HOLDER}"
    },
    {
        'input': f"Observe the partial motion {PAST_MOTION_HOLDER} and predict the full action.",
        'output': f"The completed motion is: {MOTION_HOLDER}"
    },
    {
        'input': f"From this partial sequence {PAST_MOTION_HOLDER}, infer the full motion.",
        'output': f"The full predicted sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Starting from {PAST_MOTION_HOLDER}, complete the entire motion sequence.",
        'output': f"The entire motion is: {MOTION_HOLDER}"
    },
    {
        'input': f"Using the initial segment {PAST_MOTION_HOLDER}, generate the full trajectory.",
        'output': f"The completed motion sequence is: {MOTION_HOLDER}"
    },
    {
        'input': f"Predict the remaining motion starting with {PAST_MOTION_HOLDER}.",
        'output': f"The full sequence of motion: {MOTION_HOLDER}"
    },
    {
        'input': f"From the motion start {PAST_MOTION_HOLDER}, generate the rest of the sequence.",
        'output': f"The resulting motion is: {MOTION_HOLDER}"
    },
    {
        'input': f"Predict the entire motion path based on the initial sequence: {PAST_MOTION_HOLDER}.",
        'output': f"The completed motion trajectory: {MOTION_HOLDER}"
    },
    {
        'input': f"Observe the first motion part {PAST_MOTION_HOLDER} and predict the rest.",
        'output': f"The predicted full motion is: {MOTION_HOLDER}"
    },
    {
        'input': f"Starting with {PAST_MOTION_HOLDER}, infer how the motion sequence concludes.",
        'output': f"The full motion sequence generated is: {MOTION_HOLDER}"
    },
    {
        'input': f"Based on the beginning {PAST_MOTION_HOLDER}, predict the full action.",
        'output': f"The predicted entire motion: {MOTION_HOLDER}"
    }
]
