def filename_to_speaker_id(filename: str):
    idx = filename.split('_')[0]
    return int(idx) - 1
