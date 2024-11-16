from os.path import join

import os
from mmengine.visualization import LocalVisBackend
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from mmotion.registry import VISBACKENDS


@VISBACKENDS.register_module()
class MergeAudioVideoVisBackend(LocalVisBackend):
    def add_image(self,
                  name: str,
                  key: str,
                  audio_path: str,
                  video_path: str,
                  start_frame=None,
                  step: int = 0,
                  **kwargs) -> None:

        save_dir = join(self._img_save_dir, name, f'{name}')
        if step is not None:
            save_dir = save_dir + f'_{step}'
        if start_frame is not None:
            save_dir = save_dir + f'_{start_frame}'
        os.makedirs(save_dir, exist_ok=True)
        save_file_name = f'{key}.mp4'
        save_path = join(save_dir, save_file_name)
        self.merge_audio_video(video_path, audio_path, save_path)

    @staticmethod
    def merge_audio_video(video_path, audio_path, save_path):

        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        video_duration = video_clip.duration
        audio_duration = audio_clip.duration
        final_duration = min(video_duration, audio_duration)

        # 修剪视频和音频到相同的时长
        video_clip = video_clip.subclip(0, final_duration)
        audio_clip = audio_clip.subclip(0, final_duration)
        final_video = video_clip.set_audio(audio_clip)
        final_video.write_videofile(save_path, codec='libx264', audio_codec='aac')
