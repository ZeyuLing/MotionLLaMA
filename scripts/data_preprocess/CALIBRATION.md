本项目的目标为通用人体动作生成。现需要一批标注员对人体动作数据集进行文本标注。
本项目待标数据集有chi3d, fit3d, hi4d, humansc3d, interhuman(优先级较低) 五个数据集。
1. 运行 
```
python scripts/motion_calibration.py data/motionhub/chi3d/train.json # 标注chi3d
python scripts/motion_calibration.py data/motionhub/fit3d/train.json # 标注fit3d
python scripts/motion_calibration.py data/motionhub/hi4d/train.json # 标注hi4d
python scripts/motion_calibration.py data/motionhub/humansc3d/train.json # 标注humansc3d
python scripts/motion_calibration.py data/motionhub/interhuman/all.json # 标注interhuman
``` 
任意一行标注某个或所有数据集，取决于标注员的任务分配方式。

2. 进入页面，服务器会向标注员分配需要标注的人体动作，每个动作需要标注至多3条描述文本。有些动作可能已经标注了1至2条，则标注员只需标注剩下的部分。

3. 标注员请用英语进行标注。如果需要使用翻译工具，建议使用chatgpt(https://chatgpt.com/)。我们提供Chatgpt Plus账号： borisgpt4@outlook.com
CHATgpt2024.

4. 如果你发现人体动作视频的质量很差，例如存在非常剧烈的抽搐、明显的穿模、几乎无法辨认动作内容等现象，请点击“This video is in bad quality”按钮。

5. 三条描述文本的详细程度呈递进关系。以下是若干示例

`
A man is kicking something with his left foot.
Describe the motion in detail. Keep the right foot still, and use the left foot to kick the object on the ground with full force.
The left foot swings forward forcefully while the center of gravity shifts backward; both hands are clenched into fists and swing to assist with generating power.
`

`
A speaker is giving a speech on the stage.
A speaker paces back and forth while giving a speech, with their right arm crossed over their chest and their left hand resting on their right arm, swinging slightly.
A speaker walks to the left on the stage while facing the audience and giving a speech. The left hand rests on the right arm, with fingers in a lotus shape, as if holding something. The right arm is crossed over the chest, with the fist clenched and the palm facing down. The head occasionally moves from side to side, looking at different sections of the audience.
`

6. 如果你标注的视频非常长（几十秒甚至一分钟） 你可以用以下类似的格式去标注: A man is giving a performers composed of following actions sequently: stand in the middle of the stage; lift the microphone with left hand; wave the right hand to the audience; ...
7. 如果你标注的视频中有两个人，则你要描述视频中黄色的那个。