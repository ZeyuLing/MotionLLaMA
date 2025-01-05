# <img src="assets/motion_llama_logo.png" alt="Logo" style="width:50px; vertical-align:middle;"> **MotionLLaMA: A Unified Framework for Motion Synthesis and Comprehension**

![](./assets/overview.png)

<p align="center">
  <a href='https://arxiv.org/abs/2411.17335'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://zeyuling.github.io/MotionLLaMA/'>
  <img src='https://img.shields.io/badge/Project-Page-orange?style=flat&logo=Google%20chrome&logoColor=orange'></a>
  <!-- <a href='https://youtu.be/0a0ZYJgzdWE'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a> -->
  <a href='https://github.com/ZeyuLing/MotionLLaMA'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
</p>

<p align="center">
<strong>MotionLLaMA: A Unified Framework for Motion Synthesis and Comprehension</strong>
    <br>
    <a href='https://scholar.google.be/citations?hl=nl&user=znEflnQAAAAJ&view_op=list_works&gmla=AOAOcb2TR7qEXM6UaMoS2X58UZTBNRqgsZuX5pVg44IH3QjDY34EcXsYR1ulftMWcE4I2NDA6-JqCvBmLANJgCfgDvkD' target='_blank'>Zeyu Ling*</a>&emsp;
    <a href='' target='_blank'>Bo Han*</a>&emsp;
    <a href='' target='_blank'>Shiyang Li</a>&emsp;
    <a href='' target='_blank'>Hongdeng Shen</a>&emsp;
    <a href='' target='_blank'>Jikang Cheng</a>&emsp;
    <a href='' target='_blank'>Changqing Zou</a>&emsp;
    <br>
    Zhejiang University&emsp;
    Zhejiang Lab
    <br>
</br>


## 💻 Project Page



<p align="center">

  <a href='https://zeyuling.github.io/MotionLLaMA/'></a>

  <img src='https://img.shields.io/badge/Project-Page-orange?style=flat&logo=Google%20chrome&logoColor=orange'>		</a>

</p>

## 📖 Introduction

This project introduces:

- MMotion: A public motion-related common library based on MMEngine, which includes PyTorch implementations of
  MotionLLaMA and various motion models.

- MotionHub: Currently the largest open-source multimodal, multi-task motions dataset.

## 📜 What's New 

- [x] 2024-12-27: Release the MotionHub V2, which involves following updates compared to the original version:
  - 1. Manually correct the captions in Fit3D, HumanSC3D, Hi4D subset.
  - 2. Manually filter and correct the InterHuman datset, low-quality motion clips are removed.
  - 3. Chi3D dataset is removed, since the motion quality is not good.
  - 4. Use PoseScript to generate frame-level caption for AIST++ and BEATV2 dataset, and we use ChatGPT-4o-mini to propess the frame-level caption to sentence-level caption.
  - 5. Use ChatGPT-4o-mini to correct the caption in MotionX dataset w.r.t the frame-level caption, some original captions are not correct.
  - 6. We define the granularity of all captions, including Macro, Meso and Micro. Macro is the lowest granularity, and Micro is the highest granularity.
  - 7. We segment the BEATV2 dataset into clips with duration less than 12 seconds. We use whisper to generate the corresponding spoken text of each clip. Each clip contains complete setences, we do not segment one single sentence into multiple clips.
  - 8. We remove the preclude dance clips in FineDance dataset, in the preclude clips, the dancer is not dancing but keeping the same pose. Then, we segment the remaining clips into clips with duration less than 12 seconds.
  We hope this version can be more useful for the community.
- [x] Release the MMotion Library.
- [x] Release the MotionHub dataset.
- [x] Release the demo video.



## 📥 Dataset Download

<div align="center">
<table cellspacing="0" cellpadding="0" bgcolor="#ffffff" border="0">
  <tr>
    <th align="center">Dataset</th>
    <th align="center">Clip Number</th>
    <th align="center">Caption Number</th>
    <th align="center">Google Drive</th>
    <th align="center">Baidu Disk</th>
  </tr>
  <tr></tr>
  <tr>
  <td align="center"><b>MotionHub V1</b></td>
  <td align="center"><b>131512</b></td>
  <td align="center"><b>269873</b></td>
  <td align="center"><b> Coming Soon </b></td>
  <td align="center"><b>https://pan.baidu.com/s/1vuewGrtVF9PjhEIiv153pw?pwd=AIXM</b></td>
  </tr>
  <tr>
  <td align="center"><b>MotionHub V2</b></td>
  <td align="center"><b>142350</b></td>
  <td align="center"><b>259998</b></td>
  <td align="center"><b> Coming Soon </b></td>
  <td align="center"><b>https://pan.baidu.com/s/1WO7FCC09qzkXAG0lCw1AVA?pwd=AIXM</b></td>
</table>
</div>



[//]: # (## ⚙️ Implementation)

[//]: # ()

[//]: # (Coming soon!)

[//]: # (## 🤝 Citation)

[//]: # ()

[//]: # (If you find this repository useful for your work, please consider citing it as follows:)

[//]: # ()

[//]: # (```)

[//]: # (@article{ling2023mcm,)

[//]: # (  title={Mcm: Multi-condition motion synthesis framework for multi-scenario},)

[//]: # (  author={Ling, Zeyu and Han, Bo and Wong, Yongkang and Kangkanhalli, Mohan and Geng, Weidong},)

[//]: # (  journal={arXiv preprint arXiv:2309.03031},)

[//]: # (  year={2023})

[//]: # (})

[//]: # (```)
