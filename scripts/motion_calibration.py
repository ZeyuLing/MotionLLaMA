import json
import os
import sys
from os.path import exists, join, dirname
from typing import Dict, Union, List

import fire
import gradio as gr

sys.path.append(os.curdir)

data_root = 'data/motionhub'


def write_txt(save_path, data, encoding='utf8'):
    with open(save_path, 'w', encoding=encoding) as fp:
        fp.write(data)


def read_list_txt(txt_path: str, encoding='utf8') -> List[str]:
    with open(txt_path, 'r', encoding=encoding) as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
        return lines


def write_json(save_path: str, data):
    """
    :param save_path: save path of json
    :param data: object need to write to json
    :return: None
    """
    with open(save_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


def read_json(path: str) -> Union[List, Dict]:
    """
    :param save_path: save path of json
    :return: object read from json
    """
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def get_next_sample(data):
    """ Find an uncalibrated sample and return
    :param data: data_list in the annotation file
    :return:
    """
    for key, value in data.items():
        if 'motionx' in value['smplx_path'] or 'beat_v2' in value['smplx_path']:
            continue
        if (value.get('invalid', None) is not True
                and value.get('vis_path') is not None):
            if not exists(join(data_root, value['vis_path'])):
                print(join(data_root, value['vis_path']))
                continue

            if 'caption_path' not in value or not exists(join(data_root, value['caption_path'])):
                print(f'annotating {key}')
                return key, value

            if exists(join(data_root, value['caption_path'])):
                caption = read_list_txt(join(data_root, value['caption_path']))
                if len(caption) < 3:
                    return key, value
    return None, None


# 保存标注结果
def save_annotation(anno_path: str, anno_data: Dict, sample_id, coarse_text, medium_text, fine_text):
    content = (coarse_text.replace('\n', ' ').strip() + '\n'
               + medium_text.replace('\n', ' ').strip() + '\n'
               + fine_text.replace('\n', ' ').strip())
    if anno_data['data_list'][sample_id].get('caption_path') is None:
        caption_path = anno_data['data_list'][sample_id]['smplx_path'].replace('standard_smplx/',
                                                                               'caption/').replace('.npz', '.txt')
    else:
        caption_path = anno_data['data_list'][sample_id].get('caption_path')
    os.makedirs(dirname(join(data_root, caption_path)), exist_ok=True)
    write_txt(join(data_root, caption_path), content)
    anno_data['data_list'][sample_id]['caption_path'] = caption_path
    write_json(anno_path, anno_data)
    return anno_data


# 设置无效样本
def set_invalid(anno_path, anno_data: Dict, sample_id, ):
    anno_data['data_list'][sample_id]['invalid'] = True
    write_json(anno_path, anno_data)
    return get_next_sample(anno_data['data_list'])


# 获取下一个样本
def next_sample(anno_path: str, anno_data: Dict, coarse_text, medium_text, fine_text, sample_id):
    if sample_id is not None:
        anno_data = save_annotation(anno_path, anno_data, sample_id, coarse_text, medium_text, fine_text)
    sample_id, sample = get_next_sample(anno_data['data_list'])
    if sample is None:
        return "All motions annotated", None, "", "", ""
    if sample.get('caption_path') is not None and exists(sample['caption_path']):
        annotated_text = read_list_txt(sample['caption_path'])
        annotated_text = annotated_text + [""] * (3 - len(annotated_text))
        return sample_id, join(data_root, sample['vis_path']), annotated_text[0], annotated_text[1], annotated_text[2]

    else:
        return sample_id, join(data_root, sample['vis_path']), "", "", "", anno_data


# 设置无效样本并获取下一个
def invalid_sample(anno_path: str, anno_data: Dict, sample_id):
    if sample_id is not None:
        set_invalid(anno_path, anno_data, sample_id)
    sample_id, sample = get_next_sample(anno_data['data_list'])
    if sample is None:
        return "所有样本均已标注完毕。", None, "", "", ""
    return sample_id, join(data_root, sample['vis_path']), "", "", "", anno_data


def ui(anno_path: str = 'data/motionhub/all.json'):
    anno_data = read_json(anno_path)
    # 初始化第一个样本
    initial_sample_id, initial_sample = get_next_sample(anno_data['data_list'])
    if initial_sample_id is None:
        print('No sample need to calib')

    # 创建Gradio接口
    with gr.Blocks() as demo:
        gr.Markdown("# Calibration webui for MotionLLaMA")

        sample_id = gr.Text(initial_sample_id)
        anno_path = gr.State(anno_path)
        anno_data = gr.State(anno_data)
        with gr.Row():
            video = gr.Video(value=join(data_root, initial_sample['vis_path']) if initial_sample else None)
            with gr.Column():
                coarse_text = gr.Textbox(label="Describe the event that the person in the video"
                                               " is currently engaged in. For example: A man is kicking something with his left foot")
                medium_text = gr.Textbox(
                    label="Describe the motion in detail. Keep the right foot still, and use the left foot to kick the object on the ground with full force")
                fine_text = gr.Textbox(label="Describe the motion in detail, as precisely as possible:"
                                             " For example, the left foot swings forward forcefully while the center of gravity shifts backward; both hands are clenched into fists and swing to assist with generating power.")

        with gr.Row():
            invalid_btn = gr.Button("This video is in bad quality")
            next_btn = gr.Button("Confirm annotation")

        next_btn.click(next_sample, [anno_path, anno_data, coarse_text, medium_text, fine_text, sample_id],
                       [sample_id, video, coarse_text, medium_text, fine_text, anno_data])
        invalid_btn.click(invalid_sample, [anno_path, anno_data, sample_id],
                          [sample_id, video, coarse_text, medium_text, fine_text, anno_data])
        demo.launch(server_name='0.0.0.0', share=True, debug=True, show_error=True)


if __name__ == '__main__':
    fire.Fire(ui)
