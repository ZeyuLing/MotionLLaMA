_base_ = 'tm2t_m2t.py'
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=50,
        by_epoch=True,
        max_keep_ckpts=10,
        save_best=['im2t_mm_dist'],
        less_keys=['im2t_mm_dist'],
        rule='less',
        greater_keys=['im2t_r_precision_top_1',
                      'im2t_r_precision_top_2',
                      'im2t_r_precision_top_3',
                      'inter_rouge_l',
                      'inter_CIDEr',
                      'inter_bert_f1',
                      'inter_bleu_1',
                      'inter_bleu_4']
    ),
)
pipeline = [
    dict(type='LoadHm3dTxt', keys=['union_caption'], min_duration=0.5),
    dict(type='LoadMotionVector', keys=['interhuman', 'interactor_interhuman'],
         save_keys=['motion', 'interactor_motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion', 'interactor_motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion', 'interactor_motion'],
         clip_duration=10., fps='fps', require_audio=False),
    dict(type='PackInputs', keys=['motion', 'interactor_motion'],
         data_keys=[],  # tensors
         meta_keys=['motion_path', 'num_joints', 'num_frames', 'task', 'interhuman_path',
                    'union_caption_path', 'union_caption', 'union_caption_list']),
]
model = dict(
    type='InterTM2T',
    t2m=False,
    generation_config=dict(
        max_length=400
    ),
    data_preprocessor=dict(
        motion_keys=['motion', 'interactor_motion'])
)
val_evaluator = [dict(type='IM2TMetric',
                      tmr_model=dict(
                          type='configs/tmr/tmr_inter.py',
                          init_cfg=dict(
                              type='Pretrained',
                              checkpoint='work_dirs/tmr_inter/best_mm_dist_epoch_1540.pth')))
                 ]
test_evaluator = val_evaluator

train_dataloader = dict(
    num_workers=4,
    batch_size=192,
    dataset=dict(
        datasets=[
            dict(
                type='MultiModalLlamaDataset',
                ann_file='train.json',
                metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='im2t',
                serialize_data=False,
                min_duration=0.5,
                
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='im2t',
                
                min_duration=0.5,
                serialize_data=False,
                pipeline=pipeline),
        ])
)

val_dataloader = dict(
    dataset=dict(
        tasks='im2t',
        pipeline=pipeline
    ),
)

test_dataloader = val_dataloader

train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=50)

optim_wrapper = dict(
    optimizer=dict(lr=1e-5, betas=[0.9, 0.99], weight_decay=0.0))

visualizer = dict(
    text_key='union_caption'
)


custom_hooks = [dict(type='BasicVisualizationHook', interval=2, in_batch_interval=160)]
