_base_ = 'tm2t_t2m.py'
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=50,
        by_epoch=True,
        max_keep_ckpts=10,
        save_best=['m2t_mm_dist'],
        less_keys=['m2t_mm_dist'],
        rule='less',
        greater_keys=['m2t_r_precision_top_1',
                      'm2t_r_precision_top_2',
                      'm2t_r_precision_top_3',
                      'rouge_l',
                      'CIDEr',
                      'bert_f1',
                      'bleu_1',
                      'bleu_4']
    ),
)
pipeline = [
    dict(type='LoadHm3dTxt', keys=['caption'], min_duration=0.5),
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         clip_duration=10., fps='fps', require_audio=False),
    dict(type='PackInputs', keys=['motion'],
         data_keys=[],  # tensors
         meta_keys=['motion_path', 'num_joints', 'num_frames', 'task', 'interhuman_path',
                    'caption_path', 'caption', 'caption_list']),
]
model = dict(
    t2m=False
)
val_evaluator = [dict(type='M2TMetric',
                      tmr_model=dict(
                          type='configs/tmr/tmr_motionhub.py',
                          init_cfg=dict(
                              type='Pretrained',
                              checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth')))
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
                tasks='m2t',
                serialize_data=False,
                min_duration=0.5,
                instruct_mode=True,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='m2t',
                instruct_mode=True,
                min_duration=0.5,
                serialize_data=False,
                pipeline=pipeline),
        ])
)

val_dataloader = dict(
    dataset=dict(
        tasks='m2t',
        pipeline=pipeline
    ),
)

test_dataloader = val_dataloader

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=50)

optim_wrapper = dict(
    optimizer=dict(lr=1e-5, betas=[0.9, 0.99], weight_decay=0.0))
