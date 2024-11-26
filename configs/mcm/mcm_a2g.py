_base_ = 'mcm.py'
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=200,
                ignore_last=True,
                interval_exp_name=1000),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=100,
        by_epoch=True,
        max_keep_ckpts=10,
        save_best=['fid'],
        less_keys=['fid'],
        rule='less',
        greater_keys=['l1_div', 'beat_alignment']
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
pipeline = [
    dict(type='LoadHm3dTxt', keys=['caption'], min_duration=0.5),
    dict(type='LoadAudio', keys=['audio'], sr=24000),
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         audio_keys=['audio'], clip_duration=10., fps='fps', sr='sr'),
    dict(type='PackInputs', keys=['motion', 'audio'],
         data_keys=[],  # tensors
         meta_keys=['motion_path', 'num_joints', 'num_frames', 'task', 'interhuman_path', 'audio_path',
                    'caption_path', 'caption', 'sr', 'audio_num_frames', 'fps']),
]

train_dataloader = dict(
    dataset=dict(
        datasets=[
            dict(
                type='MultiModalLlamaDataset',
                ann_file='train.json',
                metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='ta2g',
                serialize_data=False,
                min_duration=0.5,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='ta2g',
                min_duration=0.5,
                serialize_data=False,
                pipeline=pipeline),
        ]
    )

)

val_dataloader = dict(
    batch_size=256,
    dataset=dict(
        tasks='ta2g',
        pipeline=pipeline)
)

test_dataloader = val_dataloader

model = dict(
    use_control=True,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/mwnet/best_fid_epoch_1350.pth'
    ),
    audio_encoder=dict(
        type='configs/wav_tokenizer/wav_tokenizer_small_600_24k_4096.py'
    )
)

val_evaluator = [
    dict(
        type='S2GMetric',
        gesture_vae=dict(
            type='configs/gesture_vae/gesture_vae.py',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='work_dirs/gesture_vae/best_MPJPE_epoch_4600.pth'
            )
        )
    )
]

test_evaluator = val_evaluator

visualizer = dict(
    audio_key='audio',
    text_key='caption'
)

custom_hooks = [dict(type='BasicVisualizationHook', interval=10, in_batch_interval=200)]

train_cfg = dict(val_interval=200)
