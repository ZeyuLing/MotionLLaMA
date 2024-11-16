_base_ = 'mwnet.py'
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        by_epoch=True,
        max_keep_ckpts=10,
        save_best=['fid_k'],
        less_keys=['fid_k'],
        rule='less',
        greater_keys=['beat_alignment', 'diversity']
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
pipeline = [
    dict(type='LoadHm3dTxt', keys=['caption'], min_duration=0.5),
    dict(type='LoadAudio', keys=['music'], sr=24000),
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         audio_keys=['music'], clip_duration=10., fps='fps', sr='sr'),
    dict(type='PackInputs', keys=['motion', 'music'],
         data_keys=[],  # tensors
         meta_keys=['motion_path', 'num_joints', 'num_frames', 'task', 'interhuman_path', 'audio_num_frames',
                    'caption_path', 'caption', 'sr']),
]

train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['dataset', 'version', 'dataset_type', 'task_name'],
        datasets=[
            dict(
                type='MultiModalLlamaDataset',
                ann_file='train.json',
                metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='tm2d',
                serialize_data=False,
                min_duration=0.5,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='tm2d',
                min_duration=0.5,
                serialize_data=False,
                pipeline=pipeline),
        ]),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

val_dataloader = dict(
    num_workers=0,
    batch_size=512,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=False,
    drop_last=False,

    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='tm2d',
        instruct_mode=False,
        min_duration=0.5,
        verbose=True,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

model = dict(
    use_control=True,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/mwnet/best_fid_epoch_1350.pth'
    ),
    audio_encoder=dict(
        type='configs/wav_tokenizer/wav_tokenizer_large_unify_600_24k_4096.py'
    )
)

val_evaluator = [dict(type='M2DMetric')]
test_evaluator = val_evaluator
vis_backends = [
    dict(type='JointsVisBackend', name='motion'),
    dict(type='TextVisBackend', name='text'),
    dict(type='AudioVisBackend', name='audio'),
    dict(type='MergeAudioVideoVisBackend', name='merge_audio_video')
]

visualizer = dict(
    vis_backends=vis_backends,
    audio_key='music',
    text_key='caption'
)

custom_hooks = [dict(type='BasicVisualizationHook', interval=1, in_batch_interval=10)]

train_cfg = dict(val_interval=200)
