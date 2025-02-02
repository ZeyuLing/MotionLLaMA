_base_ = 'llama3.2_3b_instruct_d2m_m2d.py'

default_hooks = dict(
    checkpoint=dict(
        save_best=['beat_cover_score'],
        rule='greater',
        greater_keys=['beat_cover_score']
    )
)

pipeline = [
    dict(type='LoadAudio', keys=['music'], sr=24000),
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         audio_keys=['music'], clip_duration=10., fps='fps', sr='sr'),
    dict(type='LoadConversation'),
    dict(type='PackInputs', keys=['motion', 'music'],
         data_keys=[],  # tensors
         meta_keys=['audio_num_frames', 'music_num_frames', 'sr']),
]

model = dict(
    init_cfg=None
)

val_dataloader = dict(
    _delete_=True,
    num_workers=0,
    batch_size=64,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=False,
    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='aist/test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='d2m',
        min_duration=1.,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='D2MMetric')
]

test_evaluator = val_evaluator
