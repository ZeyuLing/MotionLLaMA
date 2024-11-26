_base_ = 'tmr_motionhub.py'

pipeline = [
    dict(type='LoadHm3dTxt', keys=['union_caption'], min_duration=0.5),
    dict(type='LoadMotionVector', keys=['interhuman', 'interactor_interhuman'],
         save_keys=['motion', 'interactor_motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion', 'interactor_motion'],
         no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='RandomCrop', keys=['motion', 'interactor_motion'], clip_len=300),
    dict(type='PackInputs', keys=['motion', 'num_frames', 'interactor_motion'],
         meta_keys=['duration', 'num_joints', 'union_caption',
                    'interhuman_path', 'union_caption_path', 'interactor_interhuman_path']),
]

model = dict(
    type='InterTMR',
    motionencoder=dict(
        nfeats=312,
        latent_dim=512,
        num_head=12,
    ),
    motiondecoder=dict(
        nfeats=312,
        latent_dim=512,
        num_head=12,
    ),
    textencoder=dict(
        latent_dim=512,
        num_head=12,
    ),
    data_preprocessor=dict(
        motion_keys=['motion', 'interactor_motion']
    ),
    loss_cfg=dict(
        type='TMRLoss',
    )
)

train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        datasets=[
            dict(
                type='MultiModalLlamaDataset',
                ann_file='train.json',
                metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='it2m',
                
                min_duration=0.5,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='it2m',
                
                min_duration=0.5,
                pipeline=pipeline),
        ])
)

val_dataloader = dict(
    batch_size=256,
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=False,

    dataset=dict(
        tasks='it2m',
        pipeline=pipeline),
)

test_dataloader = val_dataloader

visualizer = None
custom_hooks = None
