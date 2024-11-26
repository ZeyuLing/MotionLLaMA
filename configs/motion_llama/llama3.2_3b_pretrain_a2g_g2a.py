_base_ = 'llama3.2_3b_pretrain_unify.py'


pipeline = [
    dict(type='LoadAudio', keys=['audio'], sr=24000),
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         audio_keys=['audio'],
         clip_duration=10., fps='fps', sr='sr'),
    dict(type='LoadConversation'),
    dict(type='PackInputs', keys=['audio', 'motion'],
         meta_keys=['sr']),
]

train_dataloader = dict(
    batch_size=12,
    dataset=dict(
        datasets=[
            dict(
                type='MultiModalLlamaDataset',
                ann_file='train.json',
                metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                tasks=['a2g', 'g2a'],
                task_mode='preset',
                serialize_data=False,
                min_duration=1.,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                tasks=['a2g', 'g2a'],
                task_mode='preset',
                min_duration=1.,
                serialize_data=False,
                pipeline=pipeline),
        ]),
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
        ann_file='test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='a2g',
        min_duration=1.,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

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

custom_hooks = [dict(type='BasicVisualizationHook', interval=1, in_batch_interval=50)]

train_cfg=dict(max_epochs=1500, val_interval=1500)