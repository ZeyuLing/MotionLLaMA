_base_ = 'llama3.2_3b_instruct_m2m.py'

default_hooks = dict(
    checkpoint=dict(
        less_keys=['inbetween_fid'],
        save_best=['inbetween_fid']
    )
)

pipeline = [
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         clip_duration=10., fps='fps'),
    dict(type='SplitInbetween'),
    dict(type='LoadConversation'),
    dict(type='PackInputs', keys=['motion', 'past_motion', 'middle_motion', 'future_motion'])
]

model = dict(
    init_cfg=None
)

val_dataloader = dict(
    _delete_=True,
    num_workers=0,
    batch_size=256,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=False,
    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='inbetween',
        min_duration=1.,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='InbetweenMetric',
         tmr_model=dict(
             type='configs/tmr/tmr_motionhub.py',
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth')))]

test_evaluator = val_evaluator

val_cfg = dict(
    _delete_=True,
    type='ValLoop', fp16=True, dtype='bf16')  # must set dtype to bf16, if not, the distance in vqvae will be infinity.
test_cfg = val_cfg

custom_hooks = [dict(type='BasicVisualizationHook', interval=30, in_batch_interval=200)]
