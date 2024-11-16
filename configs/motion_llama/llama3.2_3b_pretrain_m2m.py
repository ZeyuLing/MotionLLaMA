_base_ = 'llama3.2_3b_pretrain_unify.py'

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
    dict(type='SplitPrediction'),
    dict(type='LoadConversation'),
    dict(type='PackInputs', keys=['motion', 'past_motion', 'middle_motion', 'future_motion'])
]

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        datasets=[
            dict(
                type='MultiModalLlamaDataset',
                ann_file='train.json',
                metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                tasks=['pred', 'inbetween'],
                task_mode='preset',
                instruct_mode=True,
                serialize_data=False,
                min_duration=1.,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                tasks=['pred', 'inbetween'],
                task_mode='preset',
                instruct_mode=True,
                min_duration=1.,
                serialize_data=False,
                pipeline=pipeline),
        ]),
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
        instruct_mode=True,
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
