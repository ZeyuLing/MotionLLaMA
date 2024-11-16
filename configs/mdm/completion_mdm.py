_base_ = 'mdm.py'

default_hooks = dict(
    checkpoint=dict(
        less_keys=['pred_fid', 'inbetween_fid'],
        save_best=['pred_fid', 'inbetween_fid']
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
    dict(type='SplitPrediction'),
    dict(type='SplitInbetween'),
    dict(type='PackInputs', keys=['motion', 'past_motion', 'future_motion', 'middle_motion'])
]

train_dataloader = dict(
    dataset=dict(
        datasets=[dict(
            type='MultiModalLlamaDataset',
            ann_file='train.json',
            metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
            data_root='data/motionhub',
            task_mode='preset',
            tasks=['pred', 'inbetween'],
            serialize_data=False,
            min_duration=1.5,
            pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks=['pred', 'inbetween'],
                min_duration=1.5,
                serialize_data=False,
                pipeline=pipeline)]
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='test.json',
        task_mode='preset',
        tasks=['pred', 'inbetween'],
        min_duration=1.5,
        pipeline=pipeline
    )
)

test_dataloader = val_dataloader

model = dict(
    type='CompletionMDM',
)

val_evaluator = [
    dict(type='InbetweenMetric',
         tmr_model=dict(
             type='configs/tmr/tmr_motionhub.py',
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth'))),
    dict(type='PredMetric',
         tmr_model=dict(
             type='configs/tmr/tmr_motionhub.py',
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth')))]
test_evaluator = val_evaluator

vis_backends = [
    dict(type='JointsVisBackend', name='motion')
]

visualizer = dict(
    _delete_=True,
    type='MotionLLaMAVisualizer',
    vis_backends=vis_backends
)

custom_hooks = [dict(type='BasicVisualizationHook', interval=3, in_batch_interval=1000)]
