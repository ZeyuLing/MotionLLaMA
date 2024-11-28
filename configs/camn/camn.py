_base_ = '../_base_/default_runtime.py'

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
        rule='less'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(auto_wrap_policy=dict(type='size_based_auto_wrap_policy',
                                             min_num_params=1e8),
                       use_orig_params=True,
                       mixed_precision=dict(
                           param_dtype='bfloat16',
                           buffer_dtype='bfloat16',
                           reduce_dtype='bfloat16')

                       ),
    state_dict_cfg=dict(
        state_dict_type='FULL_STATE_DICT',
        state_dict_config=dict(
            type='FullStateDictConfig',
            offload_to_cpu=True,
            rank0_only=True,
        ),
        optim_state_dict_config=dict(
            type='FullOptimStateDictConfig',
            offload_to_cpu=True,
            rank0_only=True,
        ),

    )
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    use_fsdp=True,
    optimizer=dict(type='AdamW', lr=2e-4, betas=[0.9, 0.99], weight_decay=0.0))

train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=10)
# train_cfg = dict(by_epoch=False, max_iters=10000, val_interval=5)
val_cfg = dict(type='ValLoop', fp16=True, dtype='bfloat16')
test_cfg = val_cfg

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

vis_backends = [
    dict(type='JointsVisBackend', name='motion'),
    dict(type='TextVisBackend', name='text'),
    dict(type='AudioVisBackend', name='audio'),
    dict(type='MergeAudioVideoVisBackend', name='merge_audio_video')
]

visualizer = dict(
    _delete_=True,
    type='MotionLLaMAVisualizer',
    vis_backends=vis_backends
)

custom_hooks = [dict(type='BasicVisualizationHook', interval=2, in_batch_interval=500)]

pipeline = [
    dict(type='LoadAudio', keys=['audio'], sr=24000),
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         audio_keys=['audio'],
         clip_duration=10., fps='fps', sr='sr'),
    dict(type='MakeMessage'),
    dict(type='PackInputs', keys=['audio', 'motion'],
         data_keys=[],  # tensors
         meta_keys=['message', 'duration', 'speaker_id',
                    'motion_path', 'num_joints', 'num_frames',
                    'audio_path', 'task', 'interhuman_path', 'sr', 'fps',
                    'audio_duration', 'language', 'audio_num_frames']),
]


train_dataloader = dict(
    num_workers=2,
    batch_size=512,
    persistent_workers=True,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    dataset=dict(
                type='MultiModalLlamaDataset',
                ann_file='beat_v2.0.0/train.json',
                metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='a2g',
                
                serialize_data=False,
                min_duration=0.5,
                pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

val_dataloader = dict(
    num_workers=2,
    batch_size=512,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=False,

    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='beat_v2.0.0/test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='a2g',
        min_duration=0.5,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

model = dict(
    type='CaMN',
    data_preprocessor=dict(
        type='MotionDataPreprocessor',
        normalizer=dict(type='BaseMotionNormalizer',
            feat_bias=1.0,
            mean_keys='pos_mean',
            std_keys='pos_std',
            norm_path='data/motionhub/statistics/interhuman.pkl'),
        vec2joints_fn='interhuman2joints',
        vec2rotation_fn='dummy_vec2rotation'
    ),
)

