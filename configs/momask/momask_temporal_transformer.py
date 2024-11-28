_base_ = '../_base_/default_runtime.py'
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=200,
                ignore_last=True,
                interval_exp_name=1000),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        by_epoch=True,
        max_keep_ckpts=10,
        save_best=['fid'],
        less_keys=['fid', 'mm_dist'],
        rule='less',
        greater_keys=['t2m_r_precision_top_1',
                      't2m_r_precision_top_2',
                      't2m_r_precision_top_3',
                      'diversity']
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

train_cfg = dict(by_epoch=True, max_epochs=10000, val_interval=50)
val_cfg = dict(type='ValLoop', fp16=True, dtype='bfloat16')
test_cfg = val_cfg

pipeline = [
    dict(type='LoadHm3dTxt', keys=['caption'], min_duration=0.5),
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         clip_duration=10., fps='fps'),
    dict(type='PackInputs', keys=['motion'])
]

train_dataloader = dict(
    num_workers=4,
    batch_size=2048,
    persistent_workers=True,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
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
                tasks='t2m',
                serialize_data=False,
                min_duration=0.5,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks='t2m',
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
    batch_size=2048,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=False,

    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='t2m',
        instruct_mode=False,
        min_duration=0.5,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

model = dict(
    type='MomaskTemporalTransformer',
    rvq_cfg=dict(
        type='configs/vqvae/v1/v1_rvq_6cb_64_1024code_512dim_3depth.py',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/v1_rvq_6cb_64_1024code_512dim_3depth/best_MPJPE_epoch_1500.pth'
        )
    ),

    clip_path='checkpoints/vit_base_patch32/',
    data_preprocessor=dict(
        type='MotionDataPreprocessor',
        non_concatenate_keys=['num_frames'],
        pad_module=dict(
            type='Pad1D',
            pad_to_max=True,
        ),
        motion_keys=['motion'],
        normalizer=dict(type='BaseMotionNormalizer',
            feat_bias=1.0,
            mean_keys='pos_mean',
            std_keys='pos_std',
            norm_path='data/motionhub/statistics/interhuman.pkl'),
        vec2joints_fn='interhuman2joints',
        vec2rotation_fn='dummy_vec2rotation'
    )
)

val_evaluator = [dict(type='T2MMetric',
                      tmr_model=dict(
                          type='configs/tmr/tmr_motionhub.py',
                          init_cfg=dict(
                              type='Pretrained',
                              checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth'
                          )
                      ))]
test_evaluator = val_evaluator

vis_backends = [
    dict(type='JointsVisBackend', name='motion'),
    dict(type='TextVisBackend', name='text')
]

visualizer = dict(
    _delete_=True,
    type='MotionLLaMAVisualizer',
    vis_backends=vis_backends,
    text_key='caption'
)

custom_hooks = [dict(type='BasicVisualizationHook', interval=10, in_batch_interval=200)]