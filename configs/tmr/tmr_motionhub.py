_base_ = '../_base_/default_runtime.py'


pipeline = [
    dict(type='LoadHm3dTxt', keys=['caption'], min_duration=0.5),
    dict(type='LoadMotionVector', keys=['interhuman'], save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion'], fps_key='fps', ori_fps_key='ori_fps',
         tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='RandomCrop', keys=['motion'], clip_len=300),
    dict(type='PackInputs', keys=['caption', 'motion', 'num_frames'],
         meta_keys=['duration', 'num_joints', 'interhuman_path', 'caption_path']),
]

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
        save_best=['mm_dist'],
        less_keys=['mm_dist'],
        rule='less',
        greater_keys=['r_precision_top_1', 'r_precision_top_2', 'r_precision_top_3'],
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
    optimizer=dict(type='AdamW', lr=1e-4)
)

val_evaluator = [
    dict(
        type='TMRMetric',
        text_key='lat_text',
        motion_key='lat_motion',
        top_k=3,
        r_precision_batch=256,
    )
]
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=10000, val_interval=20)
# train_cfg = dict(by_epoch=False, max_iters=1000000, val_interval=1)

val_cfg = dict(type='ValLoop', fp16=True, dtype='bfloat16')
test_cfg = val_cfg

model=dict(
    type='TMR',
    motionencoder=dict(
        type='ActorAgnosticEncoder',
        nfeats=156,
        vae=True,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_head=6,
        dropout=0.1,
        activation='gelu',
    ),
    motiondecoder=dict(
        type='ActorAgnosticDecoder',
        nfeats=156,
        vae=True,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_head=6,
        dropout=0.1,
        activation='gelu',
    ),
    textencoder=dict(
        type='DistilbertActorAgnosticEncoder',
        modelpath='checkpoints/distilbert-base-uncased',
        finetune=False,
        vae=True,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_head=6,
        dropout=0.1,
        activation='gelu',
    ),
    data_preprocessor=dict(
        type='MotionDataPreprocessor',
        non_concatenate_keys=['num_frames'],
        pad_module=dict(
            type='Pad1D',
            pad_to_max=True,
        ),
        motion_keys=['motion'],
        normalizer=dict(
            type='BaseMotionNormalizer',
            feat_bias=1.0,
            mean_keys='pos_mean',
            std_keys='pos_std',
            norm_path='data/motionhub/statistics/interhuman.pkl'),
        vec2joints_fn='interhuman2joints',
        vec2rotation_fn='dummy_vec2rotation'
    ),
    loss_cfg=dict(
        type='TMRLoss',
    )
)

train_dataloader = dict(
    num_workers=4,
    batch_size=256,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
                pipeline=pipeline),
        ]),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

val_dataloader = dict(
    num_workers=4,
    batch_size=256,
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=False,

    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='t2m',
        
        min_duration=0.5,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

visualizer=None
custom_hooks = None
