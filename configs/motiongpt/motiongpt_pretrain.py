_base_ = '../_base_/default_runtime.py'

strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(auto_wrap_policy=dict(type='size_based_auto_wrap_policy',
                                             min_num_params=1e4),
                       use_orig_params=True,
                       mixed_precision=dict(
                           param_dtype='bfloat16',
                           buffer_dtype='bfloat16',
                           reduce_dtype='bfloat16',
                           cast_forward_inputs=True)
                       ),
    state_dict_cfg=dict(
        state_dict_type='FULL_STATE_DICT',
        state_dict_config=dict(
            type='FullStateDictConfig',
            offload_to_cpu=True,
            rank0_only=True
        ),
        optim_state_dict_config=dict(
            type='FullOptimStateDictConfig',
            offload_to_cpu=True,
            rank0_only=True
        ),

    )
)

pipeline = [
    dict(type='LoadHm3dTxt', keys=['caption', 'union_caption', 'interactor_caption'], min_duration=0.5),
    dict(type='LoadMotionVector', keys=['interhuman', 'interactor_interhuman'],
         save_keys=['motion', 'interactor_motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion', 'interactor_motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         clip_duration=10., fps='fps'),
    dict(type='MakeMessage'),
    dict(type='PackInputs', keys=['motion',
                                  'past_motion', 'middle_motion', 'future_motion', 'duration'],
         data_keys=[],  # tensors
         meta_keys=['message',
                    'motion_path', 'num_joints', 'num_frames',
                    'task', 'interhuman_path', 'caption_path',
                    'caption', 'caption_list']),
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=2e-4, betas=[0.9, 0.99], weight_decay=0.0)
)

# train_cfg = dict(by_epoch=False, max_iters=10000000, val_interval=1)
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=50)
val_cfg = dict(
    type='ValLoop', fp16=True, dtype='bf16')  # must set dtype to bf16, if not, the distance in vqvae will be infinity.
test_cfg = val_cfg

model = dict(
    type='MotionGPT',
    data_preprocessor=dict(
        type='MotionDataPreprocessor',
        non_concatenate_keys=['message'],
        motion_keys=['motion', 'interactor_motion',
                     'past_motion', 'middle_motion', 'future_motion'],
        pad_module=dict(
            type='Pad1D',
            pad_to_max=True,
        ),
        normalizer=dict(type='BaseMotionNormalizer',
            feat_bias=1.0,
            mean_keys='pos_mean',
            std_keys='pos_std',
            norm_path='data/motionhub/statistics/interhuman.pkl'),
        vec2joints_fn='interhuman2joints',
        vec2rotation_fn='dummy_vec2rotation'
    ),

    mm_tokenizer_cfg=dict(
        motion=dict(
            type='configs/vqvae/v1/vanilla_joints_64_1024code_512dim_3depth.py',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='work_dirs/vanilla_joints_64_1024code_512dim_3depth/best_MPJPE_epoch_1300.pth'
            )
        )
    ),
    pretrained_lm='checkpoints/flan-t5-base',
    text_tokenizer_cfg=dict(
        pad_token='<pad>',
        pad_token_id=0,
        padding_side='right'

    )

)

train_dataloader = dict(
    num_workers=2,
    batch_size=96,
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
                tasks=['pre_n2m', 'pre_n2tm'],
                instruct_mode=False,
                serialize_data=False,
                min_duration=0.5,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks=['pre_n2m', 'pre_n2tm'],
                instruct_mode=False,
                min_duration=0.5,
                serialize_data=False,
                pipeline=pipeline),
        ]),
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
        ann_file='small_test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks=['pre_n2m', 'pre_n2tm'],
        instruct_mode=False,
        min_duration=0.5,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

val_evaluator = [dict(type='UnconditionalMetric',
                      tmr_model=dict(
                          type='configs/tmr/tmr_motionhub.py',
                          init_cfg=dict(
                              type='Pretrained',
                              checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1680.pth')))
                 ]
test_evaluator = val_evaluator

vis_backends = [
    dict(type='JointsVisBackend', name='motion'),
    dict(type='TextVisBackend', name='text'),
]

visualizer = dict(
    _delete_=True,
    type='MotionLLaMAVisualizer',
    vis_backends=vis_backends
)

custom_hooks = [dict(type='BasicVisualizationHook', interval=10, in_batch_interval=600)]
default_hooks = dict(
    logger=dict(interval=20),
    # checkpoint=dict(interval=10000)
)
