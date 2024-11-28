_base_ = '../_base_/default_runtime.py'

strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(auto_wrap_policy=dict(type='llama_auto_wrap_policy'),
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
    dict(type='LoadAudio', keys=['audio', 'music'], sr=24000),
    dict(type='LoadMotionVector', keys=['interhuman', 'interactor_interhuman'],
         save_keys=['motion', 'interactor_motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion', 'interactor_motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion', 'interactor_motion'],
         audio_keys=['audio', 'music'],
         clip_duration=10., fps='fps', sr='sr'),
    dict(type='SplitPrediction'),
    dict(type='SplitInbetween'),
    dict(type='LoadConversation'),
    dict(type='PackInputs', keys=['audio', 'music', 'motion', 'interactor_motion',
                                  'past_motion', 'middle_motion', 'future_motion']
         )
]

model = dict(
    type='MotionCausalLM',
    is_pretrain_stage=True,
    text_tokenizer_cfg=dict(
        padding_side="left",
        pad_token='<|finetune_right_pad_id|>',
        pad_token_id=128004
    ),
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
            type='configs/motion_tokenizers/homi_vqvae/homi_vq_64_2048code_1536dim_3depth.py',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='work_dirs/homi_gate_encoder_out_gate_64_2048code_1536dim_3depth/best_MPJPE_epoch_5000.pth'
            )
        ),
        audio=dict(
            type='configs/wav_tokenizer/wav_tokenizer_large_unify_600_24k_4096.py'
        )
    ),
    pretrained_lm='checkpoints/Llama-3.2-3B-Instruct-uncensored',
    lora_config=None,
    freeze_emb='no'
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=1e-5, betas=[0.9, 0.99], weight_decay=0.0)
)

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=500)
val_cfg = dict(
    type='ValLoop', fp16=True, dtype='bf16')  # must set dtype to bf16, if not, the distance in vqvae will be infinity.
test_cfg = val_cfg

train_dataloader = dict(
    num_workers=0,
    batch_size=16,
    persistent_workers=False,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
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
                task_mode='auto',
                serialize_data=False,
                min_duration=1.,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='auto',
                min_duration=1.,
                serialize_data=False,
                pipeline=pipeline),
        ]),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

val_dataloader = dict(
    num_workers=0,
    batch_size=48,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=False,

    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='auto',
        min_duration=1.,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

val_evaluator = [dict(type='ValidationMetric', collect_device='gpu')]
test_evaluator = val_evaluator

vis_backends = [
    dict(type='JointsVisBackend', name='motion'),
    dict(type='TextVisBackend', name='text'),
    dict(type='AudioVisBackend', name='audio'),
    dict(type='MergeAudioVideoVisBackend', name='merge_audio_video')
]
# visualizer = None
visualizer = dict(
    _delete_=True,
    type='MotionLLaMAVisualizer',
    vis_backends=vis_backends
)
# custom_hooks = None
custom_hooks = [dict(type='BasicVisualizationHook', interval=20, in_batch_interval=50)]

default_hooks = dict(
    logger=dict(interval=20),
    checkpoint=dict(
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=5,
        save_best=['val_loss'],
        rule='less',
    ),
)
