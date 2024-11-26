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
        save_best=['MPJPE'],
        less_keys=['MPJPE', 'P-MPJPE', 'N-MPJPE',
                   'MPJPE_body', 'P-MPJPE_body', 'N-MPJPE_body',
                   'MPJPE_hand', 'P-MPJPE_hand', 'N-MPJPE_hand',
                   'ADE', 'FDE'],
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

train_cfg = dict(by_epoch=True, max_epochs=10000, val_interval=100)
# train_cfg = dict(by_epoch=False, max_iters=10000, val_interval=5)
val_cfg = dict(type='ValLoop', fp16=True, dtype='bfloat16')
test_cfg = val_cfg

val_evaluator = [
    dict(
        type='MPJPE',
        gt_key='gt_joints',
        pred_key='pred_joints',
        has_hand_key='has_hand',
        alignment='none',
        device='cuda',
        whole_body=True
    ),
    dict(
        type='MPJPE',
        gt_key='gt_joints',
        pred_key='pred_joints',
        has_hand_key='has_hand',
        alignment='scale',
        device='cuda',
        whole_body=True
    ),
    dict(
        type='MPJPE',
        gt_key='gt_joints',
        pred_key='pred_joints',
        has_hand_key='has_hand',
        alignment='procrustes',
        device='cuda',
        whole_body=True
    ),
    dict(
        type='ADE',
        gt_key='gt_joints',
        pred_key='pred_joints',
    ),
    dict(
        type='FDE',
        gt_key='gt_joints',
        pred_key='pred_joints',
    ),
]

test_evaluator = val_evaluator

vis_backends = [dict(type='JointsVisBackend')]
visualizer = dict(
    _delete_=True,
    type='MotionVisualizer',
    fn_key=['motion_path'],
    vis_backends=vis_backends,
    motion_keys=['pred_joints', 'gt_joints'],
    is_smpl=False
)
custom_hooks = [dict(type='BasicVisualizationHook', interval=3, in_batch_interval=1000)]
