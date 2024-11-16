_base_ = 'momask_temporal_transformer.py'

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=200,
                ignore_last=True,
                interval_exp_name=1000),
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

model = dict(
    type='MomaskResidualTransformer',
)

val_evaluator = [
    dict(
        type='MPJPE',
        gt_key='joints',
        pred_key='pred_joints',
        has_hand_key='has_hand',
        alignment='none',
        device='cuda',
        whole_body=True
    ),
    dict(
        type='MPJPE',
        gt_key='joints',
        pred_key='pred_joints',
        has_hand_key='has_hand',
        alignment='scale',
        device='cuda',
        whole_body=True
    ),
    dict(
        type='MPJPE',
        gt_key='joints',
        pred_key='pred_joints',
        has_hand_key='has_hand',
        alignment='procrustes',
        device='cuda',
        whole_body=True
    ),
    dict(
        type='ADE',
        gt_key='joints',
        pred_key='pred_joints',
    ),
    dict(
        type='FDE',
        gt_key='joints',
        pred_key='pred_joints',
    ),
]

test_evaluator = val_evaluator

# train_cfg = dict(_delete_=True, by_epoch=False, max_iters=100000, val_interval=1)
