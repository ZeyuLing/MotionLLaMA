_base_ = 'momask_temporal_transformer.py'

model = dict(
    _delete_=True,
    type='Momask',
    temp_transformer=dict(
        type='configs/momask/momask_temporal_transformer.py',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/momask_temporal_transformer/best_fid_epoch_350.pth'
        )
    ),
    res_transformer=dict(
        type='configs/momask/momask_residual_transformer.py',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/momask_residual_transformer/best_MPJPE_epoch_550.pth'
        )
    ),
    data_preprocessor=dict(
        type='MotionDataPreprocessor',
        non_concatenate_keys=['num_frames'],
        pad_module=dict(
            type='Pad1D',
            pad_to_max=True,
        ),
        motion_keys=['motion'],
        norm=dict(
            feat_bias=1.0,
            mean_key='pos_mean',
            std_key='pos_std',
            norm_path='data/motionhub/statistics/interhuman.pkl'),
        vec2joints_fn='interhuman2joints',
        vec2rotation_fn='dummy_vec2rotation'
    )

)

