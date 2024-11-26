"""
    Use mid layers, no lstm
"""
_base_=['../base_vqvae.py']
pipeline = [
    dict(type='LoadMotionVector', keys=['interhuman'], save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion'], fps_key='fps', ori_fps_key='ori_fps',
         tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),

    dict(type='RandomCrop', keys=['motion'], clip_len=64),
    dict(type='PackInputs', keys=['motion'],
         meta_keys=['has_hand', 'rot_type', 'motion_path', 'start_frame', 'num_joints'])
]

train_dataloader = dict(
    num_workers=4,
    batch_size=3200,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['dataset', 'version', 'dataset_type', 'task_name'],
        datasets=[
            dict(
                type='MotionDataset',
                ann_file='train.json',
                metainfo=dict(dataset_type='Motion-X train subset', task_name='Motion VQ-VAE training'),
                data_root='data/motionhub',
                data_prefix=dict(
                    interhuman_path='',
                ),
                min_duration=2.2,
                pipeline=pipeline),
            dict(
                type='MotionDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='Motion-X val subset', task_name='Motion VQ-VAE training'),
                data_root='data/motionhub',
                data_prefix=dict(
                    interhuman_path='',
                ),
                min_duration=2.2,
                pipeline=pipeline),
        ]
    )
)

val_dataloader = dict(
    num_workers=0,
    batch_size=3200,
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=False,

    dataset=dict(
        type='MotionDataset',
        ann_file='test.json',
        metainfo=dict(dataset_type='Motion-X test subset', task_name='Motion VQ-VAE test'),
        data_root='data/motionhub',
        data_prefix=dict(
            interhuman_path='',
        ),
        min_duration=2.2,
        pipeline=pipeline)
)

test_dataloader = val_dataloader


model = dict(
    type='MotionVQVAE',
    quantizer=dict(
        type='ResidualVQ',
        dim=512,
        num_quantizers=1,
        layer_quantizer=dict(
            type='VectorQuantizer',
            codebook_size=16384,
            dim=512,
            decay=0.99,
            kmeans_init=True,
            kmeans_iters=200,
        )
    ),
    encoder=dict(
        type='BaseEncoder',
        in_channels=156,
        out_channels=512,
        block_out_channels=(512, 512, 512),
        layers_per_block=3,
        layers_mid_block=3,
        norm_type='group',
        activation_type='silu'
    ),
    decoder=dict(
        type='HoMiDecoder',
        in_channels=512,
        out_channels=156,
        block_out_channels=(512, 512, 512),
        layers_per_block=3,
        layers_mid_block=3,
        norm_type='group',
        activation_type='silu'
    ),
    data_preprocessor=dict(
        type='MotionDataPreprocessor',
        norm=dict(
            feat_bias=1.0,
            mean_key='pos_mean',
            std_key='pos_std',
            norm_path='data/motionhub/statistics/interhuman.pkl'),
        vec2joints_fn='interhuman2joints',
        vec2rotation_fn='dummy_vec2rotation'
    ),
    loss_cfg=dict(
        type='JointsWholeBodyLoss',
        recons_type='l1_smooth',
    )
)
