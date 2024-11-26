_base_ = 'llama3.2_3b_pretrain_it2m_im2t.py'
model = dict(
    is_pretrain_stage=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/data/lzy/projects/motion_llama/work_dirs/llama3.2_3b_pretrain_it2m_im2t/iter_130000.pth'
    )
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=2e-6, betas=[0.9, 0.99], weight_decay=0.0)
)
