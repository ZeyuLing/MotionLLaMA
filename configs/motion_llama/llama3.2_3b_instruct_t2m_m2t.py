_base_ = 'llama3.2_3b_pretrain_t2m_m2t.py'
model = dict(
    is_pretrain_stage=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/llama3.2_3b_pretrain_t2m_m2t/iter_165000.pth'
    )
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=2e-6, betas=[0.9, 0.99], weight_decay=0.0)
)
