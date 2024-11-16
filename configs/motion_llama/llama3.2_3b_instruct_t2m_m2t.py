_base_ = 'llama3.2_3b_pretrain_t2m_m2t.py'
model=dict(
    is_pretrain_stage=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/llama3.2_3b_pretrain_t2m_m2t/iter_165000.pth'
    )
)