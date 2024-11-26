_base_ = 'llama3.2_3b_pretrain_a2g_g2a.py'

model = dict(
    is_pretrain_stage=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/llama3.2_3b_pretrain_a2g_g2a/iter_65000.pth'
    )
)