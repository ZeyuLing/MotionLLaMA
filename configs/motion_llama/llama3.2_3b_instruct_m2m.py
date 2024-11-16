_base_ = 'llama3.2_3b_pretrain_m2m.py'

model = dict(
    is_pretrain_stage=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/llama3.2_3b_pretrain_m2m/iter_105000.pth'
    )
)
