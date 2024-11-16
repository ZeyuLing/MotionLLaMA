_base_ = 'llama3.2_3b_pretrain_d2m_m2d.py'

model = dict(
    is_pretrain_stage=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/llama3.2_3b_pretrain_d2m_m2d/iter_25000.pth'
    )
)

train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=1000)
