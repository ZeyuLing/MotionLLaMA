_base_ = 'llama3.2_3b_pretrain_unify.py'

model = dict(
    is_pretrain_stage=False,
    init_cfg=dict(type='Pretrained',
                  checkpoint='work_dirs/llama3.2_3b_pretrain/iter_240000.pth')
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=2e-6, betas=[0.9, 0.99], weight_decay=0.0)
)


train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=500)
