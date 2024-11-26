_base_ = 'llama3.2_3b_lora_pretrain_unify.py'

model = dict(
    is_pretrain_stage=False,
    init_cfg=dict(type='Pretrained',
                  checkpoint='work_dirs/llama3.2_3b_pretrain/iter_240000.pth'),
    # freeze_emb='all'
)

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=500)
