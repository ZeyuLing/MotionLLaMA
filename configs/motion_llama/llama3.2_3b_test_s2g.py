_base_ = 'llama3.2_3b_instruct_a2g_g2a.py'

model = dict(
    init_cfg=None
)
custom_hooks = [dict(type='BasicVisualizationHook', interval=1, in_batch_interval=1)]