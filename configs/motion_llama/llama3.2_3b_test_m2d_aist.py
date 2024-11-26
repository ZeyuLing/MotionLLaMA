_base_ = 'llama3.2_3b_test_m2d.py'


val_dataloader = dict(
    num_workers=0,
    batch_size=64,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=False,
    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='aist/test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='m2d',
        min_duration=1.,
        serialize_data=False),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='M2DMetric',
         full_body=False)
]

test_evaluator = val_evaluator

val_cfg = dict(
    _delete_=True,
    type='ValLoop', fp16=True, dtype='bf16')  # must set dtype to bf16, if not, the distance in vqvae will be infinity.
test_cfg = val_cfg

custom_hooks = [dict(type='BasicVisualizationHook', interval=1, in_batch_interval=1)]