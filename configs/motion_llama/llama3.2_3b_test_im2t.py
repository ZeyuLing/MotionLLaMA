_base_ = 'llama3.2_3b_instruct_it2m_im2t.py'
default_hooks = dict(
    checkpoint=dict(
        save_best=['im2t_mm_dist'],
        less_keys=['im2t_mm_dist'],
        rule='less',
        greater_keys=['im2t_r_precision_top_1',
                      'im2t_r_precision_top_2',
                      'im2t_r_precision_top_3',
                      'inter_rouge_l',
                      'inter_CIDEr',
                      'inter_bert_f1',
                      'inter_bleu_1',
                      'inter_bleu_4']
    ),
)
pipeline = [
    dict(type='LoadHm3dTxt', keys=['union_caption'], min_duration=1.),
    dict(type='LoadMotionVector', keys=['interhuman', 'interactor_interhuman'],
         save_keys=['motion', 'interactor_motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion', 'interactor_motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion', 'interactor_motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion', 'interactor_motion'], clip_duration=10.),
    dict(type='LoadConversation'),
    dict(type='PackInputs', keys=['motion', 'interactor_motion'],
         meta_keys=['union_caption_list']
         )
]
model = dict(
    init_cfg=None
)

val_dataloader = dict(
    _delete_=True,
    num_workers=0,
    batch_size=128,
    batch_sampler=dict(type='TaskBatchSampler', drop_last=False),
    sampler=dict(type='DefaultSampler'),
    persistent_workers=False,
    dataset=dict(
        type='MultiModalLlamaDataset',
        ann_file='test.json',
        metainfo=dict(dataset_type='MotionHub test subset', task_name='MotionLlama test'),
        data_root='data/motionhub',
        task_mode='preset',
        tasks='im2t',
        min_duration=1.,
        serialize_data=False,
        pipeline=pipeline),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='IM2TMetric',
         tmr_model=dict(
             type='configs/tmr/tmr_inter.py',
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint='work_dirs/tmr_inter/best_mm_dist_epoch_1540.pth')))
]

test_evaluator = val_evaluator
val_cfg = dict(
    _delete_=True,
    type='ValLoop', fp16=True, dtype='bf16')  # must set dtype to bf16, if not, the distance in vqvae will be infinity.
test_cfg = val_cfg

custom_hooks = [dict(type='BasicVisualizationHook', interval=10, in_batch_interval=50)]
