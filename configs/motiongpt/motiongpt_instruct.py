_base_ = 'motiongpt_pretrain.py'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=50,
        by_epoch=True,
        max_keep_ckpts=10,
        save_best=['t2m_mm_dist', 'm2t_mm_dist', ],
        less_keys=['t2m_mm_dist', 'm2t_mm_dist',
                   't2m_fid', 'm2t_fid', 'pred_fid', 'inbetween_fid',
                   'pred_ade', 'pred_fde',
                   'inbetween_ade'],
        rule='less',
        greater_keys=['m2t_r_precision_top_1',
                      'm2t_r_precision_top_2',
                      'm2t_r_precision_top_3',
                      't2m_r_precision_top_1',
                      't2m_r_precision_top_2',
                      't2m_r_precision_top_3',
                      'rouge_l',
                      'CIDEr',
                      'bert_f1',
                      'bleu_1',
                      'bleu_4']
    ),
)

pipeline = [
    dict(type='LoadHm3dTxt', keys=['caption'], min_duration=1.5),
    dict(type='LoadMotionVector', keys=['interhuman'],
         save_keys=['motion'], data_source='interhuman'),
    dict(type='MotionResampleFPS', keys=['motion'],
         fps_key='fps', ori_fps_key='ori_fps', tgt_fps=30, data_source='interhuman'),
    dict(type='InterhumanTransform', keys=['motion'], no_velocity=True, relative_joints=False,
         no_rotation=True, no_feet_contact=True),
    dict(type='MotionAudioRandomCrop', motion_keys=['motion'],
         clip_duration=10., fps='fps'),
    dict(type='SplitPrediction'),
    dict(type='SplitInbetween'),
    dict(type='LoadConversation'),
    dict(type='PackInputs', keys=['motion',
                                  'past_motion', 'middle_motion', 'future_motion']),
]

model = dict(
    init_cfg=dict(type='Pretrained',
                  checkpoint='work_dirs/motiongpt_pretrain/iter_150000.pth'),

)

train_dataloader = dict(
    dataset=dict(
        datasets=[
            dict(
                type='MultiModalLlamaDataset',
                ann_file='train.json',
                metainfo=dict(dataset_type='MotionHub train subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks=['t2m', 'm2t', 'pred', 'inbetween'],
                
                serialize_data=False,
                min_duration=1.5,
                pipeline=pipeline),
            dict(
                type='MultiModalLlamaDataset',
                ann_file='val.json',
                metainfo=dict(dataset_type='MotionHub val subset', task_name='MotionLlama pretraining'),
                data_root='data/motionhub',
                task_mode='preset',
                tasks=['t2m', 'm2t', 'pred', 'inbetween'],
                
                min_duration=1.5,
                serialize_data=False,
                pipeline=pipeline),
        ]),
    collate_fn=dict(
        type='pseudo_collate'
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='test.json',
        task_mode='preset',
        tasks=['t2m', 'pred', 'inbetween'],
        
        min_duration=1.5,
        pipeline=pipeline
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='T2MMetric',
         tmr_model=dict(
             type='configs/tmr/tmr_motionhub.py',
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth'))),
    # dict(type='M2TMetric',
    #      tmr_model=dict(
    #          type='configs/tmr/tmr_motionhub.py',
    #          init_cfg=dict(
    #              type='Pretrained',
    #              checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth'))),
    dict(type='InbetweenMetric',
         tmr_model=dict(
             type='configs/tmr/tmr_motionhub.py',
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth'))),
    dict(type='PredMetric',
         tmr_model=dict(
             type='configs/tmr/tmr_motionhub.py',
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint='work_dirs/tmr_motionhub/best_mm_dist_epoch_1380.pth'))),
]
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=50)
