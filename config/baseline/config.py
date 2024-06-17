_base_ = [
    './model/dbnetpp.py',
    './pipeline.py',
    '../_base_/textdet_runtime.py',
    '../_base_/datasets/iacc2022_chdac.py'
]

# dataset settings
train_list = _base_.train_list
test_list = _base_.test_list
val_list = _base_.val_list

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=val_list,
        pipeline=_base_.test_pipeline))

auto_scale_lr = dict(base_batch_size=16) 

test_evaluator = [dict(type='HmeanIOUMetric',
                      prefix='Iacc',
                      match_iou_thr=0.5,
                      pred_score_thrs=dict(start=0.3, stop=0.9, step=0.05)),
                    dict(type='HmeanIOUMetric',
                      prefix='Iacc75',
                      match_iou_thr=0.75,
                      pred_score_thrs=dict(start=0.3, stop=0.9, step=0.05))]
val_evaluator = test_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=250, val_interval=10)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', 
                    interval=5, 
                    max_keep_ckpts=10))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

<<<<<<< HEAD
=======
'''
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[110], gamma=0.1)
'''
>>>>>>> origin/main
param_scheduler = [dict(type='ReduceOnPlateauLR', 
                         rule='greater',
                         monitor='Iacc/recall',
                         factor=0.3, 
                         patience=1,
                         threshold=1e-4)] # use arg last_step when resuming optim!

custom_imports = dict(
    imports=['seghist'], # not support relative import
    allow_failed_imports=False)


optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4))

<<<<<<< HEAD
=======
'''
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3))'''

>>>>>>> origin/main
#resume = True
#load_from = '/home/huxingjian/model/mmocr/projects/SegHist/work_dirs_baseline/dbnetpp/epoch_5.pth'