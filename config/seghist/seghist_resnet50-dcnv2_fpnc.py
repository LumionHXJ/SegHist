_base_ = [
    '_base_db_seghist_resnet50-dcnv2_fpnc.py',
    './pipeline/seghist_pipeline_basic.py',
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
        verify_meta=False,
        pipeline=_base_.train_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        verify_meta=False,
        pipeline=_base_.test_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=val_list,
        verify_meta=False,
        pipeline=_base_.test_pipeline))

auto_scale_lr = dict(base_batch_size=16) 

test_evaluator = [dict(type='HmeanIOUMetric',
                      pred_score_thrs=dict(start=0.6, stop=1.0, step=0.05),
                      prefix='Iacc',
                      match_iou_thr=0.5)]
val_evaluator = test_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', 
                    interval=5))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


param_scheduler = [dict(type='ReduceOnPlateauLR', 
                         rule='greater',
                         monitor='Iacc/recall',
                         factor=0.3, 
                         patience=1,
                         threshold=1e-4)] # use arg last_step when resuming optim!'''
#param_scheduler = dict(
#   type='MultiStepLR', by_epoch=True, milestones=[80, 128], gamma=0.1)

custom_imports = dict(
    imports=['seghist'], # not support relative import
    allow_failed_imports=False)


optim_wrapper = dict(
    type='AmpOptimWrapper',
<<<<<<< HEAD
    optimizer=dict(type='AdamW', lr=1e-4))

#resume = False
#load_from = './work_dirs_icdar2019/pse-seghist/epoch_600.pth'
=======
    optimizer=dict(type='AdamW', lr=1e-4)) # 1e-3

#resume = False
load_from = './work_dirs_chdac/seghist/final_9712.pth'
>>>>>>> origin/main
