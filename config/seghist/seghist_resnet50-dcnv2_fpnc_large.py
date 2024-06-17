_base_ = [
    '_base_seghist_resnet50-dcnv2_fpnc.py',
    './pipeline/seghist_pipeline_largescale.py',
    '../_base_/textdet_runtime.py',
    '../_base_/datasets/iacc2022_chdac.py',
    '../_base_/schedules/schedule_adam_600e.py',
]

# dataset settings
train_list = _base_.train_list
test_list = _base_.test_list
val_list = _base_.val_list

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=val_list,
        pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16) # 对不同大小的batch_size应用不同的系数，但是设置学习率可以根据base_batch设置

val_evaluator = [dict(type='HmeanIOUMetric',
                      pred_score_thrs=dict(start=0.6, stop=1.0, step=0.1))]
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=10)

'''
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[50, 125], gamma=0.1)
'''
param_scheduler = [dict(type='LinearLR',
                        start_factor=1e-5,
                        by_epoch=False, 
                        begin=0,
                        end=125),
                    dict(type='ReduceOnPlateauLR', 
                         rule='greater',
                         factor=0.33, 
                         patience=1,
                         threshold=1e-4)] # use arg last_step when resuming optim!

custom_imports = dict(
    imports=['seghist'], # not support relative import
    allow_failed_imports=False)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3))