# optimizer
# 不同层采用不同学习率，下调学习率后scheduler也要调整
optim_wrapper = dict(type='OptimWrapper', 
                     optimizer=dict(type='AdamW', lr=1e-3))
train_cfg = dict(type='EpochBasedTrainLoop', 
                 max_epochs=200, 
                 val_interval=10)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


#param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[50, 120], gamma=0.1)
