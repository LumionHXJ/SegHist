default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
randomness = dict(seed=None)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10), #
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', 
                    interval=5, 
                    max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False),
)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True), 
                dict(type='SyncBuffersHook')]

# Logging
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

# Evaluation
val_evaluator = [dict(type='E2EHmeanIOUMetric'), 
                dict(type='HmeanIOUMetric'),
                dict(type='E2ENEDMetric')]
test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type='LocalVisBackend'), 
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='TextSpottingLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends)
