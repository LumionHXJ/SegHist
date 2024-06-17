r = 0. # shrink_ratio
stretch_ratio = 2. # 1.5
model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPNC',
        in_channels=[256, 512, 1024, 2048],
        lateral_channels=256,
        asf_cfg=dict(attention_type='ScaleChannelSpatial')),
    det_head=dict(
        type='SegHistHead',
        in_channels=256,  
        num_blocks=3,
        num_query=8,
        shallow_channels=128,
        embedding_channels=128, # = shallow channels
        use_dyrelu=True,
        dyrelu_mode='awared',
        with_m2f_mask=True,
        with_sigmoid=False,
        module_loss=dict(type='SegHistModuleLoss',
                         shrink_ratio=r,
                         stretch_ratio=stretch_ratio),
        postprocessor=dict(
            type='IterExpandPostprocessor', 
            text_repr_type='poly',
            shrink_ratio=r,
            stretch_ratio=stretch_ratio,
            epsilon_ratio=0.002,
            mask_thr=0.6)),
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))
