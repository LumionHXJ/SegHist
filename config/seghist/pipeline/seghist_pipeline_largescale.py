train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=False,
        with_polygon=True,
        with_label=True),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=0.12549019607843137,
        saturation=0.5),
    dict(type='RandomFlip', 
         prob=0.5, 
         direction=['horizontal', 'vertical']), # both direction
    dict(
        type='RandomRotate',
        max_angle=10  # [-10, 10]
    ),
    dict(
        type='MultiScaleResizeShorterSide',
        fixed_longer_side=2000,
        shorter_side_ratio=(0.8, 1.2),
        clip_object_border=True), # clip the object when outside border
    dict(type='RatioAwareCrop', crop_ratio=(0.7, 0.5)), 
    dict(type='PadDivisor', size_divisor=32), # PadDivisor must placed at last!
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_shape'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='Resize',
        scale=(1600, 1600),
        keep_ratio=True,
        clip_object_border=False),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=False,
        with_label=True),
    dict(type='PadDivisor', size_divisor=32), # PadDivisor must placed at last!
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 
                   'img_shape', 'scale_factor',
                   'valid_shape', 'instances'))
]