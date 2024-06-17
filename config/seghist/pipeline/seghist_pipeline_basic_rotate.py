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
        type='RandomChoiceResize',
        scales=[(1333, 704), (1333, 736), (1333, 768), (1333, 800),
                (1333, 832), (1333, 864), (1333, 896)],
        keep_ratio=True,
        clip_object_border=False), # clip the object when outside border
    dict(type='TextDetRandomCrop', target_size=(640, 640)), 
    dict(type='PadDivisor', size_divisor=32), # PadDivisor must placed at last!
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_shape'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=False,
        with_label=True),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True,
        clip_object_border=True),
    dict(type='RandomRotate', max_angle=15),
    dict(type='PadDivisor', size_divisor=32), # PadDivisor must placed at last!
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 
                   'img_shape', 'scale_factor',
                   'valid_shape', 'instances'))
]
model = dict(
    det_head=dict(
        postprocessor=dict(
            rescale_fields=[], # test time: first load annotations then transform
        )
    )
)