data_root = './data/historical_document/ICDAR2019HDRC_Chinese/'

trainset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='train_label.json',
    data_prefix=dict(img_path='images'),
    pipeline=None)

valset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='val_label.json',
    test_mode=True,
    data_prefix=dict(img_path='images'),
    # indices=50 在更小的数据集上尝试验证效果
    pipeline=None)

testset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='test_label.json',
    test_mode=True,
    data_prefix=dict(img_path='images'),
    # indices=50 在更小的数据集上尝试验证效果
    pipeline=None)

train_list = [trainset]
val_list = [valset]
test_list = [testset]