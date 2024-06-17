data_root = './data/historical_document/MTHv2/MTHv2'

trainset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='train_label.json',
    pipeline=None)

testset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='test_label.json',
    test_mode=True,
    # indices=50 在更小的数据集上尝试验证效果
    pipeline=None)

train_list = [trainset]
test_list = [testset]
val_list = [testset]