data_root = './data/historical_document/IACC2022_CHDAC/official_dataset'

chdac_train_preliminary = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='preliminary/train/ocr_train.json',
    data_prefix=dict(img_path='preliminary/train/image'),
    pipeline=None)

chdac_train_final = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='final/train/ocr_train.json',
    data_prefix=dict(img_path='final/train/image'),
    pipeline=None)

chdac_test = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='final/test/ocr_test.json',
    data_prefix=dict(img_path='final/test/image'),
    test_mode=True,
    #indices=150, #在更小的数据集上尝试验证效果
    pipeline=None)

train_list = [chdac_train_preliminary, chdac_train_final]
test_list = [chdac_test]
val_list = test_list