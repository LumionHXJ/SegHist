<<<<<<< HEAD
data_root = 'data/historical_document/IACC2022_CHDAC/official_dataset'
=======
data_root = './data/historical_document/IACC2022_CHDAC/official_dataset'
>>>>>>> origin/main

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
    pipeline=None)

<<<<<<< HEAD
data_root = 'data/historical_document/IACC2022_CHDAC/private_dataset'
=======
data_root = './data/historical_document/IACC2022_CHDAC/private_dataset'
>>>>>>> origin/main

chdac_train_private1 = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='dataset_1/train/ocr_train.json',
    data_prefix=dict(img_path='dataset_1/train/image'),
    pipeline=None)

chdac_test_private1 = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='dataset_1/test/ocr_test.json',
    data_prefix=dict(img_path='dataset_1/test/image'),
    test_mode=True,
    pipeline=None)

chdac_train_private2 = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='dataset_2/train/ocr_train.json',
    data_prefix=dict(img_path='dataset_2/train/image'),
    pipeline=None)

chdac_test_private2 = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='dataset_2/test/ocr_test.json',
    data_prefix=dict(img_path='dataset_2/test/image'),
    test_mode=True,
    pipeline=None)

chdac_train_private3 = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='dataset_3/train/ocr_train.json',
    data_prefix=dict(img_path='dataset_3/train/image'),
    pipeline=None)

chdac_test_private3 = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='dataset_3/test/ocr_test.json',
    data_prefix=dict(img_path='dataset_3/test/image'),
    test_mode=True,
    pipeline=None)

data_root = './data/historical_document/ICDAR2019HDRC_Chinese/'

icdar2019_trainset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='train_label_comp.json',
    data_prefix=dict(img_path='images'),
    pipeline=None)

icdar2019_testset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='test_label_comp.json',
    test_mode=True,
    data_prefix=dict(img_path='images'),
    # indices=50 在更小的数据集上尝试验证效果
    pipeline=None)

data_root = './data/historical_document/MTHv2/MTHv2'

mthv2_trainset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='train_label.json',
    pipeline=None)

mthv2_testset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='test_label.json',
    test_mode=True,
    pipeline=None)

data_root = './data/historical_document/MTHv2/twist_MTHv2'

twist_mthv2_trainset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='train_label.json',
    pipeline=None)

twist_mthv2_testset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='test_label.json',
    test_mode=True,
    pipeline=None)

data_root = './data/historical_document/Huayan'

huayan_trainset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='train_label.json',
    data_prefix=dict(img_path='images'),
    pipeline=None
)

huayan_testset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='test_label.json',
    data_prefix=dict(img_path='images'),
    test_mode=True,
    pipeline=None
)
# 没有使用mthv2
train_list = [chdac_train_preliminary, chdac_train_final, icdar2019_trainset, 
              twist_mthv2_trainset, huayan_trainset,
              chdac_train_private1, chdac_train_private2, chdac_train_private3]
test_list = [chdac_test, icdar2019_testset, huayan_testset, twist_mthv2_testset,
             chdac_test_private1, chdac_test_private2, chdac_test_private3]
val_list = test_list