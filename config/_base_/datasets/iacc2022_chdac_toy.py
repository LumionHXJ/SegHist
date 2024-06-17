data_root = './data/historical_document/IACC2022_CHDAC/official_dataset'

chdac_toy = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='final/train/ocr_toy.json',
    data_prefix=dict(img_path='final/train/image'),
    pipeline=None)

train_list = [chdac_toy]
test_list = [chdac_toy]
val_list = [chdac_toy]