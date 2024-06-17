import json
import os
import random

def main(root, label, output, sample=1):
    label = os.path.join(root, label)
    with open(label) as f:
        ann_file = json.load(f)
    datas = dict(metainfo=ann_file['metainfo'])
    datas['data_list']=random.sample(ann_file['data_list'], sample)
    with open(os.path.join(root, output), mode='w') as f:
        json.dump(datas, f)    


root = './data/historical_document/IACC2022_CHDAC/official_dataset'
label ='final/train/ocr_train.json'
output = 'final/train/ocr_toy.json'
main(root, label, output)