import json
from tqdm import tqdm
import os
import warnings

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from seghist.utils.image_utils import ImageToolkits

warnings.filterwarnings('ignore', category=np.RankWarning)

def get_image_size(file_path):
    with Image.open(file_path) as img:
        return img.size

def clean_redundant_points(poly):
    '''
    clean redundant points when len(poly) % 2 != 0 
    '''
    kmeans = KMeans(((len(poly)+1) // 2), n_init=3).fit(poly[:, 1:])
    cluster_counts = np.bincount(kmeans.labels_)
    for i in range(len(cluster_counts)):
        if cluster_counts[i] == 1:
            for idx, l in enumerate(kmeans.labels_):
                if l==i:
                    return np.concatenate([poly[:idx], poly[idx+1:]], axis=0)

def main(root, 
         label_list, 
         data_prefix_list, 
         output_list,
         separate_entry=True):
    metainfo = {"dataset_type": "TextDetDataset", "task_name": "textdet", 
                "category": [{"id": 0, "name": "single_entry_text",
                              "id": 1, "name": "double_entry_text"}]}
    for label, data_prefix, output in zip(label_list, data_prefix_list, output_list):
        label = os.path.join(root, label)
        datas = dict(metainfo=metainfo, data_list=[])
        with open(label) as f:
            ann_file = json.load(f)
        for img_path, instances in tqdm(ann_file.items()):
            data = dict(img_path=img_path, instances=[])
            data['width'], data['height'] = get_image_size(os.path.join(root, data_prefix, img_path))
            for idx, inst in enumerate(instances):
                # clean redundant points, if not in pair.
                if len(inst['points']) % 4 != 0:
                    poly = np.array(inst['points']).reshape(-1, 2)
                    poly = clean_redundant_points(poly)
                    instances[idx]['points'] = poly.reshape(-1).tolist()
            if separate_entry:
                hi = ImageToolkits([np.array(d['points']).reshape(-1, 2) for d in instances],
                                    np.array((data['height'], data['width'])),
                                    img_path,
                                    texts=[d['transcription'] for d in instances])
                hi.process()
                data['instances'] = hi.output_json()
            else:
               for idx, inst in enumerate(instances):
                data["instances"].append(dict(
                    ignore=False,
                    text=inst['transcription'],
                    bbox_label=0,
                    polygon=inst['points']
                ))
            datas['data_list'].append(data)
        with open(os.path.join(root, output), mode='w') as f:
            json.dump(datas, f)    

root = './data/historical_document/IACC2022_CHDAC/private_dataset'

label_list = ['dataset_1/test/label_test.json', 
              'dataset_1/train/label_train.json',
              'dataset_2/test/label_test.json', 
              'dataset_2/train/label_train.json',
              'dataset_3/test/label_test.json', 
              'dataset_3/train/label_train.json',]
data_prefix_list = ['dataset_1/test/image', 
                    'dataset_1/train/image',
                    'dataset_2/test/image', 
                    'dataset_2/train/image',
                    'dataset_3/test/image', 
                    'dataset_3/train/image']
output_list = ['dataset_1/test/ocr_test.json',
               'dataset_1/train/ocr_train.json',
               'dataset_2/test/ocr_test.json',
               'dataset_2/train/ocr_train.json',
               'dataset_3/test/ocr_test.json',
               'dataset_3/train/ocr_train.json']
              
main(root, label_list, data_prefix_list, output_list, separate_entry=False)
