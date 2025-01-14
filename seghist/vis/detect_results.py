import os
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2

from mmocr.utils.polygon_utils import *
from mmocr.apis import TextDetInferencer

from seghist.utils import iou

def get_prediction(inferencer):
    img_root = './data/historical_document/IACC2022_CHDAC/official_dataset/final/test'
    with open("./data/historical_document/IACC2022_CHDAC/official_dataset/final/test/ocr_test.json") as f:
        datas = json.load(f)

    with open("./results/pred.txt", mode='w') as f:
        for data in tqdm(datas['data_list']):
            ret_polys = inferencer(os.path.join(img_root, 'image', data['img_path']))['polygons']
            ret_polys = [np.array(p).reshape(-1, 2) for p in ret_polys]
            gt_polys = []
            for instance in data['instances']:
                gt_polys.append(np.array(instance['polygon']).reshape(-1,2))
            
            match_gt = [False for _ in gt_polys]
            match_pred = [False for _ in ret_polys]

            for igt, gt in enumerate(gt_polys):
                for ip, p in enumerate(ret_polys):
                    if not match_pred[ip] and iou(p, gt) > 0.5:
                        match_pred[ip] = True
                        match_gt[igt] = True
                        break
            
            # compute metric
            tp = np.sum(np.where(match_pred, 1, 0))
            fp = np.sum(np.where(match_pred, 0, 1))
            fn = np.sum(np.where(match_gt, 0, 1))
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            f.write(f'{data["img_path"]}\t{precision}\t{recall}\t{f1}\n')

def main(use_hard_example=True):
    inferencer = TextDetInferencer(ckpt='./work_dirs_chdac/pse_seghist/epoch_150.pth', 
                               config='./config/seghist/seghist_resnet50-dcnv2_fpnc.py',
                               device='cuda:0')
    img_root = './data/historical_document/IACC2022_CHDAC/official_dataset/final/train'
    with open("./data/historical_document/IACC2022_CHDAC/official_dataset/final/train/ocr_train.json") as f:
        datas = json.load(f)
    if use_hard_example:
        if not os.path.exists("./results/pred.txt"):
            get_prediction(inferencer)
        hard_example = []
        with open("./results/pred.txt") as f:
            for l in f.readlines():
                l = l.strip().split()
                if float(l[-1]) < 0.8:
                    hard_example.append(l[0])
        while True:
            data = np.random.choice(datas['data_list'])
            if data['img_path'] in hard_example:
                break
    else:
        data = np.random.choice(datas['data_list'])
    img = cv2.imread(os.path.join(img_root,'image', data['img_path']))
    ret_polys = inferencer(os.path.join(img_root,'image',  data['img_path']))['polygons']
    ret_polys = [np.array(p).reshape(-1, 2) for p in ret_polys]
    print(data['img_path'])

    # missed
    mask = np.zeros(img.shape[:2])
    for instance in tqdm(data['instances']):
        poly = np.array(instance['polygon']).reshape(-1,2)
        for p in ret_polys:
            if iou(p, poly) > 0.5:
                break
        else:
            cv2.drawContours(mask, [poly], -1, 1, 5)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.imshow(mask, alpha=0.4)
    plt.axis('off')
    plt.savefig(f"./seghist/vis/results/{data['img_path'].split('.')[0]}_miss.jpg")

    # false positive
    mask = np.zeros(img.shape[:2])
    for p in ret_polys:
        for instance in data['instances']:
            poly = np.array(instance['polygon']).reshape(-1,2)    
            if iou(p, poly) > 0.5:
                break
        else:
            cv2.drawContours(mask, [p.astype(np.int32)], -1, 1, 5)
        
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.imshow(mask, alpha=0.4)
    plt.axis('off')
    plt.savefig(f"./seghist/vis/results/{data['img_path'].split('.')[0]}_fp.jpg")

    # full detect
    mask = np.zeros(img.shape[:2])
    for i, p in enumerate(ret_polys):
        cv2.drawContours(mask, [p.astype(np.int32)], -1, i % 5 + 1, 3)
        
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.imshow(mask, alpha=0.4, cmap='jet')
    plt.axis('off')
    plt.savefig(f"./seghist/vis/results/{data['img_path'].split('.')[0]}_detect.jpg")

    # ground truth
    mask = np.zeros(img.shape[:2])
    for i, instance in enumerate(data['instances']):
        poly = np.array(instance['polygon']).reshape(-1,2)   
        cv2.drawContours(mask, [poly], -1, i % 5 + 1, 3)
        
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.imshow(mask, alpha=0.4, cmap='jet')
    plt.axis('off')
    plt.savefig(f"./seghist/vis/results/{data['img_path'].split('.')[0]}_gt.jpg")
