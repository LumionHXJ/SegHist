import json
import os
from tqdm import tqdm

import numpy as np
from numpy.typing import *
import matplotlib.pyplot as plt

from seghist.utils import ImageToolkits


def draw_fig(aspect_ratio_15, aspect_ratio_19, aspect_ratio, output='./vis/results/distribution.pdf'):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    axs[2].grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
    axs[2].hist(aspect_ratio_15, 
            np.arange(1, 41, 1),
            color=(197/255, 224/255, 180/255), 
            edgecolor='black',
            density=True,
            label='density',
            zorder=3)
    axs[2].set_yticks(np.linspace(0, 0.28, 8))
    axs[2].set_xlim(right = 40)
    axs[2].set_ylabel('SCUT-CTW1500', fontsize=16)
    axs[2].set_xlabel('Aspect ratio of text instances', fontsize=16)


    axs[1].grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
    axs[1].hist(aspect_ratio_19, 
            np.arange(1, 41, 1),
            color=(76/255,195/255,190/255), 
            edgecolor='black',
            density=True,
            label='density',
            zorder=3)
    axs[1].set_yticks(np.linspace(0, 0.28, 8))
    axs[1].set_xlim(right = 40)
    axs[1].set_ylabel('HDRC', fontsize=16)

    axs[0].grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
    axs[0].hist(aspect_ratio, 
            np.arange(1, 41, 1),
            color=(248/255,223/255,136/255), 
            edgecolor='black',
            density=True,
            label='density',
            zorder=3)
    axs[0].set_yticks(np.linspace(0, 0.28, 8))
    axs[0].set_xlim(right = 40)
    axs[0].set_ylabel('CHDAC', fontsize=16)

    for ax in axs:
        ytick_labels = ax.get_yticklabels()
        for label in ytick_labels:
            label.set_fontsize(14)
        xtick_labels = ax.get_xticklabels()
        for label in xtick_labels:
            label.set_fontsize(14)

    plt.tight_layout()
    plt.savefig(output)


def main():
    aspect_ratio_15 = []
    for file in tqdm(os.listdir('./data/scenetext/ctw1500/train/text_label_curve')):
        with open(os.path.join('./data/scenetext/ctw1500/train/text_label_curve', file)) as f:
            polygons = []
            for l in f.readlines():
                l = list(map(float, l.strip().split(',')[4:]))
                polygons.append(np.array(l).reshape(-1,2))
            hi = ImageToolkits(polygons, reorder=True)
            aspect_ratio_15.extend(hi.aspect_ratio())

    with open('./data/historical_document/ICDAR2019HDRC_Chinese/train_label.json') as f:
        datas = json.load(f)
    aspect_ratio_19 = []
    for data in tqdm(datas['data_list']):
        hi = ImageToolkits([np.array(d['polygon']).reshape(-1, 2) for d in data['instances']],
                            (data['height'], data['width']),
                            data['img_path'],
                            [d['text'] for d in data['instances']])
        aspect_ratio_19.extend(hi.aspect_ratio())
    with open('./data/historical_document/ICDAR2019HDRC_Chinese/val_label.json') as f:
        datas = json.load(f)
    for data in tqdm(datas['data_list']):
        hi = ImageToolkits([np.array(d['polygon']).reshape(-1, 2) for d in data['instances']],
                            (data['height'], data['width']),
                            data['img_path'],
                            [d['text'] for d in data['instances']])
        aspect_ratio_19.extend(hi.aspect_ratio())
    with open('./data/historical_document/ICDAR2019HDRC_Chinese/test_label.json') as f:
        datas = json.load(f)
    for data in tqdm(datas['data_list']):
        hi = ImageToolkits([np.array(d['polygon']).reshape(-1, 2) for d in data['instances']],
                            (data['height'], data['width']),
                            data['img_path'],
                            [d['text'] for d in data['instances']])
        aspect_ratio_19.extend(hi.aspect_ratio())

    data_root = './data/historical_document/IACC2022_CHDAC/official_dataset/final/train/'
    ann_file = 'ocr_train.json'
    with open(os.path.join(data_root, ann_file)) as f:
        datas = json.load(f)
    aspect_ratio = []
    for data in tqdm(datas['data_list']):
        hi = ImageToolkits([np.array(d['polygon']).reshape(-1, 2) for d in data['instances']],
                            (data['height'], data['width']),
                            data['img_path'],
                            [d['text'] for d in data['instances']])
        aspect_ratio.extend(hi.aspect_ratio())

    draw_fig(aspect_ratio_15, aspect_ratio_19, aspect_ratio)