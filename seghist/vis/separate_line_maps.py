import json
import os

import numpy as np
from numpy.typing import *
import cv2

from seghist.utils import ImageToolkits

def main(data_root = './data/historical_document/IACC2022_CHDAC/official_dataset/preliminary/train/',
         ann_file = 'label_train.json'):
    data_root = './data/historical_document/IACC2022_CHDAC/official_dataset/preliminary/train/'
    ann_file = 'label_train.json'
    with open(os.path.join(data_root, ann_file)) as f:
        datas = json.load(f)
    for img_path, instances in datas.items():
        if img_path == 'image_701.jpg':
            break
    image = cv2.imread(os.path.join(data_root, 'image', 'image_701.jpg'))
    hi = ImageToolkits([np.array(d['points']).reshape(-1, 2) for d in instances],
                       image.shape[:2],
                       'image_701.jpg',
                       texts=[d['transcription'] for d in instances])
    hi.process()
    kernel_single, kernel_double = hi.generate_kernelmap()

    kernel_single = cv2.applyColorMap(kernel_single[..., np.newaxis], cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(image, 0.6, kernel_single, 0.4, 0)
    cv2.imwrite('./vis/results/out_single.png', overlay)

    kernel_double = cv2.applyColorMap(kernel_double[..., np.newaxis], cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(image, 0.6, kernel_double, 0.4, 0)
    cv2.imwrite('./vis/results/out_double.png', overlay)