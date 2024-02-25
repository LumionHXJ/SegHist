from typing import Tuple, List, Sequence, Union
import json
import os
import warnings
from tqdm import tqdm

import numpy as np
from numpy.typing import *
import cv2
from shapely import Polygon
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

from mmocr.utils.polygon_utils import poly_make_valid

from seghist.utils import iou, unstretch_kernel, stretch_kernel, ImageToolkits

def draw_fig(iou_db, iou_db2, iou_iedp, output='./vis/results/iedp_recovery.pdf'):
    plt.rc('font', size=14)
    iou_db = np.array(iou_db)
    iou_db2 = np.array(iou_db2)
    iou_iedp = np.array(iou_iedp)

    xgrids = np.linspace(0, 40, 60)
    ygrids = np.linspace(0.8, 1, 60)

    hist0, xedge0, yedge0 = np.histogram2d(iou_db[:, 0], iou_db[:, 1], bins=(xgrids, ygrids))
    hist1, xedge1, yedge1 = np.histogram2d(iou_db2[:, 0], iou_db2[:, 1], bins=(xgrids, ygrids))
    hist2, xedge2, yedge2 = np.histogram2d(iou_iedp[:, 0], iou_iedp[:, 1], bins=(xgrids, ygrids))

    norm = LogNorm(vmin=1e-1, vmax=1e4, clip=True)

    X0, Y0 = np.meshgrid(xedge0[:-1], yedge0[:-1], indexing='ij')
    X1, Y1 = np.meshgrid(xedge1[:-1], yedge1[:-1], indexing='ij')
    X2, Y2 = np.meshgrid(xedge2[:-1], yedge2[:-1], indexing='ij')
    levels = np.logspace(-1, 4, 8)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].contourf(X0, Y0, hist0+0.1, cmap='Blues', norm=norm, levels=levels)
    axes[1].contourf(X1, Y1, hist1+0.1, cmap='Blues', norm=norm, levels=levels)
    im = axes[2].contourf(X2, Y2, hist2+0.1, cmap='Blues', norm=norm, levels=levels)

    #plt.scatter(iou_iedp[:, 0], iou_iedp[:, 1], 3, color='b', alpha=0.3)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label='density', ticks=[1, 10, 100, 1000])

    for ax in axes:
        ax.set_xlabel('vertical aspect ratio')
        ax.set_ylabel('IoU')
        ax.set_xticks(np.arange(0, 40, 10))
        ax.set_yticks(np.arange(0.8, 1, 0.05))

    plt.savefig(output, bbox_inches='tight')
    

def main():
    with open("./data/historical_document/IACC2022_CHDAC/official_dataset/final/test/ocr_test.json") as f:
        datas = json.load(f)
    iou_db = []
    iou_db2 = []
    iou_iedp = []

    r = 0
    s = 2
    c = [1.5, 2]

    for data in tqdm(datas['data_list']):
        gt_polys = []
        for instance in data['instances']:
            gt_polys.append(np.array(instance['polygon']).reshape(-1,2))

        hi = ImageToolkits(gt_polys)
        aspect_ratio = hi.vertical_aspect_ratio()
        
        for p, a in zip(gt_polys, aspect_ratio):
            shrink = stretch_kernel(p, r, s)
            if len(shrink) == 0:
                continue
            expand_db = unstretch_kernel(shrink, r, s, refinement=False, unclip_ratio=c[0])
            expand_db2 = unstretch_kernel(shrink, r, s, refinement=False, unclip_ratio=c[1])
            expand_iedp = unstretch_kernel(shrink, r, s, tolerance=0.01)
            db = iou(p, expand_db)
            db2 = iou(p, expand_db2)
            iedp = iou(p, expand_iedp)
            iou_db.append([a, db])
            iou_db2.append([a, db2])
            iou_iedp.append([a, iedp])
    
    draw_fig(iou_db, iou_db2, iou_iedp)