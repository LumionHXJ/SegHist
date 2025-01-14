import json
import os
from tqdm import tqdm

import numpy as np
from numpy.typing import *
import matplotlib.pyplot as plt

from mmocr.apis import TextDetInferencer

from seghist.utils import ImageToolkits, iou

def draw_fig():
    with open("./seghist/vis/results/db_as.txt") as f:
        db = []
        num = []
        for l in f.readlines():
            ratio, num_, db_ = tuple(map(float, l.strip().split(',')))
            db.append(db_)
            num.append(num_)

    with open("./seghist/vis/results/tks_as.txt") as f:
        tks = []
        for l in f.readlines():
            ratio, num_, tks_ = tuple(map(float, l.strip().split(',')))
            tks.append(tks_)

    with open("./seghist/vis/results/iedp_as.txt") as f:
        iedp = []
        for l in f.readlines():
            ratio, num_, iedp_ = tuple(map(float, l.strip().split(',')))
            iedp.append(iedp_)

    line = [db, tks, iedp]
    colors = ['purple', 'royalblue', 'orangered']
    labels = ['DB', 'DB+TKS', 'DB+TKS+IEDP']
    linestyle = ['dotted', 'dashed', 'solid']
    markers = ['o', 's', '^']
    marker_sizes = [8, 8, 8]

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()  # Get current axes
    ax2 = ax1.twinx()  # Create another axis that shares the same x-axis

    # Plotting lines and points
    for l, c, label, ls, m, ms in zip(line, colors, labels, linestyle, markers, marker_sizes):
        ax1.plot(range(len(l)), l, color=c, label=label, zorder=2, linestyle=ls, marker=m, markersize=ms, linewidth=2.5)  # Ensuring plots are on top

    for i, (db_, tks_, iedp_) in enumerate(zip(db, tks, iedp)):
        y = max(db_, tks_, iedp_)
        color = np.argmax([db_, tks_, iedp_])
        ax1.text(i, y+0.02, str(round(y*100, 2))+"%", fontsize=14, ha='center', va='bottom', color=colors[color])

    # Setting labels for the left y-axis
    ax1.set_ylabel('mean IoU', fontsize=14)
    ax1.legend(loc='lower right')

    # Plotting bar chart on the second y-axis and setting it to the bottom
    ax2.bar(range(len(num)), num, alpha=0.3, color='grey', zorder=1, width=0.5,
            tick_label=['x<1', '1<x<5', '5<x<10', '10<x<15', '15<x<20', '20<x'])  # Lower zorder places bars at the bottom
    # Setting labels for the right y-axis
    ax2.set_ylabel('frequency', fontsize=14)
    ax1.set_xlabel('vertical aspect ratio')

    # Adjusting the y-axis limits if necessary to ensure visibility of all elements
    ax1.set_ylim(bottom=min(min(db), min(tks), min(iedp)) - 0.03, top=0.95)
    ax2.set_ylim(0, max(num) * 1.2)
    ax1.grid()
    plt.savefig('./seghist/vis/results/ar_res1.pdf')


def main(output):
    inferencer = TextDetInferencer(ckpt='./work_dirs_chdac/dbnetpp_kernel/epoch_55_9643.pth', 
                                   config='./work_dirs_chdac/dbnetpp_kernel/seghist_resnet50-dcnv2_fpnc.py',
                                   device='cuda:0')
    img_root = './data/historical_document/IACC2022_CHDAC/official_dataset/final/test'
    with open("./data/historical_document/IACC2022_CHDAC/official_dataset/final/test/ocr_test.json") as f:
        datas = json.load(f)

    ratio_list = [-1, 1, 5, 10, 15, 20, np.infty] # need to revised ticks in draw_figs
    mean_iou = [[] for r in ratio_list[:-1]]
    curl_mean_iou = [[] for r in ratio_list[:-1]]
    straight_mean_iou = [[] for r in ratio_list[:-1]]

    with open("./res.txt", mode='w') as f:
        for data in tqdm(datas['data_list']):
            ret_polys = inferencer(os.path.join(img_root, 'image', data['img_path']))['polygons']
            ret_polys = [np.array(p).reshape(-1, 2) for p in ret_polys]
            gt_polys = []
            for instance in data['instances']:
                gt_polys.append(np.array(instance['polygon']).reshape(-1,2))

            hi = ImageToolkits(gt_polys)
            aspect_ratio = hi.vertical_aspect_ratio()
            
            for i in range(len(ratio_list[:-1])):
                gts = [poly for poly, ratio in zip(gt_polys, aspect_ratio) if ratio_list[i] < ratio < ratio_list[i+1]]            
                match_gt = np.zeros((len(gts, )))
                match_gt_curl = []
                match_gt_straight = []

                for igt, gt in enumerate(gts):
                    for ip, p in enumerate(ret_polys):
                        match_gt[igt] = max(match_gt[igt], iou(gt, p))
                    if len(gt) > 4:
                        match_gt_curl.append(match_gt[igt])
                    else:
                        match_gt_straight.append(match_gt[igt])
            
                mean_iou[i].extend(match_gt.tolist())
                curl_mean_iou[i].extend(match_gt_curl)
                straight_mean_iou[i].extend(match_gt_straight)

    with open(output, mode='w') as f:
        for i in range(len(ratio_list[:-1])):
            f.write(f"{ratio_list[i]}, {len(mean_iou[i])}, {sum(mean_iou[i])/(len(mean_iou[i])+1e-4)}\n")