import numpy as np
import matplotlib.pyplot as plt

from mmocr.utils.polygon_utils import *

from seghist.utils import stretch_kernel, iou, unstretch_kernel

def main():
    x = np.arange(0.3, 40, 0.1)
    c = np.arange(1, 10, 0.1)
    xs1, cs1 = np.meshgrid(x, c)
    ious1 = np.zeros_like(xs1)
    r = 0.16
    s = 1
    for i, x_ in enumerate(x):
        poly = np.array([[0,0],[0,x_],[1,x_],[1,0]])
        shrink = stretch_kernel(poly, r, s)
        for j, c_ in enumerate(c):
            expand1 = unstretch_kernel(shrink, r, s, refinement=False, unclip_ratio=c_)
            ious1[j, i] = iou(poly, expand1)

    x = np.arange(0.3, 40, 0.1)
    c = np.arange(1, 10, 0.1)
    xs2, cs2 = np.meshgrid(x, c)
    ious2 = np.zeros_like(xs2)
    r = 0
    s = 2
    for i, x_ in enumerate(x):
        poly = np.array([[0,0],[0,x_],[1,x_],[1,0]])
        shrink = stretch_kernel(poly, r, s)
        for j, c_ in enumerate(c):
            expand2 = unstretch_kernel(shrink, r, s, refinement=False, unclip_ratio=c_)
            ious2[j, i] = iou(poly, expand2)

    plt.rc('font', size=14)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cmap='viridis'
    vmin=0
    vmax=1
    axes[0].contour(cs1, xs1, ious1, levels=10, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].contourf(cs1, xs1, ious1, alpha=0.5, levels=10, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].contour(cs2, xs2, ious2, levels=10, cmap=cmap, vmin=vmin, vmax=vmax)
    im = axes[1].contourf(cs2, xs2, ious2, alpha=0.5, levels=10, cmap=cmap, vmin=vmin, vmax=vmax)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax, label='IoU')

    axes[0].set_xlabel('unclip ratio')
    axes[0].set_ylabel('vertical aspect ratio')
    axes[0].set_yticks([1, 5, 10, 15, 20, 25, 30, 35])
    axes[0].set_xticks(np.arange(1,10,1))
    axes[1].set_xlabel('unclip ratio')
    axes[1].set_ylabel('vertical aspect ratio')
    axes[1].set_yticks([1, 5, 10, 15, 20, 25, 30, 35])
    axes[1].set_xticks(np.arange(1,10,1))
    plt.savefig('./seghist/vis/results/tks.pdf')