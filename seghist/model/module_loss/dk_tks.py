from typing import Tuple
import copy

import cv2
import numpy as np
import torch

from mmocr.registry import MODELS
from mmocr.models.textdet.module_losses import DBModuleLoss
from mmocr.structures import TextDetDataSample

from seghist.utils import expand_poly, get_distance
from seghist.model import TKSModuleLoss

@MODELS.register_module()
class DBTKSModuleLoss(TKSModuleLoss, DBModuleLoss):
    def __init__(self, stretch_ratio: float = 2, **kwargs):
        TKSModuleLoss.__init__(self, stretch_ratio)
        DBModuleLoss.__init__(self, **kwargs)

    def _generate_thr_map(self, 
                          img_size: Tuple[int, int],
                          polygons) -> np.ndarray:
        """Generate threshold map.

        Args:
            img_size (tuple(int)): The image size (h, w)
            polygons (Sequence[ndarray]): 2-d array, representing all the
                polygons of the text region.

        Returns:
            tuple:

            - thr_map (ndarray): The generated threshold map.
            - thr_mask (ndarray): The effective mask of threshold map.
        """
        thr_map = np.zeros(img_size, dtype=np.float32)
        thr_mask = np.zeros(img_size, dtype=np.uint8)

        for polygon in polygons:
            self._draw_border_map(polygon, thr_map, 
                                  mask=thr_mask,
                                  shrink_ratio=self.shrink_ratio,
                                  stretch_ratio=self.stretch_ratio)
        thr_map = thr_map * (self.thr_max - self.thr_min) + self.thr_min

        return thr_map, thr_mask
    
    def _draw_border_map(self, 
                         polygon: np.ndarray, 
                         canvas: np.ndarray,
                         shrink_ratio: float,
                         stretch_ratio: float,
                         mask: np.ndarray) -> None:
        """Generate threshold map for one polygon.

        Args:
            polygon (np.ndarray): The polygon.
            canvas (np.ndarray): The generated threshold map.
            mask (np.ndarray): The generated threshold mask.
        """
        # 按照相同加权方法进行扩张（便于之后加权计算thr map）
        polygon = copy.deepcopy(polygon).reshape(-1, 2)
        distance = get_distance(polygon, shrink_ratio) 
        expanded_polygon = expand_poly(polygon, 
                                       shrink_ratio, 
                                       stretch_ratio)
        if len(expanded_polygon) == 0:
            print(f'Padding {polygon} gets {expanded_polygon}')
            expanded_polygon = polygon.copy().astype(np.int32)
        else:
            expanded_polygon = expanded_polygon.reshape(-1, 2).astype(np.int32)
        x_min = expanded_polygon[:, 0].min()
        x_max = expanded_polygon[:, 0].max()
        y_min = expanded_polygon[:, 1].min()
        y_max = expanded_polygon[:, 1].max()

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        polygon[:, 0] = (polygon[:, 0] - x_min) * stretch_ratio
        polygon[:, 1] = polygon[:, 1] - y_min

        # 构建坐标grid
        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width),
            (height, width)) * stretch_ratio # 横向坐标加权计算
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1),
            (height, width))

        # 原polygon的每条边对应一个map，最后取最小距离
        distance_map = np.zeros((polygon.shape[0], height, width),
                                dtype=np.float32)
        # 统计区域内每个点到每一条边的距离
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self._dist_points2line(xs, ys, polygon[i],
                                                       polygon[j])
            # 最后会用 1-distance_map 做thresh
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1) 
        distance_map = distance_map.min(axis=0) # 每个点的距离由最小距离决定

        x_min_valid = min(max(0, x_min), canvas.shape[1] - 1)
        x_max_valid = min(max(0, x_max), canvas.shape[1] - 1)
        y_min_valid = min(max(0, y_min), canvas.shape[0] - 1)
        y_max_valid = min(max(0, y_max), canvas.shape[0] - 1)

        if x_min_valid - x_min >= width or y_min_valid - y_min >= height:
            return

        # 位于扩张后多边形区域内的点会被考虑（thr有效）
        cv2.fillPoly(mask, [expanded_polygon.astype(np.int32)], 1)
        canvas[y_min_valid:y_max_valid + 1,
               x_min_valid:x_max_valid + 1] = np.fmax(
                   1 - distance_map[y_min_valid - y_min: y_max_valid - y_max +
                                    height, x_min_valid - x_min: x_max_valid -
                                    x_max + width],
                   canvas[y_min_valid:y_max_valid + 1,
                          x_min_valid:x_max_valid + 1])
    
    def _get_target_single(self, data_sample: TextDetDataSample) -> Tuple:
        """Generate loss target from a data sample.
        Modified to adapt to batch padding

        Args:
            data_sample (TextDetDataSample): The data sample.

        Returns:
            tuple: A tuple of four tensors as the targets of one prediction.
        """

        gt_shrink, gt_shrink_mask = TKSModuleLoss._get_target_single(self, data_sample)
        gt_instances = data_sample.gt_instances
        ignore_flags = gt_instances.ignored
        
        # thr mask is only effective around the text area, so there's no need to mask the padding.
        gt_thr, gt_thr_mask = self._generate_thr_map(
            data_sample.batch_input_shape, gt_instances[~ignore_flags].polygons)

        gt_thr = torch.from_numpy(gt_thr).unsqueeze(0).float()
        gt_thr_mask = torch.from_numpy(gt_thr_mask).unsqueeze(0).float()
        return gt_shrink, gt_shrink_mask, gt_thr, gt_thr_mask