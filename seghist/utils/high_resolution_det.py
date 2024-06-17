from __future__ import annotations

import math
import os
from typing import Tuple, Union, List, Optional
import warnings

import numpy as np
from numpy.typing import *
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon, box
from shapely.affinity import translate

from mmocr.apis import TextDetInferencer
from seghist.utils import iou

class DetectionResult:
    """Saving single text line detection result of split images.

    Args:
        polygon(Union[Polygon, NDArray]): part of text line.
        score(float): sum of confidence of text line.
        weight(int): number of merged text lines.
    """
    VERTICLE= 1
    HORIZENTAL = 0 # x: the zero dim
    def __init__(self, 
                 polygon: Union[Polygon, NDArray], 
                 score: float,
                 weight: int = 1):
        if isinstance(polygon, Polygon):
            self.polygon = polygon
        else:
            self.polygon = Polygon(np.array(polygon).reshape(-1, 2))
        self.score = score
        self.weight = weight

    def merge(self, other: DetectionResult):
        poly = self.polygon.union(other.polygon)
        assert isinstance(poly, Polygon), \
                'Merging result isn\'t Polygon, maybe a MultiPolygon, please check your code.'
        score = self.score + other.score
        weight = self.weight + other.weight
        return DetectionResult(poly, score, weight)
    
    def translate(self, offset, direction):
        if direction == self.HORIZENTAL:
            self.polygon = translate(self.polygon, xoff=offset)
        else:
            self.polygon = translate(self.polygon, yoff=offset)

    def max(self, direction):
        return np.max(np.array(self.polygon.exterior.coords)[:, direction])
    
    def min(self, direction):
        return np.min(np.array(self.polygon.exterior.coords)[:, direction])
    

class SplitImage:
    """Part of original image.

    Args:
        image(np.ndarray): part of image, in h-w-c style.
        corner(np.ndarray): top left corner of split images, xy-style.
    """
    SWITCH_INPUT = 2
    VERTICLE_MERGE = 1
    HORIZENTAL_MERGE = 0
    CANNOT_MERGE = -1
    CONTAIN = -2
    def __init__(self,
                 image: np.ndarray,
                 corner: Tuple):
        self.image = image
        self.height, self.width, self.channels = image.shape
        self.corner = corner

    def save_det_results(self, results: Union[dict, List[DetectionResult]]):
        """Handle detection results from mmocr inferencer and save raw results
        in object.
        Raw results represents that polygons' coordinates are offsets corresbonded 
        to the top left corner.
        The polygons are transformed to numpy.ndarray, with shape (k, 2). 

        Args:
            results: 
                - detection results mmocr inferencer, containing keys 'polygons' and 'scores'.
                - results generate by merging splited images.
        """
        if isinstance(results, dict):
            self.results = [DetectionResult(p, s) for p, s in zip(results['polygons'], results['scores'])]
        else:
            self.results = results

    @classmethod
    def mergable(cls, first: SplitImage, second: SplitImage):
        """Determine whether first can merge with second.
        We request first in the top/left direction of second, otherwise demand to switch them. 
        """
        if first.corner[1] == second.corner[1] and first.height == second.height:
            # STRICT: first is on the left side
            if first.corner[0] > second.corner[0]:
                return SplitImage.SWITCH_INPUT
            
            # no overlapping in horizental
            if first.corner[0] + first.width <= second.corner[0]:
                return SplitImage.CANNOT_MERGE
            # first contain second
            if first.corner[0] + first.width >= second.corner[0] + second.width:
                return SplitImage.CONTAIN            
            return SplitImage.HORIZENTAL_MERGE
        
        elif first.corner[0] == second.corner[0] and first.width == second.width:
            if first.corner[1] > second.corner[1]:
                return SplitImage.SWITCH_INPUT
            if first.corner[1] + first.height <= second.corner[1]:
                return SplitImage.CANNOT_MERGE
            if first.corner[1] + first.height >= second.corner[1] + second.height:
                return SplitImage.CONTAIN
            return SplitImage.VERTICLE_MERGE
        
        return SplitImage.CANNOT_MERGE
    
    @classmethod
    def intersect(cls, 
                  first_res: List[DetectionResult], 
                  second_res: List[DetectionResult],
                  overlap: box,
                  iou_thresh: float = 0.5):
        """Merging the results of first and second that touch the overlap area.
        Translated in SplitImage.merge_results
        """
        intersect_results = []

        # fetching polygons that touches overlap area.
        first_overlap = []
        second_overlap = []
        for fr in first_res:
            first_overlap.append(fr.polygon.intersection(overlap))
        for sr in second_res:
            second_overlap.append(sr.polygon.intersection(overlap))
        
        for fr, fo in zip(first_res, first_overlap):
            _iou = []
            for sr, so in zip(second_res, second_overlap):
                _iou.append(iou(fo, so))
            
            # if second_res is empty!
            if _iou == []:
                intersect_results.append(fr)
                continue
                
            argmax_iou = np.argmax(_iou)
            if _iou[argmax_iou] < iou_thresh:
                intersect_results.append(fr)
            else:
                # merge split results
                intersect_results.append(fr.merge(second_res[argmax_iou]))
                del second_res[argmax_iou]
                del second_overlap[argmax_iou]
        
        for sr in second_res:
            intersect_results.append(sr)

        return intersect_results

    @classmethod
    def merge_results(cls, 
                      first: SplitImage,
                      second: SplitImage,
                      merge_direction: int,
                      iou_thresh: float = 0.5):
        """Merging first and second in merge_direction.
        Results set of a,b are divided into a-b, b-a and a∩b.
        a∩b part is dealed with function SplitImage.intersect.
        """
        merge_results = [] # list of detection results

        if merge_direction == cls.HORIZENTAL_MERGE:
            # add offset
            offset = second.corner[0] - first.corner[0]
            second_res = second.results
            for p in second_res:
                p.translate(offset, merge_direction)

            dividing_first = second.corner[0] - first.corner[0]
            dividing_second = first.width

            intersect_first = []
            intersect_second = []  
            overlap = box(dividing_first, 0, dividing_second, first.height)          
            for fr in first.results:
                if fr.max(merge_direction) > dividing_first:
                    intersect_first.append(fr) # a∩b
                else:
                    merge_results.append(fr) # a-b
            for sr in second_res:
                if sr.min(merge_direction) < dividing_second:
                    intersect_second.append(sr) # a∩b
                else:
                    merge_results.append(sr) # b-a

            merge_results += cls.intersect(intersect_first, 
                                           intersect_second, 
                                           overlap,
                                           iou_thresh)

        else:
            # add offset
            offset = second.corner[1] - first.corner[1]
            second_res = second.results
            for p in second_res:
                p.translate(offset, merge_direction)

            dividing_first = second.corner[1] - first.corner[1]
            dividing_second = first.height

            intersect_first = []
            intersect_second = []            
            for fr in first.results:
                if fr.max(merge_direction) > dividing_first:
                    intersect_first.append(fr)
                else:
                    merge_results.append(fr)
            for sr in second_res:
                if sr.min(merge_direction) < dividing_second:
                    intersect_second.append(sr)
                else:
                    merge_results.append(sr)

            overlap = box(0, dividing_first, first.width, dividing_second)
            merge_results += cls.intersect(intersect_first, 
                                           intersect_second, 
                                           overlap,
                                           iou_thresh)
        return merge_results

    @classmethod
    def merge(cls, 
              first: SplitImage, 
              second: SplitImage,
              iou_thresh: float = 0.5):
        assert first.channels == second.channels, 'Images to merge must have same number of channels'
        assert hasattr(first, "results") and hasattr(second, "results"), \
            'Image to merge must already have detection results.'
        
        merge_flag = cls.mergable(first, second)
        assert merge_flag != cls.CANNOT_MERGE, 'First cannot merge with second, please check your code.'
        if merge_flag == cls.CONTAIN:
            warnings.warn("Object containing in the other, returning the bigger one.")
            return first
        
        if merge_flag == cls.SWITCH_INPUT:
            return cls.merge(second, first)
        
        # geometry info.
        if merge_flag == cls.HORIZENTAL_MERGE:
            corner = first.corner
            height = first.height
            width = max(first.width, second.width + second.corner[0] - first.corner[0])        
        else:
            corner = first.corner
            width = first.width
            height = max(first.height, second.height + second.corner[1] - first.corner[1])

        # combine images
        image = np.zeros((height, width, first.channels))
        image[:first.height, :first.width] = first.image
        image[-second.height:, -second.width:] = second.image
        merge_image = SplitImage(image, corner)

        # next merging results of first and second
        results = cls.merge_results(first, second, merge_flag, iou_thresh)

        merge_image.save_det_results(results)

        return merge_image
    
    def save_result_to_file(self, output_file):
        """Save detection results to output_file.
        Usually used after merging all sub-images.
        """
        polygons = [np.array(p.polygon.exterior.coords) for p in self.results]
        np.savez(output_file, *polygons)


def split_image(image: np.ndarray,
                split_size: Tuple[int, int]=None,
                overlap_size: Tuple[int, int]=None):
    """Split high resolution image to several low resolution image of split size.
    Adjacent splits shared overlap, define by overlap_size.

    Args:
        image(np.ndarray): high resolution image to split, in shape (H, W, C).
        split_size(tuple): (s_h, s_w) represents the size after split.
        overlap_size(tuple): (ol_h, oh_w) represents the overlapping of verticle and horizental respectively.

    Returns:
        results(two-dim array of SplitImage): image after spliting
    """
    if split_size is None:
        split_size = image.shape[:2]
        overlap_size = (0, 0)
    assert split_size[0] > overlap_size[0] and split_size[1] > overlap_size[1], \
        'split size must larger that overlap size.'

    img_h, img_w = image.shape[:2]
    n = math.ceil((img_h - overlap_size[0]) / (split_size[0] - overlap_size[0]))
    m = math.ceil((img_w - overlap_size[1]) / (split_size[1] - overlap_size[1]))
    results = np.zeros((n, m), dtype=object)

    for i in range(n):
        corner_y = min(i * (split_size[0] - overlap_size[0]),
                       img_h - split_size[0])
        for j in range(m):
            corner_x = min(j * (split_size[1] - overlap_size[1]),
                           img_w - split_size[1])
            results[i, j] = SplitImage(image[corner_y: corner_y + split_size[0], corner_x: corner_x + split_size[1]],
                                       (corner_x, corner_y))
            
    return results


def inference(split_images: np.ndarray[np.ndarray[SplitImage]],
              inferencer: TextDetInferencer):
    """Inferencing splited images in serial.
    """
    for _ in tqdm(split_images, desc='verticle'):
        for si in tqdm(_, desc='horizental'):
            det_results = inferencer(si.image)
            si.save_det_results(det_results)


def merging_results(split_images: np.ndarray[np.ndarray[SplitImage]],
                    iou_thresh: float = 0.5):
    """Merging splited results by horizental->verticle order."""
    n, m = split_images.shape
    row_images = np.zeros((n, ), dtype=object)
    for i in range(n):
        curr_split = split_images[i, 0]
        for j in range(1, m):
            curr_split = SplitImage.merge(curr_split, split_images[i, j], iou_thresh)
        row_images[i] = curr_split
    
    result = row_images[0]
    for i in range(1, n):
        result = SplitImage.merge(result, row_images[i], iou_thresh)

    return result


def main():
    img_root = './project_samples'
    # split_size = (1024, 2048) # ij-style here but xy-style in mmcv
    # overlap_size = (split_size[0] // 4, split_size[1] // 4)
    split_size = None
    overlap_size = None
    iou_thresh = 0.5
    inferencer = TextDetInferencer(ckpt='/home/huxingjian/model/mmocr/projects/PFRNet/work_dirs/det.pth', 
                                   config='/home/huxingjian/model/mmocr/projects/PFRNet/work_dirs/dbnetpp_kernel/dbnetpp_resnet50-dcnv2_fpnc_quad.py',
                                   device='cuda:0')
    image_list =  [f for f in os.listdir(img_root) if not os.path.isdir(os.path.join(img_root, f))]
    for img_path in image_list:
        img = cv2.imread(os.path.join(img_root, img_path))
        split_images = split_image(img, split_size, overlap_size)
        inference(split_images, inferencer)
        result = merging_results(split_images, iou_thresh)
        result.save_result_to_file(os.path.join(img_root, "det_results", img_path.replace('png', 'npz')))