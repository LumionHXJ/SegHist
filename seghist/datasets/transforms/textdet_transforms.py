from typing import Dict, List, Optional, Tuple, Union
from mmocr.registry import TRANSFORMS
from mmocr.datasets.transforms import Resize, TextDetRandomCrop
import numpy as np
from mmcv.transforms.processing import Pad

@TRANSFORMS.register_module()
class MultiScaleResizeShorterSide(Resize):
    """Resize historical image by fixing longer side 
    and using multi-scale strategy to shorter side.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_polygons


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_polygons

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        fixed_longer_side(int): length of longer side (no matter 
            it's height or width)
        shorter_side_ratio(Tuple[float, float]): range of multi-scale ratio
            on resizing the shorter side, thus we don't keep the aspect ratio.
        clip_object_border (bool): Whether to clip the objects outside the
            border of the image. Defaults to True.    
    
    """
    def __init__(self, 
                 fixed_longer_side: int = 2000,
                 shorter_side_ratio: Tuple[float, float] = (0.8, 1.2),
                 clip_object_border: bool = True) -> None:
        super().__init__(scale_factor=1.,
                         keep_ratio=False, 
                         clip_object_border=clip_object_border)
        self.fixed_longer_side = fixed_longer_side
        self.shorter_side_ratio = shorter_side_ratio
        
    @staticmethod
    def _random_sample_ratio(ratio_range: Tuple[float, float]) -> float:
        """Private function to randomly sample ratio for shorter side
         from a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. 

        Args:
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.

        Returns:
            float: The targeted ratio of the shorter side to be resized.
        """

        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        return ratio
    
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        NOTE: Scale in mmcv is in (w, h)-style.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """
        h, w = results['img'].shape[:2]
        if h > w:
            scale_factor = self.fixed_longer_side / h
            scale_factor *= MultiScaleResizeShorterSide._random_sample_ratio(self.shorter_side_ratio)
            results['scale'] = (int(w * scale_factor), self.fixed_longer_side) # wh-style
        else:
            scale_factor = self.fixed_longer_side / w
            scale_factor *= MultiScaleResizeShorterSide._random_sample_ratio(self.shorter_side_ratio)
            results['scale'] = (self.fixed_longer_side, int(h * scale_factor))

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        self._resize_polygons(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(fixed_longer_side={self.fixed_longer_side}, '
        repr_str += f'shorter_side_ratio={self.shorter_side_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        return repr_str
    

@TRANSFORMS.register_module()
class RatioAwareCrop(TextDetRandomCrop):
    """Due to many vertical text lines in historical document,
    set different different crop ratio for height and width.

    Targets size will be computed dynamically.

    NOTE: crop ratio in hw-style but target_size should in wh-style

    Required Keys:

    - img
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Modified Keys:

    - img
    - img_shape
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Args:
        crop_ratio (Tuple[float, float] or float): ratio for height and width.
            i.e. crop_ratio is in hw-style
        positive_sample_ratio (float): The probability of sampling regions
            that go through text regions. Defaults to 5. / 8.    
    """
    def __init__(self,
                 crop_ratio: Tuple[float, float] or float = (0.7, 0.5), # h, w                 
                 positive_sample_ratio: float = 5.0 / 8.0) -> None:
        super().__init__(target_size=None, 
                         positive_sample_ratio=positive_sample_ratio)
        if isinstance(crop_ratio, float):
            self.crop_ratio = (crop_ratio, crop_ratio)
        else:
            self.crop_ratio = crop_ratio


    def transform(self, results: Dict) -> Dict:
        self.target_size = (int(results['img'].shape[0] * self.crop_ratio[0]),
                            int(results['img'].shape[1] * self.crop_ratio[1]))[::-1]
        return super().transform(results)
    

@TRANSFORMS.register_module()
class PadDivisor(Pad):
    def transform(self, results: dict) -> dict:
        results['valid_shape'] = results['img_shape']
        return super().transform(results)