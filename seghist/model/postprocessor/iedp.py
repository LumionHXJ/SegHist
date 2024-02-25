from typing import Optional

import cv2
import numpy as np
import torch
from torch import Tensor
from shapely.geometry import Polygon

from mmengine.structures import InstanceData
from mmocr.structures import TextDetDataSample
from mmocr.registry import MODELS
from mmocr.models.textdet.postprocessors import DBPostprocessor

from seghist.utils import unstretch_kernel

@MODELS.register_module()
class IterExpandPostprocessor(DBPostprocessor):
    """Implementation for Iterative Expansion Distance Post-Processor.

    Args:
        shrink_ratio: r<1
        stretch_ratio: s>=1
        min_text_area: min regional area in origin scale.
        refine: refine or unclip kernel only once.
        unclip_ratio: u>0, used when refine is false.
    """
    def __init__(self, 
                 shrink_ratio: float = 0.,
                 stretch_ratio: float = 2.0, 
                 min_text_area: int = 200, # area respect to original size
                 refine: bool = True,
                 unclip_ratio: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.stretch_ratio = stretch_ratio
        self.shrink_ratio = shrink_ratio
        self.min_text_area = min_text_area
        self.refine = refine
        if not refine:
            assert unclip_ratio > 0, 'must set unclip ratio u when not refine'
        self.unclip_ratio = unclip_ratio

    def get_text_instances(self, prob_map: Tensor,
                           data_sample: TextDetDataSample
                           ) -> TextDetDataSample:
        """Get text instance predictions of one image.

        Args:
            pred_result (Tensor): DBNet's output ``prob_map`` of shape
                :math:`(H, W)`.
            data_sample (TextDetDataSample): Datasample of an image.

        Returns:
            TextDetDataSample: A new DataSample with predictions filled in.
            Polygons and results are saved in
            ``TextDetDataSample.pred_instances.polygons``. The confidence
            scores are saved in ``TextDetDataSample.pred_instances.scores``.
        """
        prob_map = prob_map[..., :data_sample.valid_shape[0], :data_sample.valid_shape[1]]

        data_sample.pred_instances = InstanceData()
        data_sample.pred_instances.polygons = []
        data_sample.pred_instances.scores = []

        text_mask = prob_map > self.mask_thr  

        score_map = prob_map.data.cpu().numpy().astype(np.float32)
        text_mask = text_mask.data.cpu().numpy() * 255
        text_mask = text_mask.astype(np.uint8)  # to numpy
        
        contours, _ = cv2.findContours(text_mask,
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        for i, poly in enumerate(contours):
            if i > self.max_candidates:
                break
            epsilon = self.epsilon_ratio * cv2.arcLength(poly, True)
            approx = cv2.approxPolyDP(poly, epsilon, True)
            poly_pts = approx.reshape(-1, 2)
            if poly_pts.shape[0] < 4:
                continue
            score = self._get_bbox_score(score_map, poly_pts)
            if score < self.min_text_score:
                continue
            
            # trying recover kernel in iterative mode
            try:
                poly = unstretch_kernel(poly_pts, 
                                        self.shrink_ratio, 
                                        self.stretch_ratio,
                                        refinement=self.refine,
                                        unclip_ratio=self.unclip_ratio) 
            except Exception as e:
                print(f'Error {e} find when unstretching kernel {poly_pts}.')
            
            # If the result polygon does not exist, or it is split into
            # multiple polygons, skip it.
            if len(poly) == 0:
                continue
            poly = poly.reshape(-1, 2)

            if self.text_repr_type == 'quad':
                rect = cv2.minAreaRect(poly.astype(np.int32))
                vertices = cv2.boxPoints(rect)
                poly = vertices.flatten() if min(
                    rect[1]) >= self.min_text_width else []
            elif self.text_repr_type == 'poly':
                scale = data_sample.scale_factor[0] * data_sample.scale_factor[1]
                poly = poly.flatten() if Polygon(
                    poly).area / scale > self.min_text_area else []

            if len(poly) < 8:
                poly = np.array([], dtype=np.float32)

            if len(poly) > 0:
                data_sample.pred_instances.polygons.append(poly)
                data_sample.pred_instances.scores.append(score)

        data_sample.pred_instances.scores = torch.FloatTensor(
            data_sample.pred_instances.scores)

        return data_sample