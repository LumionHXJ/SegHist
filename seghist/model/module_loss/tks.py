<<<<<<< HEAD
from typing import Sequence, Tuple, Optional, Dict, Union
=======
from typing import Sequence, Tuple, Optional
>>>>>>> origin/main

import cv2
import numpy as np
import torch
<<<<<<< HEAD
from torch import Tensor
=======
>>>>>>> origin/main

from mmocr.registry import MODELS
from mmocr.models.textdet.module_losses import SegBasedModuleLoss
from mmocr.structures import TextDetDataSample

from seghist.utils import stretch_kernel

<<<<<<< HEAD
=======

@MODELS.register_module()
>>>>>>> origin/main
class TKSModuleLoss(SegBasedModuleLoss):
    """Computing module loss using the Text Kernel Stretching method. 
    Generating targets for a segmentation-based model that only predicts 
    text kernel. Also serves as a subclass for the SegHist implementation 
    of a specific segmentation-based model.

    Args:
        stretch_ratio: Horizontal stretching ratio (s>1).
    """
    def __init__(self, stretch_ratio: float = 2, **kwargs):
        super().__init__(**kwargs)
        self.stretch_ratio = stretch_ratio

    def _generate_kernels(
        self,
        img_size: Tuple[int, int],
        text_polys: Sequence[np.ndarray],
        shrink_ratio: float,
        stretch_ratio: float,
        ignore_flags: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate text instance kernels according to a shrink ratio.

        Args:
            img_size (tuple(int, int)): The image size of (height, width).
            text_polys (Sequence[np.ndarray]): 2D array of text polygons.
            shrink_ratio (float or int): The shrink ratio of kernel.
            stretch_ratio (float or int): The stretch ratio of kernel.
            ignore_flags (torch.BoolTensor, optional): Indicate whether the
                corresponding text polygon is ignored. Defaults to None.

        Returns:
            tuple(ndarray, ndarray): The text instance kernels of shape
                (height, width) and updated ignorance flags.
        """
        assert isinstance(img_size, tuple)
        assert isinstance(shrink_ratio, (float, int))

        if ignore_flags is None:
            ignore_flags = [False for _ in text_polys]

        text_kernel = np.zeros(img_size, dtype=np.float32)

        for text_ind, poly in enumerate(text_polys):
            if ignore_flags[text_ind]:
                continue
            
            shrunk_poly = stretch_kernel(poly, shrink_ratio, stretch_ratio)
            
            # Split while shrinkage, resulted in empty list.
            if len(shrunk_poly) == 0:
                ignore_flags[text_ind] = True
                continue    
            
            cv2.fillPoly(text_kernel,
                         [shrunk_poly.astype(np.int32)], 
                         1)

        return text_kernel, ignore_flags
    
    def _get_target_single(self, data_sample: TextDetDataSample) -> Tuple:
        """Generate loss target from a data sample.
        Modified to adapt to batch padding

        Args:
            data_sample (TextDetDataSample): The data sample.

        Returns:
            tuple: A tuple of four tensors as the targets of one prediction.
        """

        gt_instances = data_sample.gt_instances
        ignore_flags = gt_instances.ignored
        for idx, polygon in enumerate(gt_instances.polygons):
            if self._is_poly_invalid(polygon.astype(np.float32)):
                ignore_flags[idx] = True
                
        gt_shrink, ignore_flags = self._generate_kernels(
            data_sample.batch_input_shape, # adapt to batch input shape
            gt_instances.polygons,
            self.shrink_ratio,
            self.stretch_ratio,
            ignore_flags=ignore_flags)
        
        # Get boolean mask where Trues indicate text instance pixels
        gt_shrink = gt_shrink > 0

        gt_shrink_mask = self._generate_effective_mask(
            data_sample.batch_input_shape, gt_instances[ignore_flags].polygons)
        
        # mask padding area
        gt_shrink_mask[data_sample.valid_shape[0]:data_sample.batch_input_shape[0],
                       data_sample.valid_shape[1]:data_sample.batch_input_shape[1]] = 0 

        # to_tensor
        gt_shrink = torch.from_numpy(gt_shrink).unsqueeze(0).float()
        gt_shrink_mask = torch.from_numpy(gt_shrink_mask).unsqueeze(0).float()
<<<<<<< HEAD
        return gt_shrink, gt_shrink_mask
    

@MODELS.register_module()
class SegHistModuleLoss(TKSModuleLoss):
    def __init__(self,
                 loss_prob: Dict = dict(
                     type='MaskedBalancedBCEWithLogitsLoss'),
                 weight_prob: float = 5.,
                 min_sidelength: Union[int, float] = 8) -> None:
        super().__init__()
        self.loss_prob = MODELS.build(loss_prob)
        self.weight_prob = weight_prob
        self.min_sidelength = min_sidelength

    def forward(self, preds: Tuple[Tensor],
                data_samples: Sequence[TextDetDataSample]) -> Dict:

        prob_logits = preds 
        gt_shrinks, gt_shrink_masks = self.get_targets(data_samples)
        gt_shrinks = gt_shrinks.to(prob_logits.device)
        gt_shrink_masks = gt_shrink_masks.to(prob_logits.device)

        loss_prob = self.loss_prob(prob_logits, gt_shrinks, gt_shrink_masks)

        results = dict(loss_prob=self.weight_prob * loss_prob)
        
        return results
=======
        return gt_shrink, gt_shrink_mask
>>>>>>> origin/main
