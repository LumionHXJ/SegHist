# Copyright (c) OpenMMLab. All rights reserved.
from typing import  Tuple

import numpy as np
import torch

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.models.textdet.module_losses import PANModuleLoss

from seghist.model import TKSModuleLoss

@MODELS.register_module()
class PANTKSModuleLoss(TKSModuleLoss, PANModuleLoss):
    """PAN generates multiple targets using series of ratios.
    Rewrite function _get_target_single based on TKS.
    """
    def __init__(self, stretch_ratio: float = 2, **kwargs):
        TKSModuleLoss.__init__(self, stretch_ratio)
        PANModuleLoss.__init__(self, **kwargs)

    def _get_target_single(self, data_sample: TextDetDataSample
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate loss target from a data sample.

        Args:
            data_sample (TextDetDataSample): The data sample.

        Returns:
            tuple: A tuple of four tensors as the targets of one prediction.
        """
        gt_polygons = data_sample.gt_instances.polygons
        gt_ignored = data_sample.gt_instances.ignored

        gt_kernels = []
        for ratio in self.shrink_ratio:
            gt_kernel, gt_ignored = self._generate_kernels(
                data_sample.batch_input_shape,
                gt_polygons,
                ratio,
                self.stretch_ratio,
                ignore_flags=gt_ignored)
            gt_kernels.append(gt_kernel)
        gt_polygons_ignored = data_sample.gt_instances[gt_ignored].polygons
        gt_mask = self._generate_effective_mask(data_sample.batch_input_shape,
                                                gt_polygons_ignored)
        gt_mask[data_sample.valid_shape[0]:data_sample.batch_input_shape[0],
                data_sample.valid_shape[1]:data_sample.batch_input_shape[1]] = 0 

        gt_kernels = np.stack(gt_kernels, axis=0) #K, H, W
        gt_kernels = torch.from_numpy(gt_kernels).float()
        gt_mask = torch.from_numpy(gt_mask).float()
        return gt_kernels, gt_mask