from mmocr.registry import MODELS
from mmocr.models.textdet.module_losses import PSEModuleLoss

from seghist.model import PANTKSModuleLoss

@MODELS.register_module()
class PSETKSModuleLoss(PANTKSModuleLoss, PSEModuleLoss):
    """Almost same from PANTKS, except forward method.
    """
    def __init__(self, stretch_ratio: float = 2, **kwargs):
        PANTKSModuleLoss.__init__(self, stretch_ratio)
        PSEModuleLoss.__init__(self, **kwargs)

    def forward(self, *args, **kwargs):
        return PSEModuleLoss.forward(self, *args, **kwargs)