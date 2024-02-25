from mmocr.registry import TRANSFORMS
import numpy as np
import cv2
from mmdet.datasets.transforms import ColorTransform

@TRANSFORMS.register_module()
class ChannelShuffle(ColorTransform):
    def _transform_img(self, results: dict, mag: float) -> None:
        """Invert the image."""
        img = results['img']
        channels = img.shape[-1]
        shuffle_result = np.arange(0, channels)
        np.random.shuffle(shuffle_result)
        results['img'] = results['img'][..., shuffle_result]

@TRANSFORMS.register_module()
class GaussianBlur(ColorTransform):
    def __init__(self, 
                 blur_limit = (3, 7), 
                 sigma = 0,
                 **kwargs):
        self.blur_limit = blur_limit
        self.sigma = sigma
        super().__init__(**kwargs)

    def _transform_img(self, results: dict, mag: float) -> None:
        kernel_size = np.random.choice(np.arange(self.blur_limit[0],
                                                 self.blur_limit[1] + 2,
                                                 2))
        results['img'] = cv2.GaussianBlur(results['img'], (kernel_size, kernel_size), self.sigma)
