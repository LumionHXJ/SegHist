from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.models.utils import multi_apply
from mmocr.models.textdet.heads import BaseTextDetHead, DBHead
from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample

from seghist.model.layer.layout_enhanced_block import LayoutEnhancedBlock

@MODELS.register_module()
class SegHistHead(BaseModule):
    def __init__(self, 
                 in_channels: int,
                 num_blocks: int,
                 shallow_channels: int,
                 embedding_channels: int,
                 output_channels: int = 1,
                 num_query: int = 6,
                 bridge_heads: int = 4,
                 former_heads: int = 8,
                 with_bias: bool = True,
                 with_sigmoid: bool = True,
                 use_dyrelu: bool = True,
                 dyrelu_mode: str = 'shared',
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                    dict(type='Kaiming', layer='Conv'),
                    dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
                ]):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(in_channels, shallow_channels, 3, 
                                padding=1, 
                                bias=with_bias, 
                                norm_cfg=dict(type='BN'))
        bottleneck_channels = shallow_channels // 4 # 128 / 4 = 32
        bottleneck_groups = bottleneck_channels // 4 # 32 / 4 = 8
        if num_blocks == 0:
            self.lem = None
        else:
            self.lem = nn.Sequential(*[LayoutEnhancedBlock(in_channels=shallow_channels,
                                                              bottleneck_channels=bottleneck_channels, 
                                                              bottleneck_group=bottleneck_groups,
                                                              embedding_channels=embedding_channels, 
                                                              bridge_heads=bridge_heads, 
                                                              former_heads=former_heads, 
                                                              use_dyrelu=use_dyrelu, 
                                                              dyrelu_mode=dyrelu_mode, 
                                                              with_bias=with_bias
                                                              ) for _ in range(num_blocks)])
            self.query = nn.Parameter(torch.randn(num_query, embedding_channels))
            
        self.with_sigmoid = with_sigmoid
        self.sigmoid = nn.Sigmoid()
                
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(shallow_channels, shallow_channels // 4, 2, 2),
            nn.BatchNorm2d(shallow_channels // 4), 
            nn.ReLU(),
            nn.ConvTranspose2d(shallow_channels // 4, output_channels, 2, 2)
        )
        
        self.num_query= num_query
        self.embedding_channels = embedding_channels
    
    def forward(self,
                img: Tensor,
                data_samples: Optional[List[TextDetDataSample]],
                mode: str = 'predict') -> Tuple[Tensor, Tensor, Tensor]:
        # N, H, W 
        prob_logits = self.forward_pass(img).squeeze(1) 
        prob_map = self.sigmoid(prob_logits)
        if mode == 'predict':
            return prob_map
        return prob_logits
    
    def forward_pass(self, x, mask=None):
        bs = x.size()[0]
        x = self.conv1(x)
        if self.lem is not None:
            x, _, _ = self.lem((x, 
                                self.query.expand(bs, self.num_query, self.embedding_channels), 
                                mask))        
        x = self.upconv(x)
        if self.with_sigmoid:
            x = self.sigmoid(x)
        return x # return prob map

@MODELS.register_module()
class DBSegHistHead(DBHead):
    def __init__(self, 
                 in_channels: int,
                 num_blocks: int,
                 shallow_channels: int,
                 output_channels: int = 1,
                 num_query: int = 8,
                 embedding_channels: int = 128,
                 bridge_heads: int = 4,
                 former_heads: int = 8,
                 use_dyrelu: bool = True,
                 dyrelu_mode: str = 'awared',
                 with_bias: bool = True,
                 with_m2f_mask: bool = True,
                 module_loss: Dict = None, 
                 postprocessor: Dict = None, 
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                    dict(type='Kaiming', layer='Conv'),
                    dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
                ]
    ) -> None:
        BaseTextDetHead.__init__(self, 
                                 module_loss=module_loss,
                                 postprocessor=postprocessor, 
                                 init_cfg=init_cfg)
        
        # binarization(logit in losses)
        self.binarize = SegHistHead(in_channels=in_channels, 
                                  num_blocks=num_blocks, 
                                  shallow_channels=shallow_channels,
                                  output_channels=output_channels,
                                  num_query=num_query, 
                                  embedding_channels=embedding_channels, 
                                  bridge_heads=bridge_heads, 
                                  former_heads=former_heads, 
                                  with_bias=with_bias, 
                                  use_dyrelu=use_dyrelu,
                                  dyrelu_mode=dyrelu_mode, 
                                  with_sigmoid=False,
                                  init_cfg=init_cfg)
        self.sigmoid = nn.Sigmoid()
        
        # threshold: no separation in threshold
        self.threshold = SegHistHead(in_channels=in_channels, 
                                  num_blocks=num_blocks, 
                                  shallow_channels=shallow_channels,
                                  output_channels=output_channels,
                                  num_query=num_query, 
                                  embedding_channels=embedding_channels, 
                                  bridge_heads=bridge_heads, 
                                  former_heads=former_heads, 
                                  with_bias=with_bias, 
                                  use_dyrelu=use_dyrelu,
                                  dyrelu_mode=dyrelu_mode, 
                                  with_sigmoid=True,
                                  init_cfg=init_cfg)
        
        self.with_m2f_mask = with_m2f_mask
    
    def generate_masks(self, data_samples: List[TextDetDataSample]):
        '''Generate mask for M2F(mobile2former), mask = 0 means masking a place.
        '''
        masks_h, masks_w = multi_apply(self._get_mask_single, data_samples)
        masks_h = torch.cat(masks_h, dim=0) # N, H
        masks_w = torch.cat(masks_w, dim=0) # N, W
        return torch.cat([masks_h, masks_w], dim=1) # N, H+W
        
    def _get_mask_single(self, data_sample: TextDetDataSample):
        mask_h = torch.ones(data_sample.batch_input_shape[0] // 4) 
        mask_w = torch.ones(data_sample.batch_input_shape[1] // 4) # H, W
        mask_h[data_sample.valid_shape[0] // 4:] = 0
        mask_w[data_sample.valid_shape[1] // 4:] = 0
        return mask_h.unsqueeze(0), mask_w.unsqueeze(0)

    def forward(self,
                img: Tensor,
                data_samples: Optional[List[TextDetDataSample]],
                mode: str = 'predict') -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            img (Tensor): Shape :math:`(N, C, H, W)`.
            data_samples (list[TextDetDataSample], optional): A list of data
                samples. Defaults to None.
            mode (str): Forward mode. It affects the return values. Options are
                "loss", "predict" and "both". Defaults to "predict".

                - ``loss``: Run the full network and return the prob
                  logits, threshold map and binary map.
                - ``predict``: Run the binarzation part and return the prob
                  map only.
                - ``both``: Run the full network and return prob logits,
                  threshold map, binary map and prob map.

        Returns:
            Tensor or tuple(Tensor): Its type depends on ``mode``, read its
            docstring for details. Each has the shape of
            :math:`(N, 4H, 4W)`.
        """
        if self.with_m2f_mask:
            masks = self.generate_masks(data_samples)
            masks = masks.to(img.device)
        else:
            masks = None
        
        # N, H, W 
        prob_logits = self.binarize.forward_pass(img, mask=masks).squeeze(1) 
        prob_map = self.sigmoid(prob_logits)
        if mode == 'predict':
            return prob_map
        thr_map = self.threshold.forward_pass(img, mask=masks).squeeze(1) 
        binary_map = self._diff_binarize(prob_map, thr_map, k=50).squeeze(1) 
        if mode == 'loss':
            return prob_logits, thr_map, binary_map
        return prob_logits, thr_map, binary_map, prob_map


@MODELS.register_module()
class PANSegHistHead(BaseTextDetHead):
    def __init__(self, 
                 in_channels: int,
                 num_blocks: int,
                 shallow_channels: int,
                 output_channels: int = 1,
                 num_query: int = 8,
                 embedding_channels: int = 128,
                 bridge_heads: int = 4,
                 former_heads: int = 8,
                 use_dyrelu: bool = True,
                 dyrelu_mode: str = 'shared',
                 with_bias: bool = True,
                 with_m2f_mask: bool = True,
                 module_loss: Dict = None, 
                 postprocessor: Dict = None, 
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                    dict(type='Kaiming', layer='Conv'),
                    dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
                ]
    ) -> None:
        super().__init__(module_loss=module_loss, 
                         postprocessor=postprocessor, 
                         init_cfg=init_cfg)
        
        # binarization(logit in losses)
        self.pred = SegHistHead(in_channels=in_channels, 
                                  num_blocks=num_blocks, 
                                  shallow_channels=shallow_channels,
                                  output_channels=output_channels,
                                  num_query=num_query, 
                                  embedding_channels=embedding_channels, 
                                  bridge_heads=bridge_heads, 
                                  former_heads=former_heads, 
                                  with_bias=with_bias, 
                                  use_dyrelu=use_dyrelu,
                                  dyrelu_mode=dyrelu_mode, 
                                  with_sigmoid=False,
                                  init_cfg=init_cfg)
        
        self.with_m2f_mask = with_m2f_mask
    
    def generate_masks(self, data_samples: List[TextDetDataSample]):
        '''Generate mask for M2F(mobile2former), mask = 0 means masking a place.
        '''
        masks_h, masks_w = multi_apply(self._get_mask_single, data_samples)
        masks_h = torch.cat(masks_h, dim=0) # N, H
        masks_w = torch.cat(masks_w, dim=0) # N, W
        return torch.cat([masks_h, masks_w], dim=1) # N, H+W
        
    def _get_mask_single(self, data_sample: TextDetDataSample):
        mask_h = torch.ones(data_sample.batch_input_shape[0] // 4) 
        mask_w = torch.ones(data_sample.batch_input_shape[1] // 4) # H, W
        mask_h[data_sample.valid_shape[0] // 4:] = 0
        mask_w[data_sample.valid_shape[1] // 4:] = 0
        return mask_h.unsqueeze(0), mask_w.unsqueeze(0)

    def forward(self,
                img: Tensor,
                data_samples: Optional[List[TextDetDataSample]]
                ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.with_m2f_mask:
            masks = self.generate_masks(data_samples)
            masks = masks.to(img.device)
        else:
            masks = None
        
        # N, H, W 
        outputs = self.pred.forward_pass(img, mask=masks)
        return outputs