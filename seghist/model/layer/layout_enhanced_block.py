from typing import Union, List, Tuple

import torch
from torch import Tensor, nn

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmocr.models.common.layers import TFEncoderLayer
from mmocr.models.common.modules import ScaledDotProductAttention

from seghist.model.layer.dyrelu import DyReLU


class Local(nn.Module):
    def __init__(self,
                 in_channels,
                 embedding_channels,
                 bottleneck_channels,
                 bottleneck_group,
                 n_heads,
                 use_dyrelu=True,
                 dropout=0.1,
                 dyrelu_mode='awared',
                 with_bias=True):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.n_heads = n_heads

        self.pointwise_conv = nn.Conv2d(in_channels, 
                                        bottleneck_channels,
                                        kernel_size=1)
        self.group_conv = nn.Conv2d(bottleneck_channels, 
                                    bottleneck_channels,
                                    kernel_size=7, 
                                    padding=3,
                                    groups=bottleneck_group)
        
        self.pointwise_norm = nn.BatchNorm2d(bottleneck_channels)
        self.group_norm = nn.BatchNorm2d(bottleneck_channels)

        self.linear_k = nn.Linear(embedding_channels, bottleneck_channels, bias=with_bias)
        self.linear_v = nn.Linear(embedding_channels, bottleneck_channels, bias=with_bias)
        self.pre_attn = ScaledDotProductAttention((self.bottleneck_channels / n_heads)**0.5, dropout)
        
        self.use_dyrelu = use_dyrelu
        if use_dyrelu:
            self.act1 = DyReLU(bottleneck_channels,
                              embedding_channels,
                              mode=dyrelu_mode)
            self.act2 = DyReLU(bottleneck_channels,
                              embedding_channels,
                              mode=dyrelu_mode)
        else:
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()

    def forward(self, x, z, mask=None):
        """x: N, C, H, W
        z: N, M, d
        """
        x = self.pointwise_conv(x)
        x = self.pointwise_norm(x)

        # compute attention map for multiple uses!
        bs, num_queries, _ = z.size()
        z_k = self.linear_k(z).view(bs, num_queries, 
                                    self.n_heads, 
                                    self.bottleneck_channels // self.n_heads).transpose(1, 2).contiguous() 
        z_v = self.linear_v(z).view(bs, num_queries, 
                                    self.n_heads, 
                                    self.bottleneck_channels // self.n_heads).transpose(1, 2).contiguous()
        x_q = x.view(bs, self.n_heads, 
                     self.bottleneck_channels//self.n_heads, -1).transpose(2, 3).contiguous() # N, h, HW, C_b/h
        attn_out, attn_map = self.pre_attn(x_q, z_k, z_v, mask)

        if self.use_dyrelu:
            x = self.act1(x, z, attn_map)
        else:
            x = self.act1(x)

        x = self.group_conv(x)
        x = self.group_norm(x)
        if self.use_dyrelu:
            x = self.act2(x, z, attn_map)
        else:
            x = self.act2(x)

        return x, attn_out # N, h, HW, C//h
        

class Local2Layout(nn.Module):
    def __init__(self,
                 n_heads,
                 in_channels,
                 embedding_channels,
                 dropout=0.1,
                 with_bias=True) -> None:
        super().__init__()
        assert in_channels % n_heads == 0, 'n_heads must divide in_channels'
        assert in_channels == embedding_channels, \
            'input channels should be same as embed channels for simplicity'
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.embedding_channels = embedding_channels

        self.norm1 = nn.LayerNorm(embedding_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        self.linear_q = nn.Linear(self.embedding_channels, self.in_channels, bias=with_bias)
        self.ffn = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 2, bias=with_bias),
            nn.GELU(),
            nn.Linear(self.in_channels // 2, self.embedding_channels, bias=with_bias),
            nn.Dropout(dropout)
        )

        self.attention = ScaledDotProductAttention((self.in_channels / n_heads)**0.5, dropout)

    def forward(self, x: Tensor, z: Tensor, mask=None):
        '''
        x: N, H+W, C
        z: N, M, d
        M: N, H+W
        '''
        bs, length, _ = x.shape
        num_queries = z.shape[1]
        residue = z

        # part 1: pre norm
        z = self.norm1(z)

        # part 2: linear z & shape to bs, heads, H/W, C/heads
        z: Tensor = self.linear_q(z) # N, M, C
        z = z.view(bs, num_queries, self.n_heads, 
                   self.in_channels // self.n_heads).transpose(1, 2).contiguous()
        x = x.view(bs, length, self.n_heads, 
                   self.in_channels // self.n_heads).transpose(1, 2).contiguous()

        # part 3: attend mask(N, 1(h), 1(M), H+W)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        # part 4: attention
        attn_out, _ = self.attention(z, x, x, mask) # N, h, M, C/h
        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, num_queries, -1) # N, M, C
        residue = residue + attn_out # N, M, C

        # part 5: projection(output = MHA's output)
        z = self.norm2(residue)
        z = self.ffn(z) # N, M, d

        # part 6: residue link
        z = z + residue

        return z


class Layout2Local(nn.Module):
    def __init__(self,
                 n_heads,
                 in_channels,
                 embedding_channels,
                 dropout=0.1,
                 with_bias=True) -> None:
        super().__init__()
        assert in_channels % n_heads == 0, 'n_heads must divide in_channels'
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.embedding_channels = embedding_channels

        self.norm2 = nn.LayerNorm(in_channels)

        self.ffn = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 2, bias=with_bias),
            nn.GELU(),
            nn.Linear(self.in_channels // 2, self.in_channels, bias=with_bias),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor, attn_f2m: Tensor):
        '''
        x: N, HW, C
        attn_f2m: N, h, HW, C//h
        mask: N, H, W
        '''
        bs, length, _ = x.shape

        # part 1: add precomputed attention
        attn_out = attn_f2m.transpose(1, 2).contiguous().view(bs, length, -1) # N, HW, C
        residue = x + attn_out

        # part 2: norm+ffn
        x = self.norm2(residue)
        x = self.ffn(x)

        # part 3: residue link, return N, HW, C
        x = x + residue
        return x


class LayoutEnhancedBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 bottleneck_channels,
                 bottleneck_group,
                 embedding_channels=256,
                 bridge_heads=4,
                 former_heads=8,
                 use_dyrelu=True,
                 dyrelu_mode='awared',
                 with_bias=True,
                 init_cfg: Union[dict, List[dict], None] = [
                    dict(type='Kaiming', layer='Conv'),
                    dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
                ]):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.embedding_channels = embedding_channels
        self.bridge_heads = bridge_heads
        self.former_heads = former_heads
        self.bottleneck_group = bottleneck_group
        
        self.Local = Local(in_channels=in_channels, 
                             embedding_channels=embedding_channels,
                             bottleneck_channels=bottleneck_channels, 
                             bottleneck_group=bottleneck_group,
                             use_dyrelu=use_dyrelu, 
                             n_heads=bridge_heads,
                             dyrelu_mode=dyrelu_mode,
                             with_bias=with_bias)
        self.dyrelu_mode = dyrelu_mode if use_dyrelu else 'none'

        self.out_conv = ConvModule(bottleneck_channels, in_channels, 
                                   kernel_size=1, 
                                   bias=with_bias,
                                   norm_cfg=dict(type='BN'), 
                                   act_cfg=dict(type='ReLU'))
        self.pooling = nn.AdaptiveMaxPool1d(1)
        
        self.Local2Layout = Local2Layout(n_heads=bridge_heads, 
                                           in_channels=in_channels, 
                                           embedding_channels=embedding_channels, 
                                           with_bias=with_bias)
        self.Layout2Local = Layout2Local(n_heads=bridge_heads, 
                                           in_channels=bottleneck_channels, 
                                           embedding_channels=embedding_channels, 
                                           with_bias=with_bias)
        self.Layout = TFEncoderLayer(d_model=embedding_channels, 
                                     d_inner=embedding_channels // 2,
                                     d_k=embedding_channels // former_heads,
                                     d_v=embedding_channels // former_heads,
                                     qkv_bias=with_bias,
                                     n_head=former_heads) # using GELU in FFN

    def forward(self, input: Tuple):
        '''
        x: N, C, H, W
        z: N, M, d
        masks: N, H, W
        '''
        x, z, mask = input # now mask is N, H+W
        bs, _, h, w = x.size()   

        # part 2: m2f(need to prepare mask)
        global_h = self.pooling(x.view(bs, -1, w)).view(bs, -1, h)
        global_h = global_h.transpose(1,2).contiguous() # N, H, C
        global_w = self.pooling(x.transpose(2,3).contiguous().view(bs, -1, h)).view(bs, -1, w)
        global_w = global_w.transpose(1,2).contiguous() # N, W, C
        global_x = torch.cat([global_h, global_w], dim=1) # N, (H+W), C

        z = self.Local2Layout(global_x, z, mask)

        # part 3: Layout
        z = self.Layout(z)

        # part 4: Local
        x_, attn_f2m = self.Local(x, z) # contains activation DY-ReLU

        # part 5: f2m
        x_ = self.Layout2Local(x_.view(bs, self.bottleneck_channels, -1).transpose(1,2).contiguous(), 
                                attn_f2m) # x_ is like N, HW, C_bottleneck
        
        # part 6: residue link
        x_ = x_.transpose(1,2).contiguous().view(bs, self.bottleneck_channels, h, w)
        x = x + self.out_conv(x_)

        return x, z, mask # for sequential input