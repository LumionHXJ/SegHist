import torch
from torch import Tensor, nn

class DyReLU(nn.Module):
    """Modified from PaddleViT.

    Params Info:
        in_channels: input feature map channels
        embed_dims: input token embed_dims
        k: the number of parameters is in Dynamic ReLU
        coefs: the init value of coefficient parameters
        consts: the init value of constant parameters
        reduce: the mlp hidden scale,
                means 1/reduce = mlp_ratio
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 k=2, # a_1, a_2 coef, b_1, b_2 bias
                 coefs=[1.0, 0.5], # coef init value
                 consts=[1.0, 0.0], # const init value
                 reduce=4,
                 dropout=0.1,
                 mode='shared'):
        super().__init__()
        assert mode in ['shared', 'awared']
        self.mode = mode

        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.k = k

        self.mid_channels = 2 * k * in_channels

        # 4 values
        # a_k = alpha_k + coef_k*x, 2
        # b_k = belta_k + coef_k*x, 2
        self.coef = nn.Parameter(torch.tensor([coefs[0]]*k + [coefs[1]]*k))
        self.coef.requires_grad = False
        self.const = nn.Parameter(torch.tensor([consts[0]] + [consts[1]]*(2*k-1)))
        self.const.requires_grad = False

        self.project = nn.Sequential(
            nn.Linear(embed_dims, int(embed_dims/reduce)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dims/reduce), self.mid_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.mid_channels)
        )

    def forward(self, 
                feature_map:Tensor, 
                tokens: Tensor,
                attn_map: Tensor):
        '''
        Args:
            attn_score(Tensor): attn map of mobile2former before softmax operation, 
                reusing for saving computation, with shape: (B, heads, H*W, M).
        '''
        B, M, D = tokens.size()
        B, C, H, W = feature_map.size()
        if self.mode == 'shared':
            # shared mode only pick out first token
            dy_params = self.project(tokens[:, 0]) # B, 2kC            
            dy_params = dy_params.view(B, self.in_channels, 2*self.k) # B, C, 2*k
        elif self.mode == 'awared':
            # part 2: deal with decoupled attention map, keeping prob. attributes
            attn_map = torch.mean(attn_map, dim=1) # B, HW, M
            attn_map = attn_map.view(B, H, W, M)

            # part 3: projecting tokens
            dy_params = self.project(tokens).unsqueeze(1) # B, 1, M, 2kC

            # part 4: compute dynamic parameters for spatial pixel
            dy_params = torch.matmul(attn_map, dy_params).view(B, H, W, self.in_channels, 2*self.k) # B, H, W, C, 2k
            dy_params = dy_params.permute(1, 2, 0, 3, 4).contiguous() # H, W, B, C, 2k

        dy_init_params = dy_params * self.coef + self.const
        f = feature_map.permute(2, 3, 0, 1).contiguous().unsqueeze(-1) # H, W, B, C, 1

        # output shape: H, W, B, C, k
        output = f * dy_init_params[..., :self.k] + dy_init_params[..., self.k:]
        output = torch.max(output, dim=-1)[0] # H, W, B, C(fetch out max values)
        output = output.permute(2, 3, 0, 1).contiguous() # B, C, H, W

        return output