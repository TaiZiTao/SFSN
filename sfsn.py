"""
    code based on :
        -[basicsr SwinIR] github: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/swinir_arch.py
        -[Restormer] github: https://github.com/swz30/Restormer
"""
from PIL import Image
from torchvision import transforms
from thop import profile

"""
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}


@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
"""



import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from collections import OrderedDict

# for restormer
import numbers
from pdb import set_trace as stx

from einops import rearrange



# ---------------------------------------------------------------------------------------------------------------------
# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):    # for better performance and less params we set bias=False
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------

class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2

class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(self, dim, ffn_expansion_factor, bias=False,act_layer=nn.GELU, drop=0.):
        super(SGFN, self).__init__()
        in_features = dim
        out_features = dim
        hidden_features = int(dim * ffn_expansion_factor)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        _,_,H,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x
# FFN
class BaseFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        # base feed forward network in SwinIR
        super(BaseFeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, 1, bias=bias),
        )

    def forward(self, x):
        # shortcut outside
        return self.body(x)
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
##########################################################################
# ## Multi-DConv Head Transposed Self-Attention (MDTA)

# ---------------------------------------------------------------------------------------------------------------------


class HFFELayerBlock(nn.Module):
    def __init__(self, dim, idynamic_ffn_expansion_factor=2.):
        super(HFFELayerBlock, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.IDynamicDWConv = HFFE(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.IDynamic_ffn = SGFN(dim, ffn_expansion_factor=idynamic_ffn_expansion_factor, bias=False)
    def forward(self, x):
        x = self.IDynamicDWConv(self.norm1(x)) + x
        x = self.IDynamic_ffn(self.norm2(x)) + x
        return x


class DDIALayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_expansion_factor=2.):
        super(DDIALayerBlock, self).__init__()
        self.dim = dim
        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.restormer_attn = DDIA(dim, num_heads=restormer_num_heads, bias=False)
        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.restormer_ffn = SGFN(dim,ffn_expansion_factor=restormer_ffn_expansion_factor)
    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x


# ---------------------------------------------------------------------------------------------------------------------




class DDIA(nn.Module):
    """
        SparseGSA is based on MDTA
        MDTA in Restormer: [github] https://github.com/swz30/Restormer
        TLC: [github] https://github.com/megvii-research/TLC
        We use TLC-Restormer in forward function and only use it in test mode
    """
    def __init__(self, dim, num_heads, bias=False):
        super(DDIA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fft = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.act = nn.ReLU()
        self.fre_first = FFTProjection(dim)
        self.fre_second = FFTProjection(dim)
        self.spatial = SpatialProjection(dim)
        self.soft = nn.Softmax(dim=1)
    def _forward(self, qkv, hf):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hf = rearrange(hf, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)     # Sparse Attention due to ReLU's property
        out = (attn @ v)
        hf = (attn @ hf)
        return out, hf
    def forward(self, x):
        b, c, h, w = x.shape
        hf_first = self.fre_first(x)
        qkv = self.qkv_dwconv(self.qkv(x))
        out, hf = self._forward(qkv, hf_first)
        hf = rearrange(hf, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.spatial(out, hf)
        hf = hf * self.fft + hf_first
        hf = self.fre_second(hf)
        y1 = out
        y2 = hf
        w1 = torch.unsqueeze(out, dim=1)
        w2 = torch.unsqueeze(hf, dim=1)
        w = self.soft(torch.cat([w1, w2], dim=1))
        out = y1 * w[:, 0, ::] + y2 * w[:, 1, ::]
        out = self.project_out(out)
        return out


class FFTProjection(nn.Module):
    """ Frequency Projection.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim, fft_norm="ortho"):
        super().__init__()
        self.conv_layer1 = torch.nn.Conv2d(dim, dim // 4, 1, 1, 0)
        self.conv_layer3 = torch.nn.Conv2d(dim // 4, dim, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm
    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real = ffted.real + self.conv_layer3(self.relu(self.conv_layer1(ffted.real)))
        imag = ffted.imag + self.conv_layer3(self.relu(self.conv_layer1(ffted.imag)))
        ffted = torch.complex(real, imag)
        ifft_shape_slice = x.shape[-2:]
        atten = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )
        return x * atten

class SpatialProjection(nn.Module):
    """ Spatial Projection.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
    """ Frequency Projection.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(2 * dim, dim // 8, 1, 1, 0)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(dim // 8, dim, 1, 1, 0)
    def forward(self, x, y):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        res = x
        x = torch.cat((x,y),dim=1)
        attn = self.conv_2(self.act(self.conv_1(x)))
        attn = attn + 1
        out = attn * res
        return out

# ---------------------------------------------------------------------------------------------------------------------
# BuildBlocks
class BuildBlock(nn.Module):
    # Sorry for the redundant parameter setting
    # it is easier for ablation study while during experiment
    # if necessary it can be changed to **args
    def __init__(self, dim, blocks=3, buildblock_type='edge',
                 idynamic_num_heads=6, idynamic_ffn_type='GDFN', idynamic_ffn_expansion_factor=2.,
                 restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2.):
        super(BuildBlock, self).__init__()
        # those all for extra_repr
        # --------
        self.dim = dim
        self.blocks = blocks
        self.buildblock_type = buildblock_type
        self.num_heads = (idynamic_num_heads, restormer_num_heads)
        self.ffn_type = (idynamic_ffn_type, restormer_ffn_type)
        self.ffn_expansion = (idynamic_ffn_expansion_factor, restormer_ffn_expansion_factor)
        # ---------

        # buildblock body
        # ---------
        body = []

        for _ in range(blocks):
            body.append(HFFELayerBlock(dim, idynamic_ffn_expansion_factor))
            body.append(DDIALayerBlock(dim, restormer_num_heads, restormer_ffn_expansion_factor))
        body.append(nn.Conv2d(dim, dim, 3, 1, 1))   # as like SwinIR, we use one Conv3x3 layer after buildblock
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x     # shortcut in buildblock

    def extra_repr(self) -> str:
        return f'dim={self.dim}, blocks={self.blocks}, buildblock_type={self.buildblock_type}, ' \
               f'num_heads={self.num_heads}, ffn_type={self.ffn_type}, ' \
               f'ffn_expansion={self.ffn_expansion}'

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

       but for our model, we give up Traditional Upsample and use UpsampleOneStep for better performance not only in
       lightweight SR model, Small/XSmall SR model, but also for our base model.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


# Traditional Upsample from SwinIR EDSR RCAN
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------

# Network
class SFSN(nn.Module):
    r""" DLGSANet
        A PyTorch impl of : DLGSANet: Lightweight Dynamic Local and Global Self-Attention Network for Image Super-Resolution
        'IDynamic' using the idynamic transformer block
        'Restormer' using the Restormer transformer block
        'Edge' a new way inspired by EdgeViTs and EdgeNeXt
        'SparseEdge' a new way of using ReLU's properties for Sparse Attention

    Args:
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 90
        depths (tuple(int)): Depth of each BuildBlock
        num_heads (tuple(int)): Number of attention heads in different layers
        window_size (int): Window size. Default: 7
        ffn_expansion_factor (float): Ratio of feedforward network hidden dim to embedding dim. Default: 2
        ffn_type (str): feedforward network type, such as GDFN and BaseFFN
        bias (bool): If True, add a learnable bias to layers. Default: True
        body_norm (bool): Normalization layer. Default: False
        idynamic (bool): using idynamic for local attention. Default: True
        tlc_flag (bool): using TLC during validation and test. Default: True
        tlc_kernel (int): TLC kernel_size [x2, x3, x4] -> [96, 72, 48]
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction module. 'pixelshuffle'/'pixelshuffledirect'
    """

    def __init__(self,
                 in_chans=3,
                 dim=48,
                 groups=3,
                 blocks=3,
                 buildblock_type='edge',
                 idynamic_num_heads=6, idynamic_ffn_type='SGFN', idynamic_ffn_expansion_factor=6.,
                 restormer_num_heads=6, restormer_ffn_type='SGFN', restormer_ffn_expansion_factor=6.,
                 upscale=4,
                 img_range=1.,
                 upsampler='',
                 body_norm=False,
                 input_resolution=None,     # input_resolution = (height, width)
                 **kwargs):
        super(SFSN, self).__init__()

        # for flops counting
        self.dim = dim
        self.input_resolution = input_resolution
        # MeanShift for Image Input
        # ---------
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        # -----------

        # Upsample setting
        # -----------
        self.upscale = upscale
        self.upsampler = upsampler
        # -----------

        # ------------------------- 1, shallow feature extraction ------------------------- #
        # the overlap_embed: remember to set it into bias=False
        self.overlap_embed = nn.Sequential(OverlapPatchEmbed(in_chans, dim, bias=False))

        # ------------------------- 2, deep feature extraction ------------------------- #
        m_body = []

        # Base on the Transformer, When we use pre-norm we need to build a norm after the body block
        if body_norm:       # Base on the SwinIR model, there are LayerNorm Layers in PatchEmbed Layer between body
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        for i in range(groups):
            m_body.append(BuildBlock(dim, blocks, buildblock_type,
                 idynamic_num_heads, idynamic_ffn_type, idynamic_ffn_expansion_factor,
                 restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor))

        if body_norm:
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        m_body.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1)))

        self.deep_feature_extraction = nn.Sequential(*m_body)

        # ------------------------- 3, high quality image reconstruction ------------------------- #

        # setting for pixelshuffle for big model, but we only use pixelshuffledirect for all our model
        # -------
        num_feat = 64
        embed_dim = dim
        num_out_ch = in_chans
        # -------

        if self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, input_resolution=self.input_resolution)

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        pass    # all are in forward function including deep feature extraction

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.upsample(x)

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        else:
            # for image denoising and JPEG compression artifact reduction
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_last(x)

        x = x / self.img_range + self.mean

        return x

class HFFE(nn.Module):
    def __init__(self, dim):
        super(HFFE, self).__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.pw = ADACS(dim)
        self.conv3_7 = nn.Conv2d(dim // 3, dim // 3, kernel_size=3, padding=3, groups=dim // 3, dilation=3,
                                 padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim // 3, dim // 3, kernel_size=5, padding=6, groups=dim // 3, dilation=3,
                                  padding_mode='reflect')
        self.conv3_3 = nn.Conv2d(dim // 3, dim // 3, kernel_size=3, padding=1, groups=dim // 3, dilation=1,
                                 padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim // 3, dim // 3, kernel_size=7, padding=9, groups=dim // 3, dilation=3,
                                  padding_mode='reflect')
        self.conv3_5 = nn.Conv2d(dim // 3, dim // 3, kernel_size=5, padding=2, groups=dim // 3, dilation=1,
                                 padding_mode='reflect')
        self.conv3_21 = nn.Conv2d(dim // 3, dim // 3, kernel_size=5, padding=10, groups=dim // 3, dilation=5,
                                  padding_mode='reflect')

        self.mix1 = nn.Sequential(
            nn.Conv2d(dim // 3 * 2, 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(dim // 3 * 2, 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.mix3 = nn.Sequential(
            nn.Conv2d(dim // 3 * 2, 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        pw = self.pw(x)
        x, y, z = self.conv0(pw).chunk(3, dim=1)
        x1 = self.conv3_7(x)
        x2 = self.conv3_13(x1)
        w = self.mix1(torch.cat([x1, x2], dim=1))
        x = w[:, 0, :, :].unsqueeze(1) * x1 + w[:, 1, :, :].unsqueeze(1) * x2
        y1 = self.conv3_3(y)
        y2 = self.conv3_19(y1)
        w = self.mix2(torch.cat([y1, y2], dim=1))
        y = w[:, 0, :, :].unsqueeze(1) * y1 + w[:, 1, :, :].unsqueeze(1) * y2
        z1 = self.conv3_5(z)
        z2 = self.conv3_21(z1)
        w = self.mix3(torch.cat([z1, z2], dim=1))
        z = w[:, 0, :, :].unsqueeze(1) * z1 + w[:, 1, :, :].unsqueeze(1) * z2
        out = torch.cat([x, y, z], dim=1)
        out = self.act(self.conv1(out)) * res
        return out


class ADACS(nn.Module):
    def __init__(self, dim, ratio=8):
        super(ADACS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // ratio, kernel_size=1)
        self.conv2 = nn.Conv2d(dim // ratio, dim * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(dim, dim, 1)
    def batch_shift_pytorch(features, shift_vectors):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, C, H, W = features.shape  # 假设输入的维度是 [B, C, H, W]
        # 创建坐标网格
        grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_y, grid_x), dim=-1).float().unsqueeze(0).unsqueeze(0)#.to(device) # [1, 1, H, W, 2]

        # 将shift_vectors应用于每个通道，生成平移后的坐标
        shift_vectors = shift_vectors.view(B, C, 1, 1, 2)#.to(device)  # 扩展维度 [B, C, 1, 1, 2]
        shift_vectors = F.normalize(shift_vectors, dim=4)#.to(device)
        grid = grid + shift_vectors  # 应用平移 [B, C, H, W, 2]

        # 归一化坐标，将坐标从 [0, H-1] 转换到 [-1, 1]
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # 水平坐标归一化
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # 垂直坐标归一化

        # 增加批量和通道维度，将 grid 进行转换为适合 grid_sample 的输入
        grid = grid.reshape(B * C, H, W, 2)

        # 对特征图进行平移操作
        features = features.reshape(B * C, 1, H, W)  # 调整形状为 [B * C, 1, H, W]
        shifted_features = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # 恢复形状为 [B, C, H, W]
        shifted_features = shifted_features.reshape(B, C, H, W)
        return shifted_features
    def forward(self, x):
        shift = self.avg_pool(x)
        shift = self.conv2(self.relu(self.conv1(shift)))
        output = ADACS.batch_shift_pytorch(x, shift)
        output = self.conv3(output)
        return output


if __name__ == '__main__':
    # 1. 定义与训练时完全一致的模型结构
    model = SFSN(dim=48, upscale=4, img_range=1, groups=3, blocks=3, buildblock_type='sfsn',
                 upsampler='pixelshuffledirect')


    # 2. 加载权重
    state_dict = torch.load(
        'E:/paper/code/Local-Attribution-Maps-for-Super-Resolution-main/ModelZoo/weights/SFSN/net_g_AID_x4.pth',
        map_location='cpu')

    # BasicSR 的权重可能包含 'params' 或 'params_ema' 等键，需要提取
    if 'params' in state_dict:
        state_dict = state_dict['params']
    elif 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']

    # 3. 加载到模型中
    model.load_state_dict(state_dict, strict=True)  # 建议先用 strict=True 检查是否完全匹配
    model.eval()
    input = torch.randn(1, 3, 320, 180)

    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9} G, Params: {params / 1e3} K")
    # 4. 推理
    image = Image.open(
        'E:/paper/code/Local-Attribution-Maps-for-Super-Resolution-main/resources/test_images/airplane80.tif').convert(
        'RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    lr_tensor = transform(image).unsqueeze(0)  # (1,3,H,W)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # 5. 后处理与保存（注意输出范围）
    sr_img = sr_tensor.squeeze(0).clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(sr_img).save('output.png')