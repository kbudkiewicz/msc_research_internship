import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from configs import BaseUnetConfig


class MlpBlock(nn.Module):
    """
    Default MLP Block for ViT Encoder Block. The block consists of a linear layer, activation function, dropout and
    linear layer, in that order.

    Args:
        in_dim (int): Dimension of input features.
        out_dim (int): Dimension of output features.
        activation (nn.Module): Activation function.
        dropout_rate (float): Dropout probability.
        device (optional, torch.device, str): Device used for computation.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 64,
        activation: nn.Module = nn.GELU,
        dropout_rate: float = 0.0,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, in_dim)
        )
        if device:
            self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class FourierEmbedding(nn.Module):
    """
    Taken from MIT - Flow matching course
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.register_parameter('weights', nn.Parameter(torch.randn(1, dim//2)))

    def forward(self, t: Tensor) -> Tensor:
        """
        Shapes:
            - Input t: (B,)
            - Output: (B, dim)
        """
        t = t.unsqueeze(-1)
        freqs = t * self.weights * 2 * math.pi   # -> (B, dim//2)
        sin_embd = torch.sin(freqs)
        cos_embd = torch.cos(freqs)
        return torch.cat([sin_embd, cos_embd], dim=-1) * math.sqrt(2)   # (B, dim)


class SinusoidalEmbedding(nn.Module):
    """
    Adds a sinusoidal embedding to the input. The embedding is broadcast for batched inputs too.

    Args:
        max_len(int, Optional): Maximum context window.
        embed_dim (int): Embedding dimension.
        dropout_rate (float): Dropout probability.
        device (optional, torch.device, str): Device used for computation.

    Shapes:
        - Input (Tensor):  (batch_size,) max_len
        - Output (Tensor): (batch_size,) max_len, embed_dim
    """
    def __init__(
        self,
        max_len: int = 512,
        embed_dim: int = 256,
        dropout_rate: float = 0.0,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)

        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device) * -(math.log(10_000) / self.embed_dim)
        )
        k = torch.arange(0, self.max_len, device=device).unsqueeze(1)   # (L, 1)
        pe = torch.zeros(self.max_len, self.embed_dim, device=device)   # (L, D)
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        self.pe = pe.to(device)
        del pe

        if device:
            self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            - Input (Tensor): Timesteps of shape (B,)
            - Output (Tensor): Embedding of shape (B,) max_len, embed_dim
        """
        x = self.pe.repeat(len(x), 1, 1)
        return self.dropout(x)

    @property
    def get_embedding(self):
        return self.pe


class Patchify(nn.Module):
    r"""
    Module for patchifying an image with a fixed patch size. Based on the original implementation in `An Image is Worth
    more than 16x16 Words: Transformers for Image Recognition At Scale <https://arxiv.org/abs/2010.11929>`__.

    The total number of patches :math:`N` equals :math:`HxW/P^2`. It also is "the effective input sequence length for
    the Transformer."

    The output shape is inferred automatically. If the input is not batched or is a 3D ``Tensor`` the output will be
    recast to :math:`(1, N, CP^2)`.

    Args:
        patch_size (int): The patch size.
    """
    def __init__(
        self,
        patch_size: int = 16,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        if device:
            self.to(device)

    def __call__(self, img: Tensor, patch_size: Optional[int] = None) -> Tensor:
        """
        Patchify an input image for the Vision Transformer to process.

        Args:
            img (Tensor): Input image to be divided into patches.
            patch_size (int, Optional): Patch size (P).

        Shapes:
            - img (Tensor): :math:`B, (C, H, W)`
            - patches (Tensor): :math:`B, (N, CP^2)`

        Return:
            Patches.
        """
        assert img.ndim in {3, 4}, f'Input image must be 3D or 4D but is {img.ndim}'
        patch_size = patch_size if patch_size else self.patch_size

        c, h, w = img.shape[-3:]
        n = int(h*w / patch_size**2)
        img = img.reshape(-1, n, c*patch_size**2)

        return img


class MultiHeadAttention(nn.Module):
    r"""Implementation of the Multi-head Attention block from `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{Multihead}(Q,K,V) = \text{Concat}(head_1, \dots, head_h)W^O
        \text{ where } head_i = \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)

    Args:
        embed_dim (int): Embedding dimension (D) of the module. Is later split into multiple heads with dimensions
            ``embed_dim//num_heads``
        n_heads (int): Number of attention heads (H).
        dropout_rate (float): Dropout probability. Default: ``0.0`` (no dropout).
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout_rate: float = 0.,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        self.head_dim = embed_dim // n_heads    # HD
        assert self.head_dim * n_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scalar = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        if device:
            self.to(device)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Expects the input `Tensor` to be of shape `(B, L, D)`, where `B` is the batch size, `L` is the sequence length
        (`Q` for key length, `K` for value length), and `D` is the embedding dimension of the model.

        Args:
            q (Tensor): Query inputs :math:`(B,Q,D)`.
            k (Tensor): Keys inputs :math:`(B,K,D)`.
            v (Tensor): Values inputs :math:`(B,K,D)`.
            mask (Tensor, Optional): Causal or Padding mask :math:`(B,1,Q,K)`.

        Return:
            - output: :math:`(B,Q,D)`
        """
        b, q_len, d = q.size()  # batch_size, queries_length, model_dim
        _, k_len, _ = k.size()  # keys_length
        q = self.q_proj(q).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)    # -> BxHxQxHD
        k = self.k_proj(k).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)    # -> BxHxKxHD
        v = self.v_proj(v).view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)    # -> BxHxKxHD
        q *= self.scalar

        att = q @ k.mT    # BxHxQxHD x BxHxHDxK -> ...xQxK
        if mask is not None:
            att += mask
        att = att.softmax(dim=-1)
        att = self.dropout(att)
        att = att @ v   # ...xQxK x ...xKxHD -> ...xQxHD

        att = att.transpose(1, 2).contiguous().view(b, q_len, self.embed_dim)
        att = self.out_proj(att)    # -> BxQxD

        return att


class TransformerEncoderBlock(nn.Module):
    r"""Encoder Block of a transformer

    Args:
        embed_dim (int): Latent dimension of the Multi-head Attention block.
        n_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate.
        device (optional, torch.device, str): Device used for computation.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        dropout_rate: float = 0.0,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, n_heads=n_heads, dropout_rate=dropout_rate, device=device)
        self.mlp = MlpBlock(in_dim=embed_dim, out_dim=4*embed_dim, dropout_rate=dropout_rate, device=device)
        self.layer_norm1 = nn.LayerNorm(embed_dim, device=device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, device=device)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): Input sequence to the Encoder block. :math:`(B, max_len, embed_dim)`
            mask (Tensor, Optional): Padding mask :math:`(B, 1, q_len, k_len)`
        """
        att = self.mha(x, x, x, mask=mask)
        res = self.layer_norm1(att) + x
        att = self.mlp(res)
        att = self.layer_norm2(att) + res

        return att


class ReZeroEncoderBlock(nn.Module):
    """
    Basic ReZero Transformer Encoder block. Based on `ReZero is All You Need: Fast Convergence at Large Depth
    <https://arxiv.org/abs/2003.04887>`_.

    Args:
        embed_dim (int): Latent dimension of the Multi-head Attention block.
        n_heads (int): Number of attention heads.
        dropout_rate (float, Optional): Dropout rate.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        dropout_rate: float = 0.,
        res_weight: float = 0.,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        assert 0 <= res_weight <= 1, 'res_weight must be between 0 and 1'

        self.mha = MultiHeadAttention(embed_dim, n_heads=n_heads, dropout_rate=dropout_rate, device=device)
        self.mlp = MlpBlock(in_dim=embed_dim, out_dim=4*embed_dim, dropout_rate=dropout_rate, device=device)
        self.res_weight = nn.Parameter(torch.Tensor([res_weight]))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): Input sequence to the Encoder block
            mask (Tensor, Optional): Padding mask
        """
        x = self.mha(x, x, x, mask=mask) * self.res_weight + x
        x = self.mlp(x) * self.res_weight + x

        return x


class ConvBlock(nn.Module):
    """
    A simple Convolutional block consisting of an activation function, BatchNorm2d and Conv2d, in that order.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module): Activation layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module = nn.SiLU,
        device: Optional[torch.device | str] = None,
        **conv_kwargs,
    ):
        super().__init__()
        self.net = nn.Sequential(
            activation(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, **conv_kwargs),
        )
        if device:
            self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Expects a 4D mini-batch of 2D inputs.
        """
        return self.net(x)


class ResBlock(nn.Module, BaseUnetConfig):
    r"""
    Base class for residual blocks.

    Args:
        channels (int): Number of input and output channels.
        t_dim (int): Dimension of time-adapter embedding.
        label_dim (int): Dimension of label embedding.

    Shapes:
        - Input: :math:`(B, C_in, H, W)`
        - Output: :math:`(B, C_out, (H-4)*F, (W-4)*F)` if ``do_upscale=True``, else :math:`(B, C_out, (H-4)/F, (W-4)/F)`
        See the documentation of `torch.nn.Upsample` and `torch.nn.MaxPool2d` for more details.
    """
    def __init__(
        self,
        channels: int,
        t_dim: int,
        label_dim: int,
        res_weight: float = 0.,
    ):
        super().__init__()
        assert 0. <= res_weight <= 1., 'res_weight must be between 0 and 1'

        self.in_block = ConvBlock(channels, channels, **self.conv_kwargs)
        self.out_block = ConvBlock(channels, channels, **self.conv_kwargs)
        self.time_adapter = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, channels),
        )
        self.label_adapter = nn.Sequential(
            nn.Linear(label_dim, label_dim),
            nn.SiLU(),
            nn.Linear(label_dim, channels),
        )
        self.res_weight = nn.Parameter(torch.Tensor([res_weight]))

    def forward(self, x: Tensor, t_embd: Tensor, labels_embd: Tensor) -> Tensor:
        """
        Shapes::
            - x: :math:`(B, C, H, W)`
            - t: :math:`(B, D)`
            - labels: :math:`(B, D)`
        """
        res = x.clone()
        x = self.in_block(x)

        # embeddings
        t_embd = self.time_adapter(t_embd).unsqueeze(-1).unsqueeze(-1)              # -> (B, C, 1, 1)
        labels_embd = self.label_adapter(labels_embd).unsqueeze(-1).unsqueeze(-1)   # -> (B, C, 1, 1)
        x += t_embd
        x += labels_embd

        x = self.out_block(x) * self.res_weight + res

        return x


class Rescaler(nn.Module):
    """
    Block working as a decoder and encoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        t_dim (int): Dimension of timestep embedding.
        label_dim (int): Dimension of label embedding.
        depth (int): Number of ResBlock repetitions.
        upscale (bool): Whether to upscale the input. Defaults to ``True`` and the block is equivalent to a decoder
            upsampling the input . If ``False``, the block is equivalent to an encoder downscaling the input.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_dim: int,
        label_dim: int,
        depth: int,
        upscale: bool = True,
    ):
        super().__init__()
        self.upscale = upscale

        self.blocks = nn.ModuleList([
            ResBlock(out_channels if upscale else in_channels, t_dim, label_dim) for _ in range(depth)
        ])
        if upscale:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor, t_embd: Tensor, labels_embd: Tensor) -> Tensor:
        """
        Shapes:
            - x: :math:`(B, C, H, W)`
            - t: :math:`(B, D)`
            - labels: :math:`(B, D)`
        """
        if self.upscale:
            x = self.upsample(x)

        for block in self.blocks:
            x = block(x, t_embd, labels_embd)

        if not self.upscale:
            x = self.downsample(x)
        return x
