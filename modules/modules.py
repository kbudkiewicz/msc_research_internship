import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from configs import BaseUnetConfig, BaseVitConfig


class MlpBlock(nn.Module):
    """
    Default MLP Block for ViT Encoder Block.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 64,
        n_blocks: int = 3,
        activation: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout_rate),
            activation(inplace=True),
            nn.Linear(out_dim, in_dim),
        )
        self.mlp = nn.Sequential(
            *nn.ModuleList([block for _ in range(n_blocks)])
        )
        if device:
            self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class SinusoidalEmbedding(nn.Module):
    """
    Adds a sinusoidal embedding to the input. The embedding is broadcasted for batched inputs too.

    Args:
        dropout_p (float): Dropout probability.
        max_len(int, Optional): Maximum context window.

    Shapes:
        - Output (Tensor): (batch_size,) max_len, embed_dim
    """
    def __init__(
        self,
        embed_dim: int = 256,
        dropout_p: float = 0.0,
        max_len: int = 512,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_p)

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
        if x.shape[0] > 1:  # broadcast to (B, seq_len, D)
            x += self.pe.repeat(x.shape[0], 1, 1)
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
        Expects the input `Tensor` to be of shape `(B,L,D)`, where `B` is the batch size, `L` is the sequence length
        (`Q` for key length, `K` for value length), and `D` is the embedding dimension of the model.

        Args:
            q (Tensor): Query inputs :math:`(B,Q,D)`.
            k (Tensor): Keys inputs :math:`(B,K,D)`.
            v (Tensor): Values inputs :math:`(B,K,D)`.
            mask (Tensor, Optional): Causal or Padding mask :math:`(B,1,Q,K)`.

        Return:
            - output: :math:`(BxQxD)`
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
        mlp_dim (int, Optional): Latent dimension of the MLP of the Encoder block.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        dropout_rate: float = 0.0,
        mlp_dim: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, n_heads=n_heads, dropout_rate=dropout_rate, device=device)
        self.mlp = MlpBlock(in_dim=embed_dim, out_dim=mlp_dim if mlp_dim else embed_dim, dropout_rate=dropout_rate, device=device)
        self.layer_norm1 = nn.LayerNorm(embed_dim, device=device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, device=device)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): Input sequence to the Encoder block
            mask (Tensor, Optional): Padding mask
        """
        att = self.mha(x, x, x, mask=mask)
        res = self.layer_norm1(att) + x
        att = self.mlp(res)
        att = self.layer_norm2(att) + res

        return att


class ConvBlock(nn.Module):
    """
    A simple Convolutional block with normalization and activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module): Activation layer. Defaults to ``nn.Relu()``
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
            activation(inplace=True),  # for faster calculation
            nn.Conv2d(in_channels, out_channels, **conv_kwargs),
            nn.GroupNorm(4, out_channels),  # alternatively, nn.BatchNorm(in_channels)
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
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        do_upscale (bool, Optional): Whether the residual block is an up-scaling or down-scaling block. Defaults to `True`.
            Depending on the value additional Pooling or Up-sampling layers are initialized, respectively.

    Shapes:
        - Input: :math:`(B, C_in, H, W)`
        - Output: :math:`(B, C_out, (H-4)*F, (W-4)*F)` if ``do_upscale=True``, else :math:`(B, C_out, (H-4)/F, (W-4)/F)`
        See the documentation of `torch.nn.Upsample` and `torch.nn.MaxPool2d` for more details.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        do_upscale: Optional[bool] = None,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        if do_upscale is not None:
            self.do_upscale = do_upscale
            self.do_downscale = not do_upscale
        else:
            self.do_upscale = self.do_downscale = False
        self.conv_1 = ConvBlock(in_channels, out_channels, **self.conv_kwargs)
        self.conv_2 = ConvBlock(out_channels, out_channels, **self.conv_kwargs)
        # TODO: implement time_dmbd_dim
        self.time_embd_dim = 32
        self.time_adapter = nn.Sequential(
            nn.Linear(self.time_embd_dim, self.time_embd_dim),
            nn.SiLU(),
            nn.Linear(self.time_embd_dim, out_channels),
        )

        if self.do_upscale:
            self.upsample = nn.Upsample(**self.upscale_kwargs)
        if self.do_downscale:
            self.pool = nn.MaxPool2d(**self.pool_kwargs)
        self.label_adapter = nn.Sequential(
            nn.Linear(self.time_embd_dim, self.time_embd_dim),
            nn.SiLU(),
            nn.Linear(self.time_embd_dim, out_channels),
        )
        if device:
            self.to(device)

    def forward(self, x: Tensor, time_embd: Tensor = None, label_embd: Optional[Tensor] = None) -> Tensor:
        # res = x.clone()
        x = self.conv_1(x)
        # x += self.time_adapter(time_embd)[..., None, None]
        if label_embd:
            x += self.label_adapter(label_embd)[..., None, None]
        x = self.conv_2(x)
        # x + res # TODO: res connection

        if self.do_downscale:
            x = self.pool(x)
        if self.do_upscale:
            x = self.upsample(x)

        return x
