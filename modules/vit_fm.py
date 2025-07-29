import torch
import torch.nn as nn
import math


class FourierEncoder(nn.Module):
    """
    Taken from MIT - Flow matching course
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)


class ViTFlowMatchingConditional(nn.Module):
    def __init__(self,
                 img_size=128,
                 patch_size=16,
                 in_chans=1,
                 num_classes=5,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Patch position embedding. We use a simple lookup embedding here
        self.pos_embed = nn.Embedding(self.num_patches, embed_dim)

        # Timestep embedding
        self.t_emb_proj = nn.Sequential(
            FourierEncoder(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # Label embedding
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.label_emb_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Output head
        self.head = nn.Linear(embed_dim, in_chans * patch_size * patch_size)  # patch-to-patch

    def forward(self, x, t, labels):
        """
        x: [B, C, H, W] image
        t: [B, 1, 1, 1] float tensor, timestep
        labels: [B, 1] int tensor, class labels
        """
        B = x.shape[0]
        # Patchify image
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        # Patch position ids
        pos_ids = torch.arange(self.num_patches, device=x.device).unsqueeze(0)  # [1, num_patches]
        pos_emb = self.pos_embed(pos_ids).expand(B, -1, -1)  # [B, num_patches, embed_dim]
        x = x + pos_emb

        # Timestep embedding
        t_emb = self.t_emb_proj(t).unsqueeze(1)  # [B, 1, embed_dim]
        x = x + t_emb  # add to all patches

        # Label embedding
        lbl_emb = self.label_embed(labels.view(B,))  # [B, label_emb_dim]
        lbl_emb = self.label_emb_proj(lbl_emb).unsqueeze(1)  # [B, 1, embed_dim]
        x = x + lbl_emb  # broadcast to all patches

        # Transformer
        x = self.transformer(x)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        out = self.head(x)  # [B, num_patches, out_dim]

        # Reconstruct image shape from patches
        patch_H = patch_W = int(math.sqrt(self.num_patches))
        out = out.view(B, patch_H, patch_W, self.in_chans, self.patch_size, self.patch_size)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, C, patch_H, P, patch_W, P)
        out = out.view(B, self.in_chans, patch_H * self.patch_size, patch_W * self.patch_size)  # [B, C, H, W]
        return out
