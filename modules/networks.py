import os
import torch
import torch.nn as nn

from math import sqrt
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Union
from modules.modules import FourierEmbedding, TransformerEncoderBlock, ResBlock, ConvBlock, Rescaler

from torch import Tensor
from torch.nn.functional import mse_loss


class Unet(nn.Module):
    r"""
    Basic `UNet <http://arxiv.org/abs/1505.04597>`_ implementation.

    A cache is used to save the cropped tensors for the successive skip-connections to  the ``UpscaleBlocks`` of the
    network decoder.

    Args:
        dims (iter): Hidden layer dimensions, e.g. (64, 128, 256, 512). Does not include the input, output dimensions,
            but does include the latent dimension
        embed_dim (int): Embedding dimension of time-steps and labels.
        n_labels (int): Number of labels of the dataset. Default is 4.
        config (object, Optional): Configuration class containing new set of hyper parameters.
        device (torch.device, Optional): Device on which the model is trained on.

    Shapes:
        - Input: :math:`(B, in_dim, H, W)`
        - Output: :math:`(B, out_dim, H, W)`
    """
    def __init__(
        self,
        *dims: int,
        embed_dim: int,
        in_dim: int = 1,
        out_dim: int = 1,
        residual_depth: int = 3,
        n_labels: int = 4,  # num_classes \equiv glioma, notumor, ... = 4
        config: object = BaseUnetConfig,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        if config:
            [self.__setattr__(k, v) for k, v in config.__dict__.items() if k[0] != '_']
        if not isinstance(dims, Iterable):
            raise ValueError(f'Unet dims must be an Iterable of ints but is {type(dims)}')
        assert len(dims) > 1, 'dims must contain at least 2 integers'

        self.cache = {}
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dims = dims
        self.embed_dim = embed_dim
        self.depth = residual_depth
        self.null_token = n_labels

        # additional label ('null token') for unconditional model training
        self.embed_labels = nn.Embedding(n_labels + 1, self.embed_dim)     # (B) -> (B, D)
        self.embed_t = FourierEmbedding(self.embed_dim)

        self.in_layer = ConvBlock(in_dim, dims[0], **{'kernel_size': 3, 'padding': 1})   # c = 1 -> dims[0]
        self.out_layer = nn.Conv2d(dims[0], out_dim, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            self.encoders.append(Rescaler(dim_in, dim_out, self.embed_dim, self.embed_dim, depth=self.depth, upscale=False))
            self.decoders.append(Rescaler(dim_out, dim_in, self.embed_dim, self.embed_dim, depth=self.depth, upscale=True))
        self.midcoder = nn.ModuleList([
            ResBlock(dims[-1], self.embed_dim, self.embed_dim) for _ in range(self.depth)
        ])

        if device:
            self.to(device)

    def forward(self, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): Batch of images.
            t (Tensor): Time-steps.
            labels (Tensor): Image class labels. Must contain ints, i.e. be of dtype torch.long .

        Shapes:
            - x: (B, C, H, W)
            - t: (B,) \subset [0,1]
            - labels: (B,)
        """
        # Embeddings
        t_embd = self.embed_t(t)  # (B, D)
        label_embd = self.embed_labels(labels)    # (B, D)

        x = self.in_layer(x)

        # 1. Downscale: downscale blocks and save intermediate skip connection tensors to cache
        for i, encoder in enumerate(self.encoders):
            x = encoder(x, t_embd, label_embd)
            self.cache[i] = x.clone()

        # 2. Midcoder
        for block in self.midcoder:
            x = block(x, t_embd, label_embd)

        # 3. Upscale: upscale blocks and add appropriate tensors from cache
        for decoder, res in zip(reversed(self.decoders), reversed(self.cache.values())):
            x += res
            x = decoder(x, t_embd, label_embd)

        x = self.out_layer(x)

        return x


class VisionTransformer(nn.Module):
    """
    Implementation of the Vision Transformer (ViT) from `An Image is Worth 16 x 16 Words:
    Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Note:
        The additional label is for classifier free guidance.

    Args:
        depth_encoder (int): The number of blocks in the decoder.
        patch_size (int): Size of the quare-shaped patch passed through the network. Default is 16; equal to the size
            used in the original ViT implementation.
        embed_dim (int): Embedding dimension of attention heads.
        mlp_dim (int): Embedding dimension of MLP blocks.
        dropout_rate (float): Frequency of neuron dropout.
        config (object): Configuration containing a different set of hyperparameters. Defaults to BaseVitConfig.
        device (device, str, optional): Device on which the model is trained on.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        depth_encoder: int = 5,
        patch_size: int = 16,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_labels: int = 4,
        dropout_rate: float = 0.0,
        config: Optional[dataclass] = None,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()

        if config:
            for k, v in config.__dict__.items():
                if k[0] != '_':
                    self.__setattr__(k, v)
        else:
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.mlp_dim = mlp_dim
            self.n_heads = n_heads
            self.dropout_rate = dropout_rate
            self.null_token = n_labels
        self.n_labels = n_labels
        self.n_patches = int(img_size ** 2 / self.patch_size ** 2)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Embeddings
        self.pos_embd = nn.Embedding(self.n_patches, self.embed_dim)
        # additional token for unconditional CFG
        self.label_embd = nn.Sequential(
            nn.Embedding(self.n_labels + 2, self.embed_dim),    # B -> B, D
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU()
        )
        self.timestep_embd = nn.Sequential(
            FourierEmbedding(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU()
        )
        self.patch_embd = nn.Conv2d(self.in_channels, self.embed_dim, self.patch_size, self.patch_size)
        self.encoder = nn.ModuleList(
            TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                mlp_dim=int(4 * self.embed_dim),
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                device=self.device
            )
            for _ in range(depth_encoder)
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.in_channels * self.patch_size**2),
        )

        self.to(self.device)

    def forward(self, img: Tensor, t: Tensor, labels: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Pass a batch of images and mask to Vision Transformer.

        Args:
            img (Tensor): Input image to be passed through. (B, C, H, W)
            t (Tensor): Time-steps. (B,)
            labels (Tensor, optional): Corresponding image labels. (B,)
            mask (Tensor, optional): Padding mask. (B, 1, D, D)
        """
        assert t.shape == labels.shape, 'labels and t must have equal shapes.'
        # Embeddings
        img = self.patch_embd(img)  # B, embed_dim, H/patch_size, W/patch_size
        img = img.flatten(2).transpose(1, 2)  # B, n_patches, embed_dim

        patch_ids = torch.arange(self.n_patches, device=self.device).unsqueeze(0)
        # Below all have shape: B, 1, embed_dim
        pos_embd = self.pos_embd(patch_ids).expand(img.shape[0], -1, -1)
        t_embd = self.timestep_embd(t).unsqueeze(1)
        label_embd = self.label_embd(labels).unsqueeze(1)

        img += pos_embd
        img += t_embd
        img += label_embd

        # forward pass
        for block in self.encoder:
            img = block(img, mask=mask)
        img = self.out_proj(img)    # B, n_patches, c*patch_size**2
        img = self.from_patches(img)

        return img

    def from_patches(self, img: Tensor) -> Tensor:
        """
        View patches as an image.

        Shapes:
            - img (Tensor): (B,) N, CP^2, where :math:`N=HxW/P^2` is ``n_patches``.
            - output (Tensor): (B,) C, H, W
        """
        b, *_ = img.shape
        h = w = int(sqrt(self.n_patches * self.patch_size**2))
        img = img.contiguous().view(b, self.in_channels, h, w)

        return img


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_cfg(self, x_t: Tensor, t: Tensor, labels: Tensor, guidance_scale: float = 1.) -> Tensor:
        r"""
        Return Classifier-Free score. Used at *inference time only* for qualitative analysis.

        .. math::
            \tilde\epsilon_\theta = w\epsilon_\theta(x,c) + (1-w) \epsilon_\theta(x,\emptyset)

        where :math:`\emptyset` is the null token for unconditional training.

        Shapes:
            - x_t: (B, C, H, W)
            - t: (B, )
            - labels: (B, )
            - Output: Tensor of shape (B, C, H, W)
        """
        null_labels = torch.full_like(labels, 4, dtype=torch.long, device=self.device)  # (B,)
        return (1. - guidance_scale) * self.net(x_t, t, null_labels) + guidance_scale * self.net(x_t, t, labels)

    def get_classifier_free_labels(self, labels: Tensor, rate: float = 0.2) -> Tensor:
        r"""
        Get classifier-free labels for the model by randomly setting some labels to a null token. Based off
        `Classifier-Free Diffusion Guidance <http://arxiv.org/abs/2207.12598>`_. Also see MIT 6.S184 lecture for
        the implementation.

        The original labels are substituted are replaced with a fixed null token :math:`\emptyset` for training the
        unconditional classifier. The labels are substituted only if :math:`p_\text{uncond}` is less then a randomly
        sampled float from [0,1].

        Args:
            labels (Tensor): Labels for each image. Dtype has to be ``torch.float``.
            rate (float): Probability of substitution of a conditional image label with a null token.
        """
        p_uncond = torch.rand(labels.shape[0], device=self.device)
        labels = torch.where(p_uncond < rate, 4, labels)  # 4. as the null token

        return labels

    @abstractmethod
    def sample_img(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def backprop(
        self,
        y1: Tensor,
        y2: Tensor,
        loss_fnc: Callable[[Tensor, Tensor], Tensor] = mse_loss,
        do_return: bool = False
    ) -> Tensor | None:
        """
        Perform a backpropagation step. If no ``loss_fnc`` is not provided, the loss defaults to MSE.

        Args:
            y1 (Tensor): Output of forward pass.
            y2 (Tensor): Expected (true) value.
            loss_fnc (Callable, nn.Module): Loss function to be minimised.
            do_return (bool): Whether to return the loss value. Defaults to ``False``.
        """
        assert isinstance(loss_fnc, Callable), f'The loss function must be Callable but is {type(loss_fnc)}.'
        try:
            loss = loss_fnc(y1, y2)
        except NameError or TypeError:
            loss = torch.nn.functional.mse_loss(y1, y2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        self.optimizer.step()

        if do_return:
            return loss

    def save_checkpoint(self, path: str | os.PathLike) -> None:
        """
        ``path`` should be a path to the repo root.
        """
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        path = os.path.join('./checkpoints', path)
        self.save_state_dict(path, 'checkpoint.pt')

    def save_state_dict(self, path: str | os.PathLike, filename: Optional[str | os.PathLike] = None) -> None:
        filename = filename if filename else 'params.pt'
        if not os.path.exists(path):
            torch.save(self.state_dict(), filename)
        else:
            torch.save(self.state_dict(), os.path.join(path, filename))

    def set_config(self, config: object) -> None:
        """
        Set network attributes to those found in ``config``.
        """
        if isinstance(config, object):
            [self.net.__setattr__(k, v) for k, v in config.__dict__.items() if not k.startswith('_')]
        else:
            raise TypeError(f'Config must be a dict, but got {type(config)}')

    def get_config(self) -> dict:
        if self.net.config is not None:
            return self.net.config
        else:
            return {k: v for k, v in self.net.__dict__.items() if not k.startswith('_')}


class FlowMatchingNet(CustomModel):
    """
    Flow Matching Net.

    Args:
        net (nn.Module): Neural Network used to approximate the velocity field.
        guidance_scale (float): Guidance scale. Determines the strength of model conditionality. The higher the value,
            the higher the model conditioning.
        classifier_free_rate (float): Probability of substituting conditional tokens for unconditional ones.
        lr (float): Learning rate.
        config (object, optional): Configuration object.
        device (torch.device | str, optional): Device platform used for computation.
    """
    def __init__(
        self,
        net: nn.Module,
        guidance_scale: float = 3.,
        classifier_free_rate: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        config: Optional[object] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()
        self.device = device
        self.net = net
        if config:
            self.set_config(config)
        self.guidance_scale = guidance_scale
        self.classifier_free_rate = classifier_free_rate
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        if device:
            self.to(device)

    def forward(self, x_t: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            x_t (Tensor): Batch of images (B, C, H, W)
            t (Tensor): Time-steps (B,)
            labels (Tensor): Class labels (B,)
        """
        labels = self.get_classifier_free_labels(labels)
        return self.net(x_t, t, labels)

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, label: Tensor, guidance_scale: float) -> Tensor:
        r"""
        Perform an ODE solving step to reconstruct the image step-by-step. Uses the midpoint solver given by:

        .. math::
            y_{n+1} = y_n + h*f[t_n + h/2, y_n + h/2*f(t_n,y_n)]

        See `Flow Matching Guide and Code <https://arxiv.org/abs/2412.06264>`__ for more information.
        """
        delta = t_end - t_start
        return x_t + delta * self.forward_cfg(
            x_t + self.forward_cfg(x_t, t_start, label, guidance_scale=guidance_scale) * (delta / 2),
            t_start + (delta / 2),
            label,
            guidance_scale=guidance_scale
        )

    def sample_img(self, img_size: int, label: Tensor, n_steps: int = 50, guidance_scale: float = 1.):
        if not isinstance(img_size, int):
            raise TypeError(f'img_size must be int but is {type(img_size)}')
        if not isinstance(label, Tensor):
            raise TypeError(f'label must be Tensor but is {type(label)}')

        timesteps = torch.linspace(0., 1., n_steps + 1, device=self.device).unsqueeze(-1)
        img = torch.randn([1, 1, img_size, img_size], device=self.device)

        for i in range(n_steps):
            img = self.step(img, timesteps[i], timesteps[i + 1], label, guidance_scale=guidance_scale)

        return img

    def sample_reconstruction(
        self,
        arg: Tensor | int,
        n_steps: int = 10,
        guidance_scale: Optional[float] = None
    ) -> Tensor:
        """
        Sample a reconstruction by taking multiple ODE steps.

        Args:
            arg (Tensor, int, Iterable[int]): If ``Tensor``, an image of the same shape will be reconstructed.
                Otherwise, an ``int`` defining the desired width and height is expected.
            n_steps (int): Number of reconstruction steps between :math:`x_0` and :math:`x_1`.
            guidance_scale (float, Optional): Model conditioning strength and its bias towards conditional sample
                synthesis.
        """
        timesteps = torch.linspace(0., 1., n_steps + 1, device=self.device)
        if isinstance(arg, Tensor):
            x0 = torch.randn(*arg.shape, device=self.device)
            x1 = torch.randn([n_steps + 1, *arg.shape], device=self.device)
        elif isinstance(arg, int or Iterable[int]):
            x0 = torch.randn([1, arg, arg], device=self.device)
            x1 = torch.randn([n_steps + 1, 1, arg, arg], device=self.device)
        else:
            raise ValueError(f'Unsupported argument type: {type(arg)}')

        for i in range(n_steps):
            x1[i + 1] = self.step(x0, timesteps[i], timesteps[i + 1], guidance_scale=guidance_scale)

        return x1


class DiffusionNet(CustomModel):
    """
    Base class for diffusion networks. The following implementation is heavily based on
    `Denoising Diffusion Probability Models <https://arxiv.org/abs/2006.11239>`_.

    .. Note::
        The following conventions from the paper are used.
        :math:`x_0` is the sample from target distribution, where :math:`x_T` is noise

    Args:
        model (nn.Module): Diffusion network
        timesteps (int): Number of diffusion steps
        beta_1 (Tensor, float, int):
            Beta schedule. If int or float are provided a constant schedule is instantiated by default.
        device (torch.device, str, Optional): Device used for computation
    """
    def __init__(
        self,
        net: nn.Module,
        timesteps: int,
        beta_1: Union[float, int] = 1e-4,
        beta_t: Union[float, int] = 0.02,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = net.to(device=device)

        self.n_timesteps = timesteps
        if isinstance(beta_1, (float, int)) and not isinstance(beta_t, (float, int)):
            betas = torch.full([timesteps,], beta_1, dtype=torch.float, device=device)
        else:
            betas = torch.linspace(beta_1, beta_t, self.n_timesteps, dtype=torch.float, device=device)
        self.betas = betas
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_rev_alphas_bar = torch.sqrt(1 - self.alphas_bar)

    def register_schedule(self, timesteps: int, betas: Union[float, int]):
        """
        Register a schedule of alphas and betas for diffusion networks.

        Define a *linear* beta schedule for the forward diffusion process. Default values are from `DDPM`_ paper.

        .. _`DDPM`: https://arxiv.org/abs/2006.11239
        """
        if isinstance(betas, (float, int)):
            betas = torch.full([timesteps,], betas, dtype=torch.float, device=self.device)

        assert betas.shape[0] == self.n_timesteps, 'alphas are not defined for all diffusion steps'
        self.alphas = 1. - betas
        self.alphas_bar = torch.cumsum(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_rev_alphas_bar = torch.sqrt(1 - self.alphas_bar)

    def forward(self, img: Tensor, t: int, labels: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            img (Tensor): Input images.
            t (int): Time point from which the forward trajectory is sampled.
            labels (Tensor): Image class labels.
            noise (Tensor, optional): Specific noise from which the image is to be sampled. Defaults to None.

        Shapes:
            - img: (B, C, H, W)
            - t: (B,)
            - labels: (B,)
            - noise: (B, C, H, W)
        """
        if not isinstance(t, int):
            t = torch.randint(0, self.n_timesteps, [img.shape[0],], dtype=torch.long, device=self.device)
        x_t = self.sample_q(img, t, noise=noise)     # sample noise from forward trajectory at random t
        if labels is not None:
            labels = self.get_classifier_free_labels(labels)

        return self.net(x_t, t, labels)

    def sample_q(self, x_0: Tensor, t: Union[Tensor, int], noise: Optional[Tensor] = None) -> Tensor:
        r"""
        Samples an arbitrary :math:`x_t` from forward diffusion trajectory given a sample from the target distribution
        :math:`x_0`.

        The function takes use of the fact that :math:`x_t` can be sampled at random time-step :math:`t`
        using a closed form equation (Equation 4. from `DDPM`):

        .. math::
            q(x_t,x_0) = N(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)

        Args:
            x_0 (Tensor): Original image / sample from target distribution. Shape (B,) C, H, W)
            t (Tensor): Arbitrary time-step of the forward diffusion process.
            noise (Tensor, optional): Noise used to generate :math:`x_t`. Shape equal that of :math:`x_0`. Default: None

        .. note::
            Code inspired by `DDPM`_ and `this <https://www.youtube.com/watch?v=a4Yfz2FxXiY>`_.

        .. _`DDPM`: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py#L108
        """
        if not isinstance(noise, Tensor):
            noise = torch.randn_like(x_0, device=self.device)
        assert noise.shape == x_0.shape

        return self.extract(self.sqrt_alphas_bar, t, x_0) * x_0 + self.extract(self.sqrt_rev_alphas_bar, t, x_0) * noise

    @torch.no_grad()
    def sample_img(
        self,
        img_size: int,
        label: Tensor,
        n_steps: int = 10.,
        guidance_scale: float = 1.,
        x_t: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Reconstruct an image given a label from noise.

        Iteratively applies the following function to reconstruct an image with the desired label or class from
        noise :math:`x_t`:

        .. math::
            \textbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}
            \epsilon_\theta(\textbf{x}_t,t)) + \sigma_t \textbf{z}

        where :math:`\sigma_t=\sqrt{\beta_t}` or :math:`\sigma_t=\tilde{\beta_t}`.

        Args:
            img_size (int): Dimensions of the reconstructed image. If ``x_t`` is provided, then ``size`` is overridden.
            label (Tensor, int): Image label / class. Shape (B,)
            n_steps (int): Number of reconstruction steps.
            guidance_scale (float): Guidance scale :math:`w` for Classifier Free Guidance. Default: 0.
            x_t (Tensor, optional): Noise from which the image is reconstructed. Shape (B, C, H, W)

        .. note::
            This function implements Algorithm 2. from `DDPM <https://arxiv.org/abs/2006.11239>`__.
        """
        if isinstance(label, int):
            label = torch.tensor([label], dtype=torch.long, device=self.device)
        if not isinstance(x_t, Tensor):
            x_t = torch.rand([1, 1, img_size, img_size], device=self.device)

        for t in reversed(range(n_steps)):
            t = torch.tensor([t], dtype=torch.long, device=self.device)
            z = torch.randn_like(x_t, device=self.device) if t > 1 else torch.zeros_like(x_t, device=self.device)
            a = 1 / torch.sqrt(self.extract(self.alphas, t, x_t))
            coeff = (1 - self.extract(self.alphas, t, x_t)) / torch.sqrt(self.extract(self.sqrt_rev_alphas_bar, t, x_t))
            sigma_t = torch.sqrt(self.extract(self.betas, t, x_t))
            x_t = a * x_t - coeff * self.forward_cfg(x_t, t, label, guidance_scale) + sigma_t * z

        return x_t.permute(2, 3, 1).detach().cpu()

    @staticmethod
    def extract(tensor: Tensor, t: Tensor, desired_shape: Tensor.shape) -> Tensor:
        # b, *_ = tensor.shape
        out = tensor.gather(-1, t)
        while out.ndim < desired_shape.ndim:
            out = out.contiguous().unsqueeze(-1)
        return out
