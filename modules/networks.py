import os
import torch
import torch.nn as nn

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Union
from configs import BaseUnetConfig
from modules.modules import SinusoidalEmbedding, Patchify, TransformerEncoderBlock, ResBlock

from torch import Tensor
from torch.nn.functional import mse_loss
from torchvision.transforms import CenterCrop


class Unet(nn.Module):
    r"""
    Basic `UNet <http://arxiv.org/abs/1505.04597>`_ implementation.

    A cache is used to save the cropped tensors for the successive skip-connections to  the ``UpscaleBlocks`` of the
    network decoder.

    Args:
        dims (iter): Hidden layer dimensions, e.g. (64, 128, 256, 512). Does not include the input, output dimensions,
            but does include the latent dimension
        in_dim (int): Number of input channels of the model.
        out_dim (int): Number of output channels of the model.
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
        in_dim: int = 1,
        out_dim: int = 1,
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
        self.dims = dims
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = dims[-1]
        dims_r = dims[::-1]
        self.null_token = n_labels

        # additional label ('null token') for unconditional model training
        self.label_embd = nn.Embedding(n_labels + 1, self.embed_dim)     # (B) -> (B, D)

        # time_emb != timestep_emb
        # TODO: conditional time embedding for Flow(nn.Module)
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py#L648
        # TODO: tensor (B,C,H,W) full of 't': just add to channels
        self.timestep_embd = nn.Sequential(
            # is basically a sinusoidal_embd with an activation function and mapping to higher latent dim
            # SinusoidalEmbedding(1, dim),
            # TODO: # Nxdim     timesteps: vec of N indices, 1 per batch el. may be fractional
            nn.Linear(self.in_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        self.in_layer = ResBlock(in_dim, dims[0])
        self.out_layer = nn.Conv2d(dims[0], out_dim, kernel_size=1, stride=1)
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.midcoder = nn.Sequential(
            ResBlock(dims[-1], dims[-1]),
            ResBlock(dims[-1], dims[-1]),
        )
        for i in range(len(dims) - 1):
            self.encoder.append(ResBlock(dims[i], dims[i + 1], do_upscale=False))
            self.decoder.append(ResBlock(dims_r[i], dims_r[i + 1], do_upscale=True))

        self.to(device)
        del dims_r

        # OLD VERSION. CODE self.en AND self.de AS nn.Sequential()
        # for i in range(len(dims)-1):
        #     self.encoder.append(DownscaleBlock(dims[i], dims[i + 1]))
        #     if i == len(dims)-1:
        #         self.decoder.append(UpscaleBlock(dims[1], dims[0], False))
        #     else:
        #         self.decoder.append(UpscaleBlock(dims_r[i], dims_r[i + 1]))

    @staticmethod
    def crop(x: Tensor, desired: Tensor) -> Tensor:
        """
        Center crop the input to fit for the concatenation.

        Assumes that the height and width of `x` is bigger than that of `desired`.

        Args:
            x (Tensor): Input image from the adjacent DownscaleBlock cache.
            desired (Tensor): Tensor with the desired dimensions.

        Returns:
            Tensor: Cropped input image.

        Shape:
            - x: :math:`(..., H_1, W_1)`
            - desired: :math:`(..., H_2, W_2)`
        """
        equal_dims = x.dim() == desired.dim()
        crop = CenterCrop(desired.shape[:2])

        if equal_dims:
            h1, w1 = x.shape[-2:]
            h2, w2 = desired.shape[-2:]
            dw = int((w1 - w2) / 2)
            dh = int((h1 - h2) / 2)

            return x[..., dh:h1-dh, dw:w1-dw]
        else:
            return crop(x)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        labels: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x (Tensor): Input image with shape.
            t (Tensor): Time-steps.
            labels (Tensor, Optional): Labels for each image. Can be passed only when the model is conditional.

        Shapes:
            - x: :math:`(B, C, H, W)`
            - t: :math:`(B, 1)\subset[0,1]`
            - labels: :math:`(B, 1)`
        """
        x = self.in_layer(x)

        # 1. Downscale: downscale blocks and save intermediate skip connection tensors to cache
        for i, downscale_block in enumerate(self.encoder):
            x = downscale_block(x)
            self.cache[i] = x.clone().detach()

        # 2. Midcoder
        if labels is not None:
            embd = self.label_embd(labels).squeeze()
            while embd.ndim < x.ndim:
                embd.unsqueeze_(-1)
            x += embd   # += (B, D, 1, 1)

        # temp = self.timestep_embd(t).contiguous().reshape(x.shape) # TODO: try doing that instead the bottom
        # temp = self.timestep_embd(t)[..., None, None]
        timestep_embd = torch.cat(
            [torch.full([1, *x.shape[1:]], val.item(), device=self.device) for val in t]
        )
        x += timestep_embd
        x = self.midcoder(x)

        # 3. Upscale: upscale blocks and add appropriate tensors from cache
        for upscale_block, res in zip(self.decoder, list(self.cache.values())[::-1]):
            res = self.crop(res, x)
            x = upscale_block(x, res)

        self.cache = {}
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
        depth_encoder: int = 5,
        patch_size: int = 16,
        embed_dim: int = 256,
        mlp_dim: int = 256,
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
        self.patch_dim = self.patch_size ** 2     # channel size C is omitted as the input is grayscaled, i.e., C=1
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.pos_embd = SinusoidalEmbedding(
            self.embed_dim, max_len=self.n_patches, device=self.device
        )
        # additional token for CFG unconditional model training
        self.label_embd = nn.Embedding(self.n_labels + 1, self.embed_dim, device=self.device)
        self.patch_embd = nn.Sequential(
            Patchify(self.patch_size),  # we flatten the patches and map to ...
            nn.Linear(self.patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.encoder = nn.ModuleList(
            TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                mlp_dim=self.mlp_dim,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                device=self.device
            )
            for _ in range(depth_encoder)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.patch_dim),
        )

        self.to(self.device)

    # def _set_config(self, config: object) -> None:
    #     """
    #     Set network attributes to those found in ``config``.
    #     """
    #     if isinstance(config, object):
    #         [self.__setattr__(k, v) for k, v in config.__dict__.items() if k[0] != '_']
    #     else:
    #         raise TypeError(f'Config must be a dict, but got {type(config)}')

    def forward(
        self,
        img: Tensor,
        t: Tensor,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Pass a batch of images and mask to Vision Transformer.

        Args:
            img (Tensor): Input image to be passed through
            t (Tensor): Time-steps.
            labels (Tensor, optional): Labels for each image. Used when the model is conditional.
            mask (Tensor, optional): Padding mask.
        """
        original_shape = img.shape
        # embeddings
        img += self.add_timestep_embd(img, t)
        img = self.patch_embd(img)
        img = self.pos_embd(img)    # TODO

        if labels is not None:
            img += self.label_embd(labels)
        # forward pass
        for block in self.encoder:
            img = block(img, mask=mask)
        img = self.out_proj(img)    # project back to image shape

        return img.view(original_shape)
        # StdConv
        # GroupNorm
        # relu
        # max_pool

    def add_timestep_embd(self, img: Tensor, t: Tensor) -> Tensor:
        """
        Adds a time-step-embedding to the input image.

        ``timestep_embedding`` is a ``Tensor`` with the same shape as ``img`` filled with a random number sampled
        from [0,1].
        """
        if img.ndim == 4:
            timestep_embd = torch.cat(
                [torch.full([1, *img.shape[1:]], fill_value=val.item(), device=self.device) for val in t]
            )
        elif img.ndim == 3 or t.ndim == 1:
            timestep_embd = torch.full(img.shape, fill_value=t.item(), device=self.device)
        else:
            raise ValueError(f'Timestep embedding error. Unsupported img shape {img.shape}.')

        # assert timestep_embd.shape == img.shape, f'Shape of timestep_embd ({timestep_embd.shape}) does not match img.'
        return img + timestep_embd

    def patchify(self, img: Tensor, patch_size: Optional[int] = None) -> Tensor:
        """
        Patchify an input image for the Vision Transformer to process.

        The output shape is inferred automatically. If the input is a 3D or a non-batched ``Tensor`` the output
        will be recast to :math:`(1, N, CP^2)`, where :math:`N=HxW/P^2` is the number of patches.

        Args:
            img (Tensor): Input image to be made into patches.
            patch_size (int): Patch size (P).

        Shapes:
            - img (Tensor): :math:`B, (C, H, W)`
            - patches (Tensor): :math:`B, (N, CP^2)`

        Return:
            Patches.
        """
        assert img.ndim in {3, 4}, f'Input image must be 3D or 4D but is {img.ndim}'
        patch_size = patch_size if patch_size else self.patch_size

        c, h, w = img.shape[-3:]
        n = int(h*w/patch_size**2)
        img = img.view(-1, n, c*patch_size**2)

        return img


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_cfg(
            self,
            x_t: Tensor,
            t: Tensor,
            labels: Optional[Tensor] = None,
            guidance_scale: Optional[float | Tensor] = None
    ) -> Tensor:
        r"""
        Return Classifier-Free score. Used at *inference time only* for qualitative analysis.

        .. math::
            \tilde\epsilon_\theta = w\epsilon_\theta(x,c) + (1-w) \epsilon_\theta(x,\emptyset)

        where :math:`\emptyset` is the null token for unconditional training.

        Shape:
            - Output: Tensor of shape (B, C, H, W)
        """
        if labels is None:
            guidance_scale = 0
            null_labels = None
        else:
            guidance_scale = guidance_scale if guidance_scale else self.guidance_scale
            null_labels = torch.full_like(labels, self.net.null_token, dtype=torch.int, device=self.device)  # (B, 1)
        return (1 - guidance_scale) * self.net(x_t, t, null_labels) + guidance_scale * self.net(x_t, t, labels)

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
        guidance_scale (float): Guidance scale. Determines the strength of model conditionality. The higher the value, the higher
            the model conditioning.
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
        device: Optional[torch.device | str] = None,
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

    def forward(self, x_t: Tensor, t: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if labels is not None:
            labels = self.get_classifier_free_labels(labels)
        return self.net(x_t, t, labels)

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
        rate = rate if rate else self.unconditional_rate
        p_uncond = torch.rand_like(labels.to(torch.float), device=self.device)
        labels = torch.where(p_uncond < rate, 4., labels)  # 4. as the null token

        return labels.to(torch.int) # labels.long()

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, guidance_scale: Optional[float] = None) -> Tensor:
        r"""
        Perform an ODE solving step to reconstruct the image step-by-step. Uses the midpoint solver given by:

        .. math::
            y_{n+1} = y_n + h*f[t_n + h/2, y_n + h/2*f(t_n,y_n)]

        See `Flow Matching Guide and Code <https://arxiv.org/abs/2412.06264>`__ for more information.
        """
        if x_t.ndim == 3:
            t_start = t_start.view(1, 1).expand(x_t.shape[0], -1)
        elif x_t.ndim == 4:
            t_start = t_start.view(1, 1).expand(x_t.shape[1], -1)
        # TODO:
        # temp = t_start.view(1,1)
        # t_start = t_start[:, None]
        delta_t = t_end - t_start

        if guidance_scale:
            return x_t + delta_t * self.forward_cfg(
                x_t + self.forward_cfg(x_t, t_start, guidance_scale=guidance_scale) * delta_t / 2, t_start + delta_t / 2,
                guidance_scale=guidance_scale
            )
        else:
            return x_t + delta_t * self(x_t + self(x_t, t_start)*delta_t/2, t_start + delta_t/2)

    def sample_img(
        self,
        input_: Tensor | int,
        n_steps: int = 50,
        return_reconstruction: bool = False,
        guidance_scale: Optional[float] = None,
    ):
        timesteps = torch.linspace(0., 1., n_steps + 1, device=self.device)
        if isinstance(input_, Tensor):
            x0 = torch.randn_like(input_, device=self.device)
            # reconstruction = torch.randn([n_steps + 1, *input.shape[1:]], device=self.device, requires_grad=False)
        elif isinstance(input_, int):
            x0 = torch.randn([1, input_, input_], device=self.device)
            # reconstruction = torch.randn([n_steps + 1, 1, input, input], device=self.device, requires_grad=False)
        else:
            raise TypeError(f'Unexpected type: {type(input_)}')

        for i in range(n_steps):
            input_ = self.step(x0, timesteps[i], timesteps[i + 1], guidance_scale=guidance_scale)

        if return_reconstruction:
            reconstruction = torch.randn([n_steps + 1, *x0.shape], device=self.device, requires_grad=False)
            with torch.no_grad():
                for i in range(n_steps):
                    reconstruction[i + 1] = self.step(x0, timesteps[i], timesteps[i + 1], guidance_scale=guidance_scale)

        if return_reconstruction:
            return input_, reconstruction
        else:
            return input_

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
            betas = torch.full((timesteps,), beta_1, dtype=torch.float, device=device)
        else:
            betas = torch.linspace(beta_1, beta_t, self.n_timesteps, dtype=torch.float, device=device)
        self.betas = betas
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_rev_alphas_bar = torch.sqrt(1 - self.alphas_bar)

    def register_schedule(self, timesteps, betas: Union[Tensor, float, int]):
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

    def forward(
        self,
        img: Tensor,
        t: Union[Tensor, int] = None,
        labels: Optional[Tensor] = None,
        noise: Optional[Tensor] = None
    ) -> Tensor:
        if not isinstance(t, (Tensor, int)):
            t = torch.randint(0, self.n_timesteps, [img.shape[0],], dtype=torch.long, device=self.device)
        x_t = self.sample_q(img, t, noise=noise)     # sample noise from forward trajectory at random t

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
            x_0 (Tensor): Original image / sample from target distribution. Shape :math:`(B,) C, H, W)`
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
        size: int,
        label: Union[Tensor, int],
        guidance_scale: float = 0.,
        steps: Optional[int] = None,
        x_t: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Reconstruct an image given a label from noise.

        Iteratively applies the following function to reconstruct an image with the desired label or class from
        noise :math:`x_t`:

        .. math::
            \begin{aligned}
                &\rule{110mm}{0.5pt}                                                                                  \\
                &\textbf{for}\  t=T,\ldots,1 \textbf{do}
                &\hspace{5mm} \textbf{z}\sim N(\textbf{0},\textbf{I})\ \textbf{if} \ t>1 \textbf{else}\ \textbf{z}=
                \textbf{0} \\

            \end{aligned}

        .. math::
            \textbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}
            \epsilon_\theta(\textbf{x}_t,t)) + \sigma_t \textbf{z}

        where :math:`\sigma_t=\sqrt{\beta_t}` or :math:`\sigma_t=\tilde{\beta_t}`.

        Args:
            size (int): Dimensions of the reconstructed image. If ``x_t`` is provided, then ``size`` is overridden.
            label (Tensor, int): Image label / class. Shape :math:`(B,)`
            guidance_scale (float): Guidance scale :math:`w` for Classifier Free Guidance. Default: 0.
            x_t (Tensor, optional): Noise from which the image is reconstructed. Shape :math:`(B,) C, H, W`
            steps (int, optional): Number of reconstruction steps.

        .. note::
            This function implements Algorithm 2. from `DDPM <https://arxiv.org/abs/2006.11239>`__.
        """
        if isinstance(label, int):
            label = torch.Tensor([label], device=self.device).long()
        if not isinstance(steps, int):
            steps = self.n_timesteps
        if not isinstance(x_t, Tensor):
            x_t = torch.rand([1, size, size], device=self.device)

        for t in reversed(range(steps)):
            t = torch.Tensor([t], device=self.device).long()
            z = torch.randn_like(x_t, device=self.device) if t > 1 else torch.zeros_like(x_t, device=self.device)
            a = 1 / torch.sqrt(self.extract(self.alphas, t, x_t))
            coeff = (1 - self.extract(self.alphas, t, x_t)) / torch.sqrt(self.extract(self.sqrt_rev_alphas_bar, t, x_t))
            sigma_t = torch.sqrt(self.extract(self.betas, t, x_t))
            x_prev = a * x_t - coeff * self.forward_cfg(x_t, t, label, guidance_scale) + sigma_t * z

        return x_prev.permute(1, 2, 0).detach().cpu()

    @staticmethod
    def extract(tensor: Tensor, t: Tensor, desired_shape: Tensor.shape) -> Tensor:
        # b, *_ = tensor.shape
        out = tensor.gather(-1, t)
        while out.ndim < desired_shape.ndim:
            out = out.contiguous().unsqueeze(-1)
        return out
