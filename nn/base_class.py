import os
import torch
import torch.nn as nn

from abc import abstractmethod
from typing import Optional, Callable, Tuple
from torch import Tensor
from torch.nn.functional import mse_loss


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_cfg(
        self,
        x_t: Tensor,
        t: Tensor,
        labels: Tensor,
        guidance_scale: float = 1.
    ) -> Tensor:
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
        null_labels = torch.full_like(labels, self.net.null_token, dtype=torch.long, device=self.device)  # (B,)
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
        labels = torch.where(p_uncond < rate, self.net.null_token, labels)  # 4 as the null token

        return labels

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
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
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

    @abstractmethod
    def sample_img(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def project(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get parallel and orthogonal projection between two vectors. Used for Adaptive Projected Guidance.
        """
        dtype = x.dtype
        x, y = x.double(), y.double()
        y = torch.nn.functional.normalize(y, dim=y.dim[1:])
        parallel = (x * y).sum(dim=y.dim[1:], keepdim=True) * y
        orthogonal = x - parallel
        return parallel.to(dtype), orthogonal.to(dtype)
