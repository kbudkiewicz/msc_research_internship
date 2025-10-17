import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from typing import Iterable, Optional, Union
from torch import Tensor
from torch.nn import Module
from mpl_toolkits.axes_grid1 import ImageGrid


def change_filename_if_taken(path: str | os.PathLike, file_name: str | os.PathLike) -> str | os.PathLike:
    """
    Check if a given filename on a given path is taken. If it is, recursively add an iterator.
    """
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        idx = 0
        path, extension = os.path.splitext(file_path)
        while os.path.isfile(file_path):
            file_path = path + f'_{idx}' + extension
            idx += 1
    return file_path


def compare_imgs(
    original: Tensor,
    reconstructed: Tensor,
    show_plot: bool = False,
    labels: Optional[Tensor] = None,
):
    """
    Args:
        original (Tensor): Batch of original images from validation set.
        reconstructed (Tensor): Batch of reconstructed images.
        show_plot (bool): Whether to show the plot.
        labels (Tensor, optional): Batch of ground truth labels.
    """
    original = original.detach()
    reconstructed = reconstructed.detach()
    assert original.shape == reconstructed.shape, 'Shapes mismatch. Make sure all tensors have the same shape.'
    b, *_ = original.shape
    assert b == len(labels), 'Batch size and the number of labels mismatch.'

    fig, axes = plt.subplots(2, b, figsize=(2*b, 4), tight_layout=True, sharex=True, sharey=True)
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    for i, ax in enumerate(axes.T):
        ax[0].imshow(original[i])
        ax[1].imshow(reconstructed[i])
        if labels is not None:
            ax[0].set_title(f'Label: {labels[i].item()}')

    if show_plot:
        plt.show()
    else:
        return fig


def plot_reverse_process(x: Tensor):
    """
    Visualize the reverse process.

    Args:
        x (Tensor): Series of reconstruction steps from :math:`t_start` to :math:`t_end`.
            Has shape :math:`(steps, 1, H, W)`
    """
    width = x.shape[0]
    x = x.detach().cpu().permute(0, 2, 3, 1)
    fig, axes = plt.subplots(1, width, figsize=(width * 2, 4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.)

    for idx, ax in enumerate(axes):
        img = x[idx].numpy().astype(np.float64)
        ax.set_title(idx)
        ax.imshow(img)

    return fig


@torch.no_grad()
def plot_variable_guidance_scale(
    model: Module,
    img_size: int,
    n_steps: int,
    guidance_scale: Iterable[float],
    labels: Iterable[int],
    do_save: bool = False,
    path: Union[str, os.PathLike] = None,
) -> None:
    if not isinstance(labels, Iterable):
        raise ValueError(f'Invalid labels value: {labels}')
    if not isinstance(guidance_scale, Iterable):
        raise ValueError(f'Invalid guidance_scale value: {guidance_scale}')

    model = model.eval()
    fig = plt.figure(figsize=(12, 4))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(labels), len(guidance_scale)), axes_pad=0)

    imgs = []
    for i, label in enumerate(labels):
        # grid.axes_row[i][0].set_ylabel(f'{label=}')
        for j, w in enumerate(guidance_scale):
            grid.axes_row[0][j].set_title(f'{w=}')
            label = torch.tensor([label], dtype=torch.long, requires_grad=False, device=model.device)
            img = model.sample_img(img_size=img_size, label=label, guidance_scale=w, n_steps=n_steps)
            img.clamp_(0, 1)
            img = img.permute(0, 2, 3, 1).cpu().detach()[0]
            imgs.append(img)

    for ax, img in zip(grid, imgs):
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    if do_save:
        plt.savefig(path + '/variable_guidance_scale.pdf')
    else:
        plt.show()


def plot_metrics(data: pd.DataFrame) -> None:
    """
    Plots metrics over the training period.
    """
    # iterate over DataFrame inputs
    for k, v in data.iterrows():
        plt.plot(v, label=k)
        plt.legend()
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


class EarlyStopper(object):
    def __init__(self, experiment_path: str | os.PathLike, patience: int = 10):
        self.experiment_path = experiment_path
        self.patience = patience
        self.counter = 0
        self.current_epoch = 0
        self.lowest_val_loss = None

    def update(self, model, val_loss: float | Tensor):
        """
        Save model parameters if validation has improved. Otherwise, terminate the training loop early.
        """
        self.current_epoch += 1
        if isinstance(val_loss, Tensor):
            val_loss = val_loss.item()

        if self.lowest_val_loss is None or val_loss < self.lowest_val_loss:
            self.lowest_val_loss = val_loss
            model.save_state_dict(self.experiment_path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False
