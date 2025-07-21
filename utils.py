import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Iterable, Optional
from torch import Tensor
from torch.nn import Module


def change_filename_if_taken(path: str | os.PathLike, file_name: str | os.PathLike) -> str | os.PathLike:
    """
    Check if a given filename on a given path is taken. If it is, recursively add an iterator.
    """
    file_path = os.path.join(path, file_name)
    if os.path.isfile(file_path):
        addon = 0
        while os.path.isfile(file_path):
            name, format_ = file_name.split('.')
            name += f'_{addon}'
            file_name = f'{name}.{format_}'
            file_path = os.path.join(path, file_name)
            addon += 1
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

    # fig, axes = plt.subplots(b, 2, figsize=(4, 2*b), tight_layout=True, sharex=True, sharey=True)
    # axes[0, 0].set_title('Original')
    # axes[0, 1].set_title('Reconstructed')
    # for i, ax in enumerate(axes):
    #     ax[0].imshow(original[i])
    #     ax[1].imshow(reconstructed[i])
    #     if labels is not None:
    #         ax[0].set_ylabel(f'Label: {labels[i].item()}')

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


def plot_reconstructed_img(
        img: Tensor,
        path: str | os.PathLike = './assets',
        file_name: str | os.PathLike = 'reconstructed_img.pdf',
        do_save: bool = False,
    ) -> None:
    """
    Plot reconstructed image. The

    Args:
        img (Tensor): Series of reconstruction steps from :math:`t_start` to :math:`t_end`.
            Has shape :math:`(T, B, H, W, 1)` or `(T, H, W, 1)`, where `T` is the number of time-steps.
        path (str | os.PathLike): Location to save the plot.
        file_name (str | os.PathLike): Name of plot file.
        do_save (bool): Whether to save the plot.
    """
    img = img.detach()
    time_steps = img.shape[0] - 1
    if img.ndim == 5:       # (T, B, H, W, C)
        b = img.shape[1]
    elif img.ndim == 4:     # (T, H, W, C)
        b = 1
    else:
        raise ValueError(f'Unexpected image shape: {img.shape}')

    indices = np.linspace(0., 1., time_steps + 1)
    fig, axes = plt.subplots(b, ncols=time_steps + 1, figsize=(20, 4*b), tight_layout=True, sharex=True, sharey=True)

    axes[0, 0].set_title(f't = 0.00', fontsize=10)
    if b > 1:
        for j, ax in enumerate(axes):
            ax[j].imshow(img[0, j])  # original imgs
    else:
        axes[0, 0].imshow(img[0])  # original img

    for i, row in enumerate(axes):
        row[i + 1].set_title(f't = {indices[i + 1]:.2f}', fontsize=10)
        for j, col in enumerate(row):
            col.imshow(img[i + 1, j])

    if do_save:
        file_path = change_filename_if_taken(path, file_name)
        plt.savefig(file_path)
    plt.show()


# TODO
def plot_variable_guidance_scale(
        model: Module,
        guidance_scale: int | Iterable,
        labels: Optional[Tensor] = None,
        path: str | os.PathLike = None,
        do_save: bool = False
    ) -> None:

    if not isinstance(guidance_scale, (float, Iterable[float])):
        raise ValueError(f'Invalid guidance_scale value: {guidance_scale}')
    if not isinstance(labels, (int, Iterable[int])):
        raise ValueError(f'Invalid labels value: {labels}')

    if isinstance(guidance_scale, Iterable) or isinstance(guidance_scale, Iterable):
        fig, axes = plt.subplots(len(labels), len(guidance_scale))
        for i, val in enumerate(guidance_scale):
            axes[i].imshow(val)
    else:
        pass

    # noise = torch.randn()
    # output = model.forward_cfg(guidance_scale=guidance_scale)

    if do_save:
        plt.savefig(path + '/variable_guidance_scale.png')
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
