import os
import torch
import mlflow
import matplotlib.pyplot as plt

from sys import exit
from tqdm import tqdm
from math import ceil
from dotenv import load_dotenv
from typing import Optional
from configs import *
from data.dataset import MRIDataset
from modules.networks import Unet, VisionTransformer, FlowMatchingNet, DiffusionNet
from utils import compare_imgs, EarlyStopper

from torch import Tensor
from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.transforms.v2 import Compose, ConvertImageDtype, Normalize, RandomRotation, GaussianBlur
load_dotenv()


@torch.no_grad()
def validate(
        model: FlowMatchingNet,
        epoch: int,
        batch_size: int,
        val_loader: DataLoader,
        mode: str,
        # loss_fnc: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
    """
    Validate the model on the validation set.
    """
    model.eval()
    val_tqdm = tqdm(val_loader, total=len(val_loader), desc='Validation')
    val_losses = torch.zeros(len(val_loader), dtype=torch.float32, device=model.device, requires_grad=False)

    for i, batch in enumerate(val_tqdm):
        x1, labels = batch
        x1, labels = x1.to(model.device), labels.to(model.device)  # x1 = (B, D)
        x0 = torch.randn_like(x1, device=model.device)

        if mode == 'flow-matching':
            # t: random [0,1] -> (B, 1, H, W) / as additional channel
            t = torch.rand(batch_size, 1, 1, 1, device=model.device)  # (B, 1) -> embd -> +
            xt = (1 - t) * x0 + t * x1  # (B, D)
            delta = x1 - x0
            loss = mse_loss(model(xt, t, labels), delta)
        elif mode == 'diffusion':
            loss = mse_loss(model(x1, labels=labels, noise=x0), x1)
        else:
            raise ValueError('Invalid validation mode.')

        val_tqdm.set_postfix(validation_loss=f'{loss.item():.8f}')
        val_losses[i] = loss.item()

    mean_val_loss = torch.mean(val_losses)
    # compare first 8 samples of the last batch
    if epoch % 10 == 0:
        for n_steps in {1, 10, 100}:
            print(f'Sampling {n_steps} steps...')
            img = model.sample_img(x1, n_steps=n_steps)
            fig = compare_imgs(x1[:8].permute(0, 2, 3, 1), img[:8].permute(0, 2, 3, 1), False, labels[:8])
            mlflow.log_figure(fig, f'{epoch}[{n_steps}]_validation_imgs.pdf')
            plt.close(fig)

    # TODO
    # with torch.no_grad():
    #     _, reconstruction = model.sample_img(x1[:4], return_reconstruction=True)
    #     plot_reconstructed_img(reconstruction)

    return mean_val_loss


def train(
        net: torch.nn.Module,
        img_size: int,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 5e-4,
        run_name: str = 'default',
        path: str | os.PathLike = './checkpoints',
        mode: str = 'flow-matching',
        augmented: bool = True,
        patience: Optional[int] = None,
        config: Optional[object] = None,
        mlflow_config: Optional[object] = None,
    ) -> None:
    """
    Train a Flow-Matching network on MRI data.

    Args:
        run_name (str): Name of experiment to identify  the run.
        net (nn.Module): FlowMatching network.
        img_size (int): Image width on which to train on.
        epochs (int): Number of training epochs. One epoch is equivalent to a full iteration over the dataset.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        path (str, os.PathLike): Path to directory where checkpoints and weights will be saved.
        augmented (bool). Whether to augment the training set with additional samples. Adds rotated and blurred samples.
        patience (int, optional): Patience for early stopping.
        config (object, optional): Configuration of model parameters
        mlflow_config (object, optional): Config class of Mlflow experiment.
        mode (str): Type of model to be trained. Either `flow-matching` or `diffusion`.
    """
    run_name = mode + '-' + run_name + f'_{img_size}'
    experiment_path = os.path.join(path, run_name)
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mode == 'flow-matching':
        model = FlowMatchingNet(net, lr, device=device)
    elif mode == 'diffusion':
        model = DiffusionNet(net, 1000, lr, device=device)
    else:
        raise ValueError('Invalid model mode.')
    if torch.cuda.device_count() > 1:
        print(f'{torch.cuda.device_count()} GPUs available.')
        model = DistributedDataParallel(model)
    if config is not None:
        model.set_config(config)

    if patience:
        early_stopper = EarlyStopper(experiment_path=experiment_path, patience=patience)

    # data import and augmentation
    transform = Compose([
        ConvertImageDtype(torch.float),
        Normalize([0], [1], inplace=True),  # TODO: Tensor contains vals > 1
    ])
    training_data, validation_data = random_split(
        MRIDataset(f'./data/preprocessed_{img_size}/annotations.csv', transform=transform),
        [0.8, 0.2],
        torch.manual_seed(42)   # for reproducible validation set
    )
    if augmented:
        gaussian_blurred = MRIDataset(
            f'./data/preprocessed_{img_size}/annotations.csv',
            transform=Compose([transform, GaussianBlur(kernel_size=5, sigma=3.)])
        )
        rotated = MRIDataset(
            f'./data/preprocessed_{img_size}/annotations.csv',
            transform=Compose([transform, RandomRotation(90)])
        )
        training_data = ConcatDataset([training_data, gaussian_blurred, rotated])
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, drop_last=True)

    # set up mlflow tracking
    if mlflow_config is not None:
        # mlflow.set_tracking_uri(mlflow_config.mlflow_server_uri)
        mlflow.set_experiment(experiment_name=mlflow_config.mlflow_experiment_name)
        mlflow.set_tags({
            'mlflow.runName': run_name,
            'mlflow.user': os.environ['MLFLOW_USER'],
            'model_type': mode
        })
    mlflow.log_params({
        'model_class': model.net.__class__.__name__,
        'model_config': config,
        'img_size': img_size,
        'batch_size': batch_size,
        'learning_rate': lr,
        'patience': patience,
        'epochs': epochs,
        'augmented': augmented,
        'device': model.device,
    })
    print(f'Initialization done. Training on {device}...')

    # training loop
    try:
        model.train()
        for epoch in range(epochs):
            train_tqdm = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f'Training Epoch [{epoch}/{epochs - 1}]',
                ncols=100
            )
            train_loss = torch.randn(len(train_loader), dtype=torch.float32, device=device, requires_grad=False)

            # iterate over the dataset
            for i, batch in enumerate(train_tqdm):
                x1, labels = batch
                x1, labels = x1.to(device), labels.to(device)    # x1 = (B, D)
                x0 = torch.randn_like(x1, device=device)

                if mode == 'flow-matching':
                    t = torch.rand(batch_size, 1, 1, 1, device=device)     # (B, 1) -> embd -> +
                    xt = (1 - t) * x0 + t * x1  # (B, D)
                    delta = x1 - x0
                    loss = model.backprop(model(xt, t, labels, mask), delta, do_return=True)
                elif mode == 'diffusion':
                    loss = model.backprop(model(x1, labels=labels, noise=x0), x1, do_return=True)

                train_tqdm.set_postfix(training_loss=f'{loss.item():.8f}')
                train_loss[i] = loss.item()

            # logging
            mean_train_loss = torch.mean(train_loss)
            mean_val_loss = validate(model, epoch, batch_size, val_loader, mode)
            mlflow.log_metrics(
                {
                    'mean_train_loss': mean_train_loss.item(),
                    'mean_val_loss': mean_val_loss.item()
                }, step=epoch
            )
            if epoch % ceil(0.1 * epochs) == 0:
                mlflow.pytorch.log_model(
                    pytorch_model=flow, step=epoch, model_type=flow.net.__class__.__name__,
                )
                mlflow.pytorch.log_state_dict(flow.state_dict(), f'{flow.net.__class__.__name__}')
            elif epoch == epochs:
                mlflow.pytorch.log_model(
                    pytorch_model=flow, step=epoch, name=f'{flow.net.__class__.__name__}_final',
                    model_type=flow.net.__class__.__name__
                )
                mlflow.pytorch.log_state_dict(flow.state_dict(), f'{flow.net.__class__.__name__}_final')
                flow.save_state_dict(experiment_path)
                print('Training done.')

            # early stopping
            if patience:
                do_stop = early_stopper.update(model, mean_val_loss)
                if do_stop:
                    print(f'Stopping early at epoch {epoch}.')
                    break

    except KeyboardInterrupt:
        flow.save_checkpoint(experiment_path)
        mlflow.pytorch.log_model(
            pytorch_model=flow, step=epoch,
            name=f'{flow.net.__class__.__name__}_checkpoint_{epoch}',
            model_type=flow.net.__class__.__name__
        )
        mlflow.pytorch.log_state_dict(flow.state_dict(), f'{flow.net.__class__.__name__}_checkpoint_{epoch}')
        exit('Caught KeyboardInterrupt. Saving checkpoint...')


# TODO
def train_diffusion():
    pass


if __name__ == '__main__':
    train_flow(
        VisionTransformer(
            128,
            depth_encoder=6,
            n_heads=16,
            mlp_dim=768,
            patch_size=16,
            embed_dim=768,
        ),
        128,
        run_name='vit',
        epochs=50,
        augmented=True,
        patience=8,
        config=BaseVitConfig(),
        mlflow_config=MlflowConfig
    )
    # train_flow(
    #     Unet(128, 64, 32),
    #     128,
    #     run_name='unet',
    #     epochs=200,
    #     patience=10,
    #     augmented=False,
    #     mlflow_config=MlflowConfig(),
    # )
