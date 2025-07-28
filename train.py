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
from modules.unet_fm import UNet
from modules.vit_fm import ViTFlowMatchingConditional
from utils import compare_imgs, EarlyStopper

from torch import Tensor
from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.transforms.v2 import Compose, ConvertImageDtype, Normalize, RandomRotation, GaussianBlur
load_dotenv()
from torchvision.utils import save_image


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
    val_losses = []

    for i, batch in enumerate(val_tqdm):
        x1, labels = batch
        x1, labels = x1.to(model.device), labels.to(model.device)  # x1 = (B, D)
        x0 = torch.randn_like(x1, device=model.device)

        if mode == 'flow-matching':
            # t: random [0,1] -> (B, 1, H, W) / as additional channel
            t = torch.rand((x1.shape[0], 1, 1, 1), device=model.device)  # (B, 1) -> embd -> +
            xt = (1 - t) * x0 + t * x1  # (B, D)
            delta = x1 - x0
            loss = mse_loss(model(xt, t, labels), delta)
        elif mode == 'diffusion':
            loss = mse_loss(model(x1, labels=labels, noise=x0), x1)
        else:
            raise ValueError('Invalid validation mode.')

        val_tqdm.set_postfix(validation_loss=f'{loss.item():.8f}')
        val_losses.append(loss.item())

    mean_val_loss = torch.mean(torch.tensor(val_losses))
    # compare first 8 samples of the last batch
    #if epoch % 10 == 0:
    #    for n_steps in {1, 10, 100}:
    #        print(f'Sampling {n_steps} steps...')
    #        img = model.sample_img(x1, n_steps=n_steps)
    #        fig = compare_imgs(x1[:8].permute(0, 2, 3, 1), img[:8].permute(0, 2, 3, 1), False, labels[:8])
    #        mlflow.log_figure(fig, f'{epoch}[{n_steps}]_validation_imgs.pdf')
    #        plt.close(fig)


    img_size = 128
    n_steps = 100
    guidance_scale = 3.
    label = 1
    model_out = model.sample_single_image(
        img_size=img_size,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        label=label
    )
    # De-normalize to [0,1]
    img = model_out[0] * 0.5 + 0.5
    save_image(img, f"./data/temp/gen_epoch_{epoch}_{img_size}x{img_size}_{n_steps}steps_{guidance_scale}cfg_{label}.png")


    # TODO
    # with torch.no_grad():
    #     _, reconstruction = model.sample_img(x1[:4], return_reconstruction=True)
    #     plot_reconstructed_img(reconstruction)

    return mean_val_loss


def train(
        net: torch.nn.Module,
        img_size: int,
        epochs: int = 1,
        batch_size: int = 16,
        lr: float = 5e-4,
        run_name: str = 'default',
        path: str | os.PathLike = './checkpoints',
        mode: str = 'flow-matching',
        augmented: bool = True,
        patience: Optional[int] = None,
        config: Optional[object] = None,
        mlflow_config: Optional[object] = None,
        device: Optional[torch.device | str] = None
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

    device = torch.device(device if device is not None else "cpu")
    if mode == 'flow-matching':
        model = FlowMatchingNet(net, lr, device=device)
    elif mode == 'diffusion':
        model = DiffusionNet(net, 1000, lr, device=device)
    else:
        raise ValueError('Invalid model mode.')
    #if torch.cuda.device_count() > 1:
    #    print(f'{torch.cuda.device_count()} GPUs available.')
    #    model = DistributedDataParallel(model)
    if config is not None:
        model.set_config(config)

    if patience:
        early_stopper = EarlyStopper(experiment_path=experiment_path, patience=patience)

    # data import and augmentation
    transform = Compose([
        ConvertImageDtype(torch.float),
        # Note by Jonathan: If you use Normalize, then make sure it's not 0 and 1, since this expects the _actual_ mean and actual _std_.
        Normalize([0.5], [0.5], inplace=True),
    ])
    training_data, validation_data = random_split(
        MRIDataset(f'./data/preprocessed_{img_size}/annotations.csv', transform=transform),
        [0.9, 0.1],
        torch.manual_seed(42)   # for reproducible validation set
    )
    if augmented:
        # Note by Jonathan: To me this seems like a strange order. You have your training data with the transform,
        # which includes Normalization, but then you blur the normalized image. I would augment before, and then normalize.
        gaussian_blurred = MRIDataset(
            f'./data/preprocessed_{img_size}/annotations.csv',
            transform=Compose([transform, GaussianBlur(kernel_size=5, sigma=3.)])  # Note by Jonathan: Maybe randomize the parameters?
        )
        rotated = MRIDataset(
            f'./data/preprocessed_{img_size}/annotations.csv',
            transform=Compose([transform, RandomRotation(90)])
        )
        training_data = ConcatDataset([training_data, gaussian_blurred, rotated])
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=False)
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
                desc=f'Training Epoch [{epoch + 1}/{epochs}]',
                ncols=100
            )
            train_loss_list = []

            # iterate over the dataset
            for i, batch in enumerate(train_tqdm):
                x1, labels = batch
                x1, labels = x1.to(device), labels.to(device)    # x1 = (B, D)
                x0 = torch.randn_like(x1, device=device)

                if mode == 'flow-matching':
                    t = torch.rand((x1.shape[0], 1, 1, 1), device=device)     # (B, 1) -> embd -> +
                    xt = (1 - t) * x0 + t * x1  # (B, D)
                    delta = x1 - x0
                    loss = model.backprop(model(xt, t, labels), delta, do_return=True)
                elif mode == 'diffusion':
                    loss = model.backprop(model(x1, labels=labels, noise=x0), x1, do_return=True)

                train_tqdm.set_postfix(training_loss=f'{loss.item():.8f}')
                train_loss_list.append(loss.item())

            # logging
            mean_train_loss = torch.mean(torch.tensor(train_loss_list))
            mean_val_loss = validate(model, epoch, batch_size, val_loader, mode)
            mlflow.log_metrics(
                {
                    'mean_train_loss': mean_train_loss.item(),
                    'mean_val_loss': mean_val_loss.item()
                }, step=epoch
            )
            if epoch % ceil(0.1 * epochs) == 0:
                mlflow.pytorch.log_model(
                    pytorch_model=model, step=epoch, model_type=model.net.__class__.__name__,
                )
                mlflow.pytorch.log_state_dict(model.state_dict(), f'{model.net.__class__.__name__}')
            elif epoch == epochs:
                mlflow.pytorch.log_model(
                    pytorch_model=model, step=epoch, name=f'{model.net.__class__.__name__}_final',
                    model_type=model.net.__class__.__name__
                )
                mlflow.pytorch.log_state_dict(model.state_dict(), f'{model.net.__class__.__name__}_final')
                model.save_state_dict(experiment_path)
                print('Training done.')

            # early stopping
            if patience:
                do_stop = early_stopper.update(model, mean_val_loss)
                if do_stop:
                    print(f'Stopping early at epoch {epoch}.')
                    break

    except KeyboardInterrupt:
        model.save_checkpoint(experiment_path)
        mlflow.pytorch.log_model(
            pytorch_model=model, step=epoch,
            name=f'{model.net.__class__.__name__}_checkpoint_{epoch}',
            model_type=model.net.__class__.__name__
        )
        mlflow.pytorch.log_state_dict(model.state_dict(), f'{model.net.__class__.__name__}_checkpoint_{epoch}')
        exit('Caught KeyboardInterrupt. Saving checkpoint...')


# TODO
def train_diffusion():
    pass


if __name__ == '__main__':
    use_unet = True
    vit_config = MyVit()
    device = torch.device("mps")

    if not use_unet:
        train(
            ViTFlowMatchingConditional(
                img_size=128,
                patch_size=vit_config.patch_size,
                in_chans=1,
                num_classes=vit_config.n_labels + 1,  # 4 + 1
                embed_dim=vit_config.embed_dim,
                depth=vit_config.depth_encoder,
                num_heads=vit_config.n_heads,
                dropout=vit_config.dropout_rate,
            ),
            device=device,
            img_size=128,
            run_name='vit',
            epochs=50,
            augmented=False,
            patience=0,
            config=vit_config,
            mlflow_config=None
        )
    else:
        train(
            UNet(
                channels=[16, 32, 64, 128],
                num_residual_layers=3,
                t_embed_dim=64,
                y_embed_dim=64
            ),
            device=device,
            img_size=128,
            run_name='unet',
            epochs=50,
            augmented=False,
            patience=0,
            config=None,
            mlflow_config=None
        )


    #train(
    #    VisionTransformer(
    #        img_size=128,
    #        depth_encoder=vit_config.depth_encoder,
    #        n_heads=vit_config.n_heads,
    #        mlp_dim=vit_config.mlp_dim,
    #        patch_size=vit_config.patch_size,
    #        embed_dim=vit_config.embed_dim,
    #        device=device
    #    ),
    #    device=device,
    #    img_size=128,
    #    run_name='vit',
    #    epochs=50,
    #    augmented=False,
    #    patience=8,
    #    config=vit_config,
    #    mlflow_config=None
    #)
    # train_flow(
    #     Unet(128, 64, 32),
    #     128,
    #     run_name='unet',
    #     epochs=200,
    #     patience=10,
    #     augmented=False,
    #     mlflow_config=MlflowConfig(),
    # )
