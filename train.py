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

from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.transforms.v2 import Compose, RandomRotation, GaussianBlur, ToImage, ToDtype
from torchvision.utils import save_image
load_dotenv()


@torch.no_grad()
def validate(
        model: FlowMatchingNet,
        epoch: int,
        val_loader: DataLoader,
        run_name: str,
    ):
    """
    Validate the model on the validation set.
    """
    model.eval()
    mode = run_name.split('_')[0]
    val_tqdm = tqdm(val_loader, total=len(val_loader), desc='Validation')
    val_losses = torch.zeros(len(val_loader), dtype=torch.float32, device=model.device, requires_grad=False)

    for i, batch in enumerate(val_tqdm):
        x1, labels = batch  # x1: (B, C, H, W), labels: (B,)
        x1, labels = x1.to(model.device), labels.to(model.device)
        x0 = torch.randn_like(x1, device=model.device)

        if mode == 'flow-matching':
            t = torch.rand([x1.shape[0], 1, 1, 1], device=model.device)
            xt = (1 - t) * x0 + t * x1  # (B, D)
            delta = x1 - x0
            loss = mse_loss(model(xt, t.squeeze(), labels), delta)
        elif mode == 'diffusion':
            t = torch.randint(1, model.n_timesteps, [x1.shape[0]], dtype=torch.long, device=model.device)
            loss = mse_loss(model(x1, t=t, labels=labels, noise=x0), x0)
        else:
            raise ValueError('Invalid validation mode.')

        val_tqdm.set_postfix(validation_loss=f'{loss.item():.8f}')
        val_losses[i] = loss.item()

    mean_val_loss = torch.mean(val_losses)
    # compare first 8 samples of the last batch
    if epoch % 10 == 0:
        img_size = x1.shape[-1]
        for n_steps in {1, 10, 100}:
            print(f'Sampling {n_steps} steps...')
            img = model.sample_img(
                img_size=img_size, n_steps=n_steps, label=torch.tensor([1], dtype=torch.long, device=model.device)
            )
            if mode == 'diffusion':
                img.clamp_(0, 1)
                # img = (img + 1)/2   # rescale to [0,1]
            save_image(img, f'./data/assets/{run_name}/{epoch:03}_[{n_steps}]_validation_image.pdf')

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
    run_name = mode + '_' + run_name + f'_{img_size}'
    experiment_path = os.path.join(path, run_name)
    if not os.path.isdir(f'./data/assets/{run_name}'):
        os.makedirs(f'./data/assets/{run_name}', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mode == 'flow-matching':
        model = FlowMatchingNet(net, lr, device=device)
    elif mode == 'diffusion':
        model = DiffusionNet(net, 1000, lr, device=device)
    else:
        raise ValueError('Invalid model mode.')
    if config is not None:
        model.set_config(config)

    if patience:
        early_stopper = EarlyStopper(experiment_path=experiment_path, patience=patience)

    # data import and augmentation
    basic_transform = Compose([
        ToImage(),
        ToDtype(torch.float, scale=True),
    ])
    training_data, validation_data = random_split(
        MRIDataset(f'./data/preprocessed_{img_size}/annotations.csv', transform=basic_transform),
        [0.9, 0.1],
        torch.manual_seed(42)   # for reproducible validation set
    )
    if augmented:
        gaussian_blurred = MRIDataset(
            f'./data/preprocessed_{img_size}/annotations.csv',
            transform=Compose([GaussianBlur(kernel_size=11, sigma=5.), basic_transform])
        )
        rotated = MRIDataset(
            f'./data/preprocessed_{img_size}/annotations.csv',
            transform=Compose([RandomRotation(15), basic_transform])
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
        'model_mode': mode,
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
                x1, labels = batch  # (B, 1, W, H), (B)
                x1, labels = x1.to(device), labels.to(device)
                x0 = torch.randn_like(x1, device=device)

                if mode == 'flow-matching':
                    t = torch.rand(x1.shape[0], 1, 1, 1, device=device)
                    xt = (1 - t) * x0 + t * x1  # (B, D)
                    delta = x1 - x0
                    loss = model.backprop(model(xt, t.squeeze(), labels), delta, do_return=True)
                elif mode == 'diffusion':
                    t = torch.randint(1, model.n_timesteps, [x1.shape[0]], dtype=torch.long, device=device)  # (B)
                    loss = model.backprop(model(x1, t=t, labels=labels, noise=x0), x0, do_return=True)

                train_tqdm.set_postfix(training_loss=f'{loss.item():.8f}')
                train_loss[i] = loss.item()

            # logging
            mean_train_loss = torch.mean(train_loss)
            mean_val_loss = validate(model, epoch, val_loader, run_name)
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


if __name__ == '__main__':
    train(
        Unet(64, 128, 256, 512, embed_dim=128),
        128,
        lr=1e-4,
        run_name='unet',
        epochs=100,
        patience=10,
        augmented=True,
        mlflow_config=MlflowConfig,
    )
    # train(
    #     VisionTransformer(
    #         128,
    #         in_channels=1,
    #         depth_encoder=5,
    #         n_heads=16,
    #         patch_size=16,
    #         embed_dim=512,
    #     ),
    #     128,
    #     run_name='vit',
    #     epochs=60,
    #     augmented=True,
    #     patience=6,
    #     mlflow_config=MlflowConfig
    # )
    train(
        Unet(64, 128, 256, 512, embed_dim=64),
        img_size=128,
        mode='diffusion',
        lr=1e-3,
        run_name='unet',
        epochs=50,
        patience=10,
        augmented=False,
        mlflow_config=MlflowConfig,
    )
