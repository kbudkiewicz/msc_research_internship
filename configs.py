import os

from dotenv import load_dotenv
from dataclasses import dataclass
load_dotenv()


@dataclass
class MlflowConfig:
    """
    Tracking username, password and server uri have to be set manually below or in a .env file.
    """
    mlflow_experiment_name: str = 'msc_research_internship'
    mlflow_server_uri: str = os.environ.get('MLFLOW_SERVER_URI', None)
    mlflow_tracking_username: str = os.environ.get('MLFLOW_TRACKING_USERNAME', None)
    mlflow_tracking_passwd: str = os.environ.get('MLFLOW_TRACKING_PASSWORD', None)


class BaseUnetConfig:
    """
    Default parameters of the `UNet <http://arxiv.org/abs/1505.04597>`_.
    """
    lr: float = 1e-4
    embed_dim: int = 64
    conv_kwargs: dict = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    pool_kwargs: dict = {'kernel_size': 2, 'stride': 2, 'padding': 0}  # when downscaling
    upscale_factor: int = 2
    channel_factor: int = 2
    upscale_kwargs: dict = {'scale_factor': upscale_factor, 'mode': 'bilinear', 'align_corners': True}


@dataclass
class VitBase:
    """
    Default parameters of the ViT from Table 1. of `VisionTransformer <http://arxiv.org/abs/2010.11929>`_.
    """
    patch_size: int = 16
    depth_encoder: int = 12
    embed_dim: int = 768
    mlp_dim: int = 3072
    n_heads: int = 12
    dropout_rate: float = 0.0


@dataclass
class MyVit(VitBase):
    n_labels: int = 4
    patch_size: int = 16
    depth_encoder: int = 6
    embed_dim: int = 512
    mlp_dim: int = 128
    n_heads: int = 8
    dropout_rate: float = 0.0
    num_groups: int = 32  # if GroupNorm is used


@dataclass
class VitLarge(VitBase):
    depth_encoder: int = 24
    embed_dim: int = 1024
    mlp_dim: int = 4096
    n_heads: int = 16


@dataclass
class VitHuge(VitBase):
    depth_encoder: int = 32
    embed_dim: int = 1280
    mlp_dim: int = 5120
    n_heads: int = 16
