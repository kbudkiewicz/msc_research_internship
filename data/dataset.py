import os
import random
import pandas as pd
import torch
import torchvision.transforms.functional as F

from typing import Any, Callable, Optional, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image, write_jpeg, ImageReadMode
from torchvision.transforms import RandomCrop, RandomRotation, GaussianBlur

data_label_mapping = {
    'notumor': 0,
    'glioma': 1,
    'meningioma': 2,
    'pituitary': 3
}


def create_annotations_file(
        path: os.PathLike | str = './preprocessed',
        filename: str = 'annotations.csv',
        sep: str = ',',
    ) -> None:
    """
    Create an annotation .csv file for the Dataset. If no path is provided, the file will be saved in current directory.
    The annotation file contains the *absolute* paths to the respective samples / images.

    Args:
        path (os.PathLike, str): Path to the folder containing the training dataset. Annotation file is created there.
        filename (str): Name of the annotation file.
        sep (str): Separator used to separate data in the annotation file. Default is a comma.

    .. Note::
        The annotation file is created with `.csv` extension in ``/path``, e.g., ``path='preprocessed'`` it will be
        created in ./preprocessed.
    """
    destination_path = os.path.join(path, filename)
    assert os.path.exists(path), f'The path {path} does not exist. Make sure path is an existing directory.'
        # os.makedirs(path)

    with open(destination_path, 'w') as f:
        for root, _, files in os.walk(path):
            _, label_str = os.path.split(root)
            # check if dir describes a tumor type
            if label_str in data_label_mapping.keys():
                label = data_label_mapping[label_str]
                # save sample name with according label
                for sample_name in files:
                    if sample_name.endswith('.jpg'):
                        image_path = os.path.join(root, sample_name)
                        abs_path = os.path.abspath(image_path)
                        f.write(f'{abs_path}{sep}{label}\n')


def preprocess(
        final_img_size: int = 512,
        original_dir_path: os.PathLike | str = './original/training',
        preprocessed_dir: os.PathLike | str = './preprocessed',
    ) -> None:
    """
    Preprocess training data into a new directory. The image files are rewritten into `.jpeg`.

    Args:
        final_img_size (int): Final height and width of the preprocessed image. Defaults to 512.
        original_dir_path (os.PathLike, str): Path to the original training dataset.
        preprocessed_dir (os.PathLike, str): Path to the new, preprocessed training dataset.

    .. note::
        The data is preprocessed with similarly (if not exactly) as in `Deep Residual Learning for Image Recognition
        <https://arxiv.org/abs/1512.03385>`__ or `ImageNet Classification with Deep Convolutional Neural Networks
        <https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>`__.

        The images are first padded with 0. along the shorter dimension of height or width. Then they are resized to
        (512, 512).
    """
    preprocessed_dir += f'_{final_img_size}'
    if not os.path.isdir(preprocessed_dir):
        print(f'Path {preprocessed_dir} does not exist.\nCreating directory {preprocessed_dir}...')
        os.mkdir(preprocessed_dir)
        # TODO: redo for generality
        os.mkdir(preprocessed_dir + f'/training')
        os.mkdir(preprocessed_dir + f'/training/glioma')
        os.mkdir(preprocessed_dir + f'/training/meningioma')
        os.mkdir(preprocessed_dir + f'/training/notumor')
        os.mkdir(preprocessed_dir + f'/training/pituitary')
        print(f'Directory {preprocessed_dir} created.')

    # preprocess every file from /dir_path into /preprocessed
    for dirpath, dirnames, filenames in os.walk(original_dir_path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                img_path = os.path.join(dirpath, filename)
                destination_path = img_path.replace('./original', preprocessed_dir).replace('\\', '/')
                img_tensor = decode_image(img_path, mode=ImageReadMode.GRAY)
                assert img_tensor.ndim == 3, f'Image tensor must be a 3D tensor but is {img_tensor.shape}.'
                # assert img_tensor.shape[0] in {1, 3}, f'Number of channels must be 1 but is {img_tensor.shape[0]}.'

                # pad to an aspect ratio ~1:1
                if img_tensor.shape[1] != img_tensor.shape[2]:
                    width_is_bigger = img_tensor.shape[1] > img_tensor.shape[2]
                    diff = abs(img_tensor.shape[1] - img_tensor.shape[2])
                    if width_is_bigger:
                        img_tensor = F.pad(img_tensor, [diff//2, 0])
                    elif not width_is_bigger:
                        img_tensor = F.pad(img_tensor, [0, diff//2])

                # resize to the desired W and H
                if img_tensor.shape[-2:] != [final_img_size, final_img_size]:
                    img_tensor = F.resize(img_tensor, [final_img_size, final_img_size])  # redo with (1, 256, 256)
                assert img_tensor.shape[-2:] == torch.Size([final_img_size, final_img_size]), \
                    f'Image was not resized. Got: {img_tensor.shape}'
                write_jpeg(img_tensor, destination_path)
            else:
                continue

    print('Preprocessing finished. Creating annotations file...')
    create_annotations_file(preprocessed_dir)
    print('Done.')

preprocess(128)

class MRIDataset(Dataset):
    """
    See the `following <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset
    -for-your-files>`__ implementation.

    All required paths are absolute.

    Args:
        annotations_file_path(os.PathLike or str): Path to annotations file.
        sep (str): Separator or delimiter used in the .csv.
        img_dir (os.PathLike or str): Path to directory containing images.
        transform (Callable, optional): Optional transforms to be applied on a sample image.
        random_transforms (tuple, optional): Tuple of torchvision transforms (Module) to be applied randomly while
            sampling a batch

    The (transformed) images are saved as ``Tensors`` with :math:`CxHxW`, and their according label encoded as an
    integer via the ``label_map``.
    """
    def __init__(
        self,
        annotations_file_path: os.PathLike | str,
        sep: str = ',',
        img_dir: Optional[os.PathLike | str] = None,
        transform: Optional[Callable] = None,
        random_transforms: Optional[tuple[torch.nn.Module, ...]] = None,
    ) -> None:
        if img_dir:
            assert os.path.isdir(img_dir), f'Directory {img_dir} does not exist!'
        assert os.path.exists(annotations_file_path), 'annotations_file does not exist!'
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file_path, sep=sep)
        self.label_map = data_label_mapping
        self.transform = transform
        self.random_transforms = random_transforms if random_transforms else (
            RandomCrop([256]), RandomRotation([-180, 180]), GaussianBlur(kernel_size=11, sigma=10)
        )

    def apply_random_transform(self, img: Tensor, return_name: bool = False) -> Tuple[Tensor, str]:
        """
        Apply a random image transform to the input image. Default transforms are saved in `self.transforms`.
        Args:
            img (Tensor): Image tensor to be transformed.
            return_name (bool, optional): Whether to return the name of the applied transform. Meant for debugging
                purposes.
        Return:
            Transformed image tensor.
        """
        transform = random.choice(self.random_transforms)
        if return_name:
            return transform(img), transform.__class__.__name__     # for debugging purposes
        else:
            return transform(img)

    def __getitem__(
        self, idx: int,
        apply_random_transform: bool = False
    ) -> Tuple[Any, Any]:
        if self.img_dir:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        else:
            img_path = self.img_labels.iloc[idx, 0]
        label = torch.tensor(self.img_labels.iloc[idx, 1], dtype=torch.int)
        img = decode_image(img_path)    # CxHxW,

        # normalize to [0, 1]
        if self.transform:
            img = self.transform(img)
        if apply_random_transform:
            img = self.apply_random_transform(img)

        return img, label.unsqueeze(-1)

    def __len__(self):
        return len(self.img_labels)
