import os
import torch
import pandas as pd
import PIL.Image as pil

from datasets import load_dataset
from typing import Callable, Optional, Tuple
from PIL.Image import Image
from PIL.ImageOps import pad, grayscale
from torch import Tensor
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    """
    See the `following <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset
    -for-your-files>`__ implementation.

    All required paths are absolute.

    Args:
        img_size (int): The desired size of images.
        sep (str): Separator or delimiter used in the .csv.
        img_dir (os.PathLike or str): Path to directory containing images.
        transform (Callable, optional): Optional transforms to be applied on a sample image.

    The (transformed) images are saved as ``Tensors`` with :math:`CxHxW`, and their according label encoded as an
    integer via the ``label_map``.
    """
    def __init__(
        self,
        img_size: int,
        sep: str = ',',
        img_dir: Optional[os.PathLike | str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        if img_dir:
            assert os.path.isdir(img_dir), f'Directory {img_dir} does not exist!'
        annotations_file_path = f'./data/preprocessed_{img_size}/annotations.csv'
        assert os.path.exists(annotations_file_path), 'annotations_file does not exist!'
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file_path, sep=sep)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Image, Tensor]:
        r"""
        Returns:
            - img (PIL.Image): (C, H, W) subset [0,1]
            - label (torch.LongTensor)
        """
        if self.img_dir:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        else:
            img_path = self.img_labels.iloc[idx, 0]
        label = torch.tensor(self.img_labels.iloc[idx, 1], dtype=torch.int)
        img = pil.open(img_path)    # (C, H, W)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_labels)


class SmithsonianButterfliesDataset(Dataset):
    """
    Base class for `Smithsonian Butterflies dataset <https://huggingface.co/datasets/ceyda/smithsonian_butterflies>`_
    from HuggingFace.

    The target classes (N=178) are chosen here to be the lowest level of taxonomic classification.
    """
    def __init__(self, img_size: int, transform: Optional[Callable] = None):
        self.img_size = img_size
        self.data = load_dataset("ceyda/smithsonian_butterflies", num_proc=2)['train']
        self.transform = transform
        self.str_to_int, self.n_labels = self.setup_labels_map()

    def setup_labels_map(self) -> Tuple[dict, int]:
        temp = dict.fromkeys(self.data['taxonomy'], 0)
        labels_map, idx = {}, 0
        for k in temp.keys():
            if k not in labels_map.keys():
                labels_map[k] = idx
                idx += 1
        del temp, idx

        return labels_map, len(labels_map)

    def __getitem__(self, idx: int):
        img = self.data['image'][idx]
        label = self.data['taxonomy'][idx]
        label = self.str_to_int[label]
        label = torch.tensor(label, dtype=torch.int)

        # preprocess
        # WARNING: the image is not cropped as the ruler is put in different places and cannot be reliably removed
        img = grayscale(img)
        # w, h = img.size
        # img = img.crop((0, 0, 1600, h))     # remove the ruler from the image
        img = pad(image=img, size=(self.img_size, self.img_size), color=255)    # pad with white

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)
