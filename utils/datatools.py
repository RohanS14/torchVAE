"""Tools for loading and preprocessing data."""

import torch
import torchvision

from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, random_split

from tqdm import tqdm
from collections import defaultdict


class EncodedDataset(Dataset):
    """Dataset wrapper that encodes all images beforehand."""

    def __init__(self, dataset, encode_fn, device):
        self.dataset = dataset
        self.encode_fn = encode_fn
        self.device = device
        self.encoded_images = []
        self.labels = []

        # Encode all images beforehand and store them
        for image, label in tqdm(dataset):
            encoded_image = self.encode_fn(image, device)
            self.encoded_images.append(encoded_image)
            self.labels.append(label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.encoded_images[idx], self.labels[idx]


def load_data(dataset_name, num_train):
    """Load the data for the given dataset name and number of training samples."""
    preprocess = transforms.Compose([transforms.ToTensor()])

    if dataset_name == "MNIST":
        input_size = 28 * 28

        train_data = torchvision.datasets.MNIST(
            "./data", transform=preprocess, download=True, train=True
        )
        valid_data = torchvision.datasets.MNIST(
            "./data", transform=preprocess, download=True, train=False
        )

    elif dataset_name == "CIFAR10":
        input_size = 3 * 32 * 32

        train_data = torchvision.datasets.CIFAR10(
            "./data", transform=preprocess, download=True, train=True
        )
        valid_data = torchvision.datasets.CIFAR10(
            "./data", transform=preprocess, download=True, train=False
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    # Generate custom train/val split
    combined_dataset = ConcatDataset([train_data, valid_data])
   
    total_size = len(combined_dataset)
    num_unlabeled = total_size - num_train
    data_l, data_u = random_split(
        combined_dataset,
        [num_train, num_unlabeled],
        generator=torch.Generator().manual_seed(42),
    )

    return data_l, data_u, input_size
