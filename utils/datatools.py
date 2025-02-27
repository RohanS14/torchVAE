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
            image = image.unsqueeze(0) # TODO: needed for saheli's conv
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

# Create a dataset and apply transformations
def blackout_dataloader(DATASET_NAME, NUM_TRAIN=None):
    """Create labeled and unlabeled dataloaders with blackout transformations."""
    
    blackout_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1, scale=(0.5, 0.75), ratio=(0.3, 3.3), value=0)  # Directly applying RandomErasing
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if DATASET_NAME == "MNIST":        
        # Unmasked dataset
        train_data_unmasked = torchvision.datasets.MNIST('./data', transform=transform, download=True, train=True)
        valid_data_unmasked = torchvision.datasets.MNIST('./data', transform=transform, download=True, train=False)
        
        # Masked dataset
        train_data_masked = torchvision.datasets.MNIST('./data', transform=blackout_transform, download=True, train=True)
        valid_data_masked = torchvision.datasets.MNIST('./data', transform=blackout_transform, download=True, train=False)
    else:
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

    if NUM_TRAIN is not None:
        # Generate custom train/val split for unmasked and masked data
        combined_dataset_masked = ConcatDataset([train_data_masked, valid_data_masked])
        combined_dataset_unmasked = ConcatDataset([train_data_unmasked, valid_data_unmasked])
        
        total_size = len(combined_dataset_masked)
        num_labeled = min(NUM_TRAIN, total_size)  # Ensure we don't exceed dataset size
        num_unlabeled = total_size - num_labeled
        
        data_l_masked, data_u_masked = random_split(
            combined_dataset_masked, [num_labeled, num_unlabeled],
            generator=torch.Generator().manual_seed(42)
        )
        data_l_unmasked, data_u_unmasked = random_split(
            combined_dataset_unmasked, [num_labeled, num_unlabeled],
            generator=torch.Generator().manual_seed(42)
        )
    
        return (data_l_masked, data_u_masked), (data_l_unmasked, data_u_unmasked)
    else:
        return (train_data_masked, valid_data_masked), (train_data_unmasked, valid_data_unmasked)