import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
import torchvision
import torch.nn as nn


from datetime import datetime

from torch.utils.data import DataLoader, ConcatDataset, random_split


# Define a transformation to black out a random rectangular region
class Blackout(object):
    def __init__(self, p=1, scale=(0.5, 0.75), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        #if torch.rand(1) < self.p:
        return transforms.RandomErasing(p=self.p, scale=self.scale, ratio=self.ratio, value=0)(img)
        return img

# Create a dataset and apply transformations
def blackout_dataloader(DATASET_NAME, NUM_TRAIN):
    blackout_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing()
    ])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if DATASET_NAME == "MNIST":
        input_size = 28 * 28
        
        # Unmasked dataset
        train_data_unmasked = torchvision.datasets.MNIST('./data', transform=transform,
                                                         download=True, train=True)
        blackout_data_unmasked = torchvision.datasets.MNIST('./data', transform=transform,
                                                            download=True, train=False)
        
        # Masked dataset
        train_data_masked = torchvision.datasets.MNIST('./data', transform=blackout_transform,
                                                       download=True, train=True)
        blackout_data_masked = torchvision.datasets.MNIST('./data', transform=blackout_transform,
                                                          download=True, train=False)
        
    if NUM_TRAIN != "None":
        # Generate custom train/val split for unmasked and masked data
        combined_dataset_masked = ConcatDataset([train_data_masked, blackout_data_masked])
        combined_dataset_unmasked = ConcatDataset([train_data_unmasked, blackout_data_unmasked])
        total_size = len(combined_dataset_masked)
        num_labels = NUM_TRAIN
        num_unlabeled = total_size - num_labels
        
        data_l_masked, data_u_masked = random_split(combined_dataset_masked, [num_labels, num_unlabeled],
                                                    generator=torch.Generator().manual_seed(42))
        data_l_unmasked, data_u_unmasked = random_split(combined_dataset_unmasked, [num_labels, num_unlabeled],
                                                        generator=torch.Generator().manual_seed(42))
    
    return (data_l_masked, data_u_masked), (data_l_unmasked, data_u_unmasked)
