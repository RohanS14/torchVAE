import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

from models.vae import VariationalAutoencoder
from train_vae import returnVAE
from train_logreg import returnLogReg, loadData

import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm

class EncodedDataset(Dataset):
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

def main(config=None):
    
    # default config
    if config is None:
        config = {
            "run_name": "default",
            "VAE": {
                "model": {
                    "name": "VAE",
                    "latent_dims": 20,
                    "architecture": "linear",
                    "beta": 1.0
                },
                "training": {
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "num_epochs": 2,
                    "save_model": "True"
                }
            },
            "logreg": {
                "model": {
                    "name": "logreg",
                    "input_size": 3072,
                    "num_classes": 10
                },
                "training": {
                    "learning_rate": 0.001,
                    "num_epochs": 2,
                    "regularization": "None",
                    "lambda": 0.01,
                    "stop_100": "False",
                    "batch_size": 64,
                    "save_model": "True"
                }
            },
            "dataset": {
                "name": "MNIST",
                "n_train": 100 
            }
        }
    
    # set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ## Train VAE
    
    # set up unlabeled dataloader for VAE
    vae_config = config["VAE"]
    vae_config["run_name"] = config["run_name"]
    vae_config["dataset"] = config["dataset"]
    vae = returnVAE(vae_config)
    
    # Define encoding function
    def encode(x, device):
        with torch.no_grad():
            vae.encoder.eval()
            x = x.to(device)
            mean, sigma = vae.encoder(x.to(device))
            
            # Sample an image from the distribution
            z = vae.encoder.sample(mean, sigma)
            z = z.detach()
            
            return z

    ## Train logistic regression model on encoded images
    logreg_config = config["logreg"]
    logreg_config["run_name"] = config["run_name"]
    logreg_config["dataset"] = config["dataset"]
    
    # Load and preprocess labelled dataloader
    DATASET_NAME = logreg_config["dataset"]["name"]
    NUM_TRAIN = config["dataset"]["num_train"]
    BATCH_SIZE = logreg_config["training"]["batch_size"]
    
    # Get data and encode using VAE
    train_data, valid_data = loadData(DATASET_NAME, NUM_TRAIN)
    print("Encoding data...")
    encoded_train_data = EncodedDataset(train_data, encode, device)
    encoded_val_data = EncodedDataset(valid_data, encode, device)
    print(f'Encoded train data shape: {encoded_train_data[0][0].shape}')

    train_dataLoader = DataLoader(encoded_train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataLoader = DataLoader(encoded_val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Train logistic regression model
    model = returnLogReg(logreg_config, data=(train_dataLoader, valid_dataLoader))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)