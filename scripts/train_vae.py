import torch
import torchvision
import argparse
import json
import os
from datetime import datetime

from models.vae import VariationalAutoencoder
from training.trainlib import trainVAE

def returnVAE(config):
    # get config values (not the best but in case config structure changes)
    MODEL_NAME = config["model"]["name"]
    LATENT_DIMS = config["model"]["latent_dims"]
    ARCHITECTURE = config["model"]["architecture"]
    BETA = config["model"]["beta"]
    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    SAVE_MODEL = config["training"]["save_model"]
    DATASET_NAME = config["dataset"]["name"]
    
    # custom run name with params and timestamp
    RUN_NAME = config["run_name"]
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_NAME += f'-vae-{DATASET_NAME}-{ARCHITECTURE}-{LATENT_DIMS}-{BETA}-{LEARNING_RATE}-{date}'
    
    # download and preprocess data
    if DATASET_NAME == "MNIST":            
        data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', 
                transform=torchvision.transforms.ToTensor(), 
                download=True),
            batch_size=BATCH_SIZE,
            shuffle=True)
    elif DATASET_NAME == "CIFAR10":
            data = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10('./data',
                    transform=torchvision.transforms.ToTensor(),
                    download=True, train=True),
            batch_size=BATCH_SIZE,
            shuffle=True)
    else:
        raise ValueError(f"Dataset {DATASET_NAME} not supported")
    
    # set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize model
    vae = VariationalAutoencoder(ARCHITECTURE, LATENT_DIMS).to(device) # GPU
    
    # train model
    recon_losses, kl_losses, vae = trainVAE(vae, data, BETA, NUM_EPOCHS, LEARNING_RATE, RUN_NAME, device)
    
    # save model
    if SAVE_MODEL:
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, RUN_NAME)
        torch.save(vae.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")
        
    return vae

def main(config=None):
    # default config
    if config is None:
        config = {
            "run_name": "default",
            "model": {
                "name": "VAE",
                "latent_dims": 20,
                "architecture": "linear",
                "beta": 1.0
            },
            "training": {
                "batch_size": 128,
                "learning_rate": 0.001,
                "num_epochs": 100,
                "save_model": True
            },
            "dataset": {
                "name": "MNIST"
            }
        }
        
    vae = returnVAE(config)
    return vae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)