"""Training script for VAE model."""

import torch
import torchvision
import argparse
import yaml
import os

from models.vae import VariationalAutoencoder
from training.trainlib import trainVAE

def returnVAE(config):
    """Train the VAE from the config file.

    Config keys:
        latent_dims: The number of latent dimensions.
        architecture: The architecture type of the VAE.
        beta: Coefficient of KL loss term.
        batch_size: The batch size for training.
        learning_rate: The learning rate for training.
        num_epochs: The number of epochs to train.
        save_model: Whether to save the model.
        dataset_name: The name of the dataset (MNIST/CIFAR).
    """
    # get config values (not the best but in case config structure changes)
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

    # download and preprocess data
    if DATASET_NAME == "MNIST":
        INPUT_SIZE = 28 * 28

        data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "./data", transform=torchvision.transforms.ToTensor(), download=True
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    elif DATASET_NAME == "CIFAR10":
        INPUT_SIZE = 3 * 32 * 32

        data = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                "./data",
                transform=torchvision.transforms.ToTensor(),
                download=True,
                train=True,
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    else:
        raise ValueError(f"Dataset {DATASET_NAME} not supported")

    # set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize model
    vae = VariationalAutoencoder(ARCHITECTURE, LATENT_DIMS, INPUT_SIZE).to(
        device
    )  # GPU

    # train model
    recon_losses, kl_losses, vae = trainVAE(
        vae, data, BETA, NUM_EPOCHS, LEARNING_RATE, RUN_NAME, device, config=config
    )

    # save model
    if SAVE_MODEL:
        checkpoint_dir = "./checkpoints"
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
                "beta": 1.0,
            },
            "training": {
                "batch_size": 128,
                "learning_rate": 0.001,
                "num_epochs": 100,
                "save_model": True,
            },
            "dataset": {"name": "MNIST"},
        }

    vae = returnVAE(config)
    return vae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
