"""Training script for Variational Autoencoder (VAE) with Blackout Augmentation"""

import torch
from torch.utils.data import DataLoader
from utils.datatools import blackout_dataloader  # Updated import
import argparse
import yaml
import os

from models.vae import VariationalAutoencoder  # Import the VAE model from vae.py
from training.trainlib import trainVAE_blackout  # Import the new training function

def returnVAE(config):
    """
    Train a plain VAE from the config file using blackout transformations.

    Config keys:
        latent_dims: The number of latent dimensions.
        architecture: The architecture type of the VAE.
        beta: Coefficient for the KL loss term.
        batch_size: The batch size for training.
        learning_rate: The learning rate for training.
        num_epochs: The number of epochs to train.
        save_model: Whether to save the model.
        dataset_name: The name of the dataset (e.g. MNIST).
        num_train: The number of samples (used for blackout_dataloader).
    """
    # Extract config values
    LATENT_DIMS = config["model"]["latent_dims"]
    ARCHITECTURE = config["model"]["architecture"]
    BETA = config["model"]["beta"]

    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    SAVE_MODEL = config["training"]["save_model"]

    DATASET_NAME = config["dataset"]["name"]
    NUM_TRAIN = config["dataset"]["num_train"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Custom run name with params and timestamp
    RUN_NAME = config["run_name"]

    # Create an instance of the VAE using the specified architecture.
    # Note: For MNIST, input_dims = 28*28.
    INPUT_SIZE = 28 * 28
    vae = VariationalAutoencoder(ARCHITECTURE, LATENT_DIMS, input_dims=INPUT_SIZE).to(device)

    # Train the VAE with blackout-augmented data.
    vae = trainVAE_blackout(
        vae,
        DATASET_NAME,
        NUM_TRAIN,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        beta=BETA,
        run_name=RUN_NAME,
        device=device,
        config=config,
    )

    # Save the trained model if configured to do so.
    if bool(SAVE_MODEL):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, RUN_NAME)
        torch.save(vae.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

    return vae

def main(config=None):
    """Main function to train a VAE with blackout-augmented data."""
    # Default config if none provided
    if config is None:
        config = {
            "run_name": "test_vae_blackout",
            "model": {
                "name": "VAE",
                "latent_dims": 20,
                "architecture": "linear",  # Options: "linear", "fc", or "conv"
                "beta": 10,
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "num_epochs": 20,
                "save_model": True,
            },
            "dataset": {"name": "MNIST", "num_train": 1000},  # Using 1000 samples for blackout_dataloader
        }

    vae = returnVAE(config)
    return vae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config file"
    )
    args = parser.parse_args()

    # Load the configuration from the specified file if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = None

    main(config)