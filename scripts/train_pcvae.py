"""Training script for Prediction Constrained Variational Autoencoder (PCVAE)"""

import torch

from torch.utils.data import DataLoader

import argparse
import yaml
import os

from models.pcvae import PredictionConstrainedVAE
from training.trainlib import trainPCVAE
from utils.datatools import load_data

def returnPCVAE(config):
    """Train the PCVAE from the config file.

    Config keys:
        latent_dims: The number of latent dimensions.
        architecture: The architecture type of the VAE.
        beta: Coefficient of KL loss term.
        lambda: Coefficient of prediction loss term.
        label_weight: Coefficient of loss computed on labelled data.
        unlabel_weight: Coefficient of loss computed on unlabelled data.
        num_classes: The number of classes for the classifier.
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
    LAMBDA_VAL = config["model"]["lambda"]
    L_WEIGHT = config["model"]["label_weight"]
    U_WEIGHT = config["model"]["unlabel_weight"]
    NUM_CLASSES = config["model"]["num_classes"]

    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    SAVE_MODEL = config["training"]["save_model"]

    DATASET_NAME = config["dataset"]["name"]
    NUM_TRAIN = config["dataset"]["num_train"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # custom run name with params and timestamp
    RUN_NAME = config["run_name"]

    # Get unlabelled data
    data_l, data_u, INPUT_SIZE = load_data(DATASET_NAME, NUM_TRAIN)
    dataLoader_l = DataLoader(data_l, batch_size=BATCH_SIZE, shuffle=True)
    dataLoader_u = DataLoader(data_u, batch_size=BATCH_SIZE, shuffle=False)

    # Create an instance of PCVAE
    pcvae = PredictionConstrainedVAE(
        ARCHITECTURE, LATENT_DIMS, NUM_CLASSES, INPUT_SIZE
    ).to(device)  # GPU

    # Train PCVAE
    pcvae = trainPCVAE(
        pcvae,
        dataLoader_u,
        dataLoader_l,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        beta=BETA,
        lambda_=LAMBDA_VAL,
        l_weight=L_WEIGHT,
        u_weight=U_WEIGHT,
        run_name=RUN_NAME,
        device=device,
        config=config,
    )

    # Save the model
    if bool(SAVE_MODEL):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, RUN_NAME)
        torch.save(pcvae.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

    return pcvae


def main(config=None):
    # Example config
    if config is None:
        config = {
            "run_name": "test_pcvae",
            "model": {
                "name": "PCVAE",
                "latent_dims": 20,
                "architecture": "linear",
                "beta": 1,
                "lambda": 1,
                "label_weight": 1,
                "unlabel_weight": 1,
                "num_classes": 10,
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "num_epochs": 20,
                "save_model": True,
            },
            "dataset": {"name": "MNIST", "num_train": 100},
        }

    pcvae = returnPCVAE(config)
    return pcvae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
