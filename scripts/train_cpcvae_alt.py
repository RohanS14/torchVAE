"""Training script for Consistency Constrained Variational Autoencoder (CPCVAE)"""

import torch
from torch.utils.data import DataLoader

import argparse
import yaml
import os

from models.cpcvae import ConsistencyConstrainedVAE
from training.trainlib import trainCPCVAE, trainCPCVAE_small, trainCPCVAE_saheli
from utils.datatools import load_data


def returnCPCVAE(config):
    """Train the VAE from the config file.

    Config keys:
        latent_dims: The number of latent dimensions.
        architecture: The architecture type of the VAE.
        beta: Coefficient of KL loss term.
        lambda: Coefficient of prediction loss term.
        gamma: Coefficient of consistency loss term.
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
    DIST = config["model"]["dist"]
    BETA = config["model"]["beta"]
    LAMBDA_VAL = config["model"]["lambda"]
    GAMMA = config["model"]["gamma"]
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

    # custom run name
    RUN_NAME = config["run_name"]

    # Get unlabelled data
    # data_l, data_u, INPUT_SIZE = load_data(DATASET_NAME, NUM_TRAIN)
    # dataLoader_l = DataLoader(data_l, batch_size=BATCH_SIZE, shuffle=True)
    # dataLoader_u = DataLoader(data_u, batch_size=BATCH_SIZE, shuffle=True)

    # Saheli's data loading code
    
    import torchvision
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset

    batch_size = 50
    valid_batch_size = 6000

    # batch_size = 128

    class SimpleDataset(Dataset):
        def __init__(self, x,y):
            self.x = x
            self.y = y

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
        
        def __len__(self):
            return len(self.y)

    data = torchvision.datasets.MNIST('./mnist_data', transform=torchvision.transforms.ToTensor(), download=True, train=True)
    images, labels = zip(*data)
    train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=.1, stratify=labels, random_state=72)
    unlabelled_images, labelled_images, unlabelled_labels, labelled_labels = train_test_split(train_images, train_labels, test_size=100/len(train_labels), stratify=train_labels, random_state=54)

    dataLoader_l = torch.utils.data.DataLoader(SimpleDataset(labelled_images, labelled_labels),batch_size=batch_size,shuffle=True)
    dataLoader_u = torch.utils.data.DataLoader(SimpleDataset(unlabelled_images, unlabelled_labels),batch_size=batch_size,shuffle=True)
    valid_data = torch.utils.data.DataLoader(SimpleDataset(valid_images, valid_labels),batch_size=valid_batch_size,shuffle=True)

    test_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./mnist_data',
                transform=torchvision.transforms.ToTensor(),
                download=True, train=False),
            batch_size=1000,
            shuffle=True)

    # check the distribution of labels in the train set
    for i in range(NUM_CLASSES):
        print(
            f"Number of samples with label {i}: {len([x for x in dataLoader_l.dataset if x[1] == i])}"
        )

    INPUT_SIZE = 28 * 28

    # Create an instance of CPCVAE
    cpcvae = ConsistencyConstrainedVAE(
        ARCHITECTURE, LATENT_DIMS, NUM_CLASSES, INPUT_SIZE, DIST
    ).to(device)

    # Train CPCVAE
    # cpcvae = trainCPCVAE(
    #     cpcvae,
    #     dataLoader_u,
    #     dataLoader_l,
    #     epochs=NUM_EPOCHS,
    #     lr=LEARNING_RATE,
    #     beta=BETA,
    #     lambda_=LAMBDA_VAL,
    #     gamma = GAMMA,
    #     l_weight=L_WEIGHT,
    #     u_weight=U_WEIGHT,
    #     run_name=RUN_NAME,
    #     device=device,
    #     config=config,
    # )
    
    cpcvae = trainCPCVAE_saheli(
        cpcvae,
        dataLoader_u,
        dataLoader_l,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        beta=BETA,
        lambda_=LAMBDA_VAL,
        gamma = GAMMA,
        l_weight=L_WEIGHT,
        u_weight=U_WEIGHT,
        run_name=RUN_NAME,
        device=device,
        config=config,
        valid_data=valid_data
    )

    # Save the model
    if bool(SAVE_MODEL):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, RUN_NAME)
        torch.save(cpcvae.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

    return cpcvae


def main(config=None):
    # Example config
    if config is None:
        config = {
            "run_name": "test_cpcvae",
            "model": {
                "name": "CPCVAE",
                "latent_dims": 50,
                "architecture": "fc",
                "dist": "bern",
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

    cpcvae = returnCPCVAE(config)
    return cpcvae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config file"
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = None

    main(config)
