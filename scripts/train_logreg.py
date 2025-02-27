import torch
import torchvision
import torch.nn as nn

from torch.utils.data import DataLoader

import argparse
import yaml
import os


from models.logreg import LogisticRegression
from training.trainlib import train_logreg
from utils.datatools import load_data

def returnLogReg(config, data=None):
    # get config values (not the best but in case config structure changes)
    RUN_NAME = config["run_name"]
    INPUT_SIZE = config["model"]["input_size"]
    NUM_CLASSES = config["model"]["num_classes"]
    LEARNING_RATE = config["training"]["learning_rate"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    REGULARIZATION = config["training"]["regularization"]
    LAMBDA_VAL = config["training"]["lambda"]
    STOP_100 = config["training"]["stop_100"]
    BATCH_SIZE = config["training"]["batch_size"]
    DATASET_NAME = config["dataset"]["name"]
    NUM_TRAIN = config["dataset"]["num_train"]

    # custom run name with params and timestamp
    RUN_NAME = config["run_name"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    if data is None:
        train_data, valid_data, _ = load_data(DATASET_NAME, NUM_TRAIN)
        train_dataLoader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataLoader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_dataLoader, valid_dataLoader = data

    # Initialize model
    model = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    print(
        f"Training on {len(train_dataLoader.dataset)} samples, valid {len(valid_dataLoader.dataset)}"
    )
    model = train_logreg(
        model,
        train_dataLoader,
        valid_dataLoader,
        device,
        optimizer,
        loss_fn,
        LEARNING_RATE,
        NUM_EPOCHS,
        INPUT_SIZE,
        REGULARIZATION,
        LAMBDA_VAL,
        STOP_100,
        RUN_NAME,
    )

    # Save the model
    if bool(config["training"]["save_model"]):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, RUN_NAME)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

    return model


def main(config):
    # default config
    if config is None:
        print("Using default config")
        config = {
            "run_name": "default",
            "model": {"name": "logreg", "input_size": 784, "num_classes": 10},
            "training": {
                "learning_rate": 0.001,
                "num_epochs": 5,
                "regularization": "None",
                "lambda": 0.01,
                "stop_100": "False",
                "batch_size": 64,
            },
            "dataset": {"name": "MNIST"},
        }

    model = returnLogReg(config)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
