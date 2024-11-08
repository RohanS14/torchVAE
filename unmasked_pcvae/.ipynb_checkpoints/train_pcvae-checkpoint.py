import torch
import torchvision
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

import argparse
import json
import os
from datetime import datetime

from models.pcvae import PredictionConstrainedVAE
from training.trainlib import trainPCVAE

def loadData(DATASET_NAME, NUM_TRAIN):
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    
    if DATASET_NAME == "MNIST":
        input_size = 28*28
        
        train_data = torchvision.datasets.MNIST('./data', transform=preprocess,
                                                download=True, train=True)
        valid_data = torchvision.datasets.MNIST('./data', transform=preprocess,
                                                download=True, train=False)

    elif DATASET_NAME == "CIFAR10":
        input_size = 3*32*32
        
        train_data = torchvision.datasets.CIFAR10('./data', transform=preprocess,
                                                    download=True, train=True)            
        valid_data = torchvision.datasets.CIFAR10('./data', transform=preprocess,
                                                    download=True, train=False)
    else:
        raise ValueError(f"Dataset {DATASET_NAME} not supported")
    
    if NUM_TRAIN != "None":
        # Generate custom train/val split
        combined_dataset = ConcatDataset([train_data, valid_data])
        total_size = len(combined_dataset)
        num_labels = NUM_TRAIN
        num_unlabeled = total_size - num_labels
        data_l, data_u = random_split(combined_dataset, [num_labels, num_unlabeled],  \
                                                generator=torch.Generator().manual_seed(42))
    
    return data_l, data_u

def returnPCVAE(config):

    # get config values (not the best but in case config structure changes)
    MODEL_NAME = config["model"]["name"]
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # custom run name with params and timestamp
    RUN_NAME = config["run_name"]
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_NAME += f'-PCvae-{DATASET_NAME}-{ARCHITECTURE}-{LATENT_DIMS}-{BETA}-{LEARNING_RATE}-{date}'

    # Get unlabelled data
    data_l, data_u = loadData(DATASET_NAME, NUM_TRAIN)
    dataLoader_l = DataLoader(data_l, batch_size=BATCH_SIZE, shuffle=True)
    dataLoader_u = DataLoader(data_u, batch_size=BATCH_SIZE, shuffle=False)

    # Create an instance of PCVAE
    pcvae = PredictionConstrainedVAE(ARCHITECTURE, LATENT_DIMS, NUM_CLASSES).to(device) # GPU

    # Train PCVAE
    pcvae = trainPCVAE(pcvae, dataLoader_u, dataLoader_l, epochs=NUM_EPOCHS, lr=LEARNING_RATE, beta=BETA, \
                lambda_=LAMBDA_VAL, l_weight=L_WEIGHT, u_weight=U_WEIGHT, run_name=RUN_NAME, device=device)

    # Save the model
    if bool(config['training']['save_model']):
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, RUN_NAME)
        torch.save(pcvae.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

    return pcvae

def main(config=None):
    # Example config.json
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
                "num_classes": 10
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "num_epochs": 20,
                "save_model": True
            },
            "dataset": {
                "name": "MNIST",
                "num_train": 100
            }
        }
        
    pcvae = returnPCVAE(config)
    return pcvae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)