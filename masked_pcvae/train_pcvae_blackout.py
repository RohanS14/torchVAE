import torch
import torchvision
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import argparse
import json
import os
from datetime import datetime
import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.pcvae import PredictionConstrainedVAE
from masked_pcvae.trainlib_blackout import kl_divergence

# def loadData(DATASET_NAME, NUM_TRAIN):
#     preprocess = transforms.Compose([transforms.ToTensor()])

#     if DATASET_NAME == "MNIST":
#         input_size = 28 * 28
#         train_data = torchvision.datasets.MNIST('./data', transform=preprocess, download=True, train=True)
#         valid_data = torchvision.datasets.MNIST('./data', transform=preprocess, download=True, train=False)
    
#     if NUM_TRAIN != "None":
#         combined_dataset = ConcatDataset([train_data, valid_data])
#         total_size = len(combined_dataset)
#         num_labels = NUM_TRAIN
#         num_unlabeled = total_size - num_labels
#         data_l, data_u = random_split(combined_dataset, [num_labels, num_unlabeled], 
#                                       generator=torch.Generator().manual_seed(42))
#         return data_l, data_u

#     return train_data, valid_data

def loadData(DATASET_NAME, NUM_TRAIN):

    preprocess_unmasked = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing()
    ])
    
    # Blackout transformation for masked data
    preprocess_masked = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.2, 0.4), ratio=(0.3, 1.5))  # Blacking out random regions
    ])
    
    if DATASET_NAME == "MNIST":
        input_size = 28 * 28
        train_data_unmasked = torchvision.datasets.MNIST('./data', transform=preprocess_unmasked, download=True, train=True)
        train_data_masked = torchvision.datasets.MNIST('./data', transform=preprocess_masked, download=True, train=True)
        valid_data_unmasked = torchvision.datasets.MNIST('./data', transform=preprocess_unmasked, download=True, train=False)
        valid_data_masked = torchvision.datasets.MNIST('./data', transform=preprocess_masked, download=True, train=False)
    
    else:
        raise ValueError(f"Dataset {DATASET_NAME} not supported")
    
    if NUM_TRAIN != "None":
        combined_unmasked = ConcatDataset([train_data_unmasked, valid_data_unmasked])
        combined_masked = ConcatDataset([train_data_masked, valid_data_masked])
        
        total_size = len(combined_unmasked)
        num_labels = NUM_TRAIN
        num_unlabeled = total_size - num_labels
        
        data_l_unmasked, data_u_unmasked = random_split(combined_unmasked, [num_labels, num_unlabeled], 
                                                        generator=torch.Generator().manual_seed(42))
        data_l_masked, data_u_masked = random_split(combined_masked, [num_labels, num_unlabeled], 
                                                    generator=torch.Generator().manual_seed(42))
        
        return (data_l_masked, data_u_masked), (data_l_unmasked, data_u_unmasked)
    
    return (train_data_masked, valid_data_masked), (train_data_unmasked, valid_data_unmasked)


def returnPCVAE(config):
    DATASET_NAME = config["dataset"]["name"]
    NUM_TRAIN = config["dataset"]["num_train"]
    BATCH_SIZE = config["training"]["batch_size"]

    # Load data (masked and unmasked)
    (data_l_masked, data_u_masked), (data_l_unmasked, data_u_unmasked) = loadData(DATASET_NAME, NUM_TRAIN)

    # Create data loaders
    dataLoader_l_masked = DataLoader(data_l_masked, batch_size=BATCH_SIZE, shuffle=True)
    dataLoader_u_masked = DataLoader(data_u_masked, batch_size=BATCH_SIZE, shuffle=False)
    dataLoader_l_unmasked = DataLoader(data_l_unmasked, batch_size=BATCH_SIZE, shuffle=True)
    dataLoader_u_unmasked = DataLoader(data_u_unmasked, batch_size=BATCH_SIZE, shuffle=False)

    # Create an instance of PCVAE
    pcvae = PredictionConstrainedVAE(config["model"]["architecture"], config["model"]["latent_dims"], config["model"]["num_classes"]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Train PCVAE
    pcvae = trainPCVAE(
        pcvae, dataLoader_u_masked, dataLoader_u_unmasked, 
        dataLoader_l_masked, dataLoader_l_unmasked, 
        epochs=config["training"]["num_epochs"], 
        lr=config["training"]["learning_rate"], 
        beta=config["model"]["beta"], 
        lambda_=config["model"]["lambda"], 
        l_weight=config["model"]["label_weight"], 
        u_weight=config["model"]["unlabel_weight"], 
        run_name=config["run_name"]
    )
    
    return pcvae


def trainPCVAE(pcvae, unlabeled_masked_loader, unlabeled_unmasked_loader, labeled_masked_loader, labeled_unmasked_loader, 
               epochs=20, lr=0.001, beta=1, lambda_=1, l_weight=1, u_weight=1, run_name="default", device='cuda'):
    
    print("weights", l_weight, u_weight, lambda_)
    criterion_recon = nn.MSELoss(reduction='sum')
    criterion_class = nn.CrossEntropyLoss()
    
    # set up tensorboard writer
    print(f"Logging to {run_name}")
    writer = SummaryWriter(log_dir=f'./logs/{run_name}')
    
    optimizer = torch.optim.Adam(pcvae.parameters(), lr=lr)
    progress = tqdm.trange(epochs)
    
    for epoch in progress:
        print(epoch)
        pcvae.train()

        unlabeled_batches = zip(unlabeled_masked_loader, unlabeled_unmasked_loader)
        labeled_batches = zip(labeled_masked_loader, labeled_unmasked_loader)

        # Unlabeled data
        for batch_idx, ((x_u_masked, _), (x_u_unmasked, _)) in enumerate(unlabeled_batches):
            x_u_masked = x_u_masked.to(device)
            x_u_unmasked = x_u_unmasked.to(device)

            optimizer.zero_grad()
            
            # masked and unmasked processing
            mu_u_masked, logvar_u_masked, x_hat_u_masked, _ = pcvae(x_u_masked)
            mu_u_unmasked, logvar_u_unmasked, x_hat_u_unmasked, _ = pcvae(x_u_unmasked)
            
            recon_loss_u = criterion_recon(x_hat_u_masked, x_u_unmasked)
            kl_loss_u = kl_divergence(mu_u_masked, logvar_u_masked)
            matching_loss = kl_divergence(mu_u_masked, logvar_u_masked) - kl_divergence(mu_u_unmasked, logvar_u_unmasked)
            
            total_unlabeled_loss = recon_loss_u + beta * kl_loss_u + lambda_ * matching_loss
            total_unlabeled_loss.backward()
            optimizer.step()

        # Labeled data
        for batch_idx, ((x_l_masked, y_l), (x_l_unmasked, _)) in enumerate(labeled_batches):
            x_l_masked = x_l_masked.to(device)
            x_l_unmasked = x_l_unmasked.to(device)
            y_l = y_l.to(device)

            optimizer.zero_grad()
            
            mu_l_masked, logvar_l_masked, x_hat_l_masked, logits = pcvae(x_l_masked)
            mu_l_unmasked, logvar_l_unmasked, x_hat_l_unmasked, _ = pcvae(x_l_unmasked)

            recon_loss_l = criterion_recon(x_hat_l_masked, x_l_unmasked)
            kl_loss_l = kl_divergence(mu_l_masked, logvar_l_masked)
            class_loss = criterion_class(logits, y_l)
            matching_loss_l = kl_divergence(mu_l_masked, logvar_l_masked) - kl_divergence(mu_l_unmasked, logvar_l_unmasked)
            
            total_labeled_loss = recon_loss_l + beta * kl_loss_l + lambda_ * class_loss + matching_loss_l
            total_labeled_loss.backward()
            optimizer.step()

        # Log the losses
        writer.add_scalar('Reconstruction Loss (Unlabeled)', recon_loss_u, epoch)
        writer.add_scalar('KL Loss (Unlabeled)', kl_loss_u, epoch)
        writer.add_scalar('Posterior Matching Loss (Unlabeled)', matching_loss, epoch)
        writer.add_scalar('Total Unlabeled Loss', total_unlabeled_loss, epoch)

        writer.add_scalar('Reconstruction Loss (Labeled)', recon_loss_l, epoch)
        writer.add_scalar('KL Loss (Labeled)', kl_loss_l, epoch)
        writer.add_scalar('Classification Loss', class_loss, epoch)
        writer.add_scalar('Posterior Matching Loss (Labeled)', matching_loss_l, epoch)
        writer.add_scalar('Total Labeled Loss', total_labeled_loss, epoch)

        progress.set_description(f'Epoch: {epoch} | Total Loss: {total_unlabeled_loss + total_labeled_loss}')
    
    return pcvae

def main(config=None):
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