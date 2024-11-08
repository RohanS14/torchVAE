import sys
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import tqdm
import argparse
import json
import itertools

from models.vae import VariationalAutoencoder
from models.logreg import LogisticRegression
from models.pcvae import PredictionConstrainedVAE

def kl_divergence(mu, logvar):
    """
    Compute the KL divergence between a normal distribution with mean mu and log variance logvar
    and a standard normal distribution.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def trainVAE(vae, data, beta, epochs, lr, run_name, device='cuda'):
    recon_losses = []
    kl_losses = []
    
    print(f"Logging to {run_name}")
    
    writer = SummaryWriter(log_dir=f'./logs/{run_name}')
    
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    progress = tqdm.trange(epochs)
    
    for epoch in progress:
        print(epoch)
        for x, _ in data:
            x = x.to(device) # GPU
            
            opt.zero_grad()
            mu_z, logvar_z, x_hat = vae(x)
            
            kl_loss = kl_divergence(mu_z, logvar_z)
            
            #recon_loss = ((x - x_hat)**2).sum()
            recon_loss = 0 
            
            loss = recon_loss + beta * kl_loss
            loss.backward()
            
            recon_losses.append(recon_loss)
            kl_losses.append(kl_loss)
            opt.step()
        progress.set_description(f'Loss: {loss}, Recon: {recon_loss}, KL: {kl_loss}')
        
        # Log the losses to TensorBoard
        writer.add_scalar('Reconstruction Loss', recon_loss, epoch)
        writer.add_scalar('KL Divergence Loss', kl_loss, epoch)
        writer.add_scalar('Total Loss', loss, epoch)
    
    return recon_losses, kl_losses, vae

def validate(model, dataLoader, device, input_size, loss_fn):
    model.eval()
    accurate = 0
    total = 0
    dl_losses = []

    with torch.no_grad():
        for images,labels in dataLoader:
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)
    
            output = model(images)
            _,predicted = torch.max(output.data, 1)
    
            # compute loss
            dl_losses.append(loss_fn(output, labels).item())
            
            # total labels
            total += labels.size(0)
            
            # total correct predictions
            accurate += (predicted == labels).sum().item()
    
    accuracy_score = 100 * accurate/total
    return np.mean(dl_losses), accuracy_score

def train_logreg(model, train_dataLoader, valid_dataLoader, device, optimizer, loss_fn, \
                 learning_rate, num_epochs, input_size, regularization, lambda_, stop_100, run_name):
    
    # comment out code to return loss and acc for now
    # train_losses = []
    # val_losses = []
    # val_accs = []
    # train_accs = []
    
    # summary writer with custom run name based on configs and time
    print(f"Logging to {run_name}")
    writer = SummaryWriter(log_dir=f'./logs/{run_name}')
    
    progress = tqdm.trange(num_epochs)
    
    for epoch in progress:
        model.train()
        accurate = 0  
        total = 0
        
        print("Epoch", epoch)
        for images, labels in train_dataLoader:
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            train_loss = loss_fn(output, labels)
            
            # Adding Regularization
            if regularization == 'l2':
                l2_reg = lambda_ * sum(p.pow(2.0).sum() for p in model.parameters())
                train_loss += l2_reg
            elif regularization == 'l1':
                l1_reg = lambda_ * sum(p.abs().sum() for p in model.parameters())
                train_loss += l1_reg
            
            train_loss.backward()
            optimizer.step()
            # comment out code to return loss
            # train_losses.append(train_loss.item())
            
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            accurate += (predicted == labels).sum().item()
        
        train_acc = 100 * accurate / total
        valid_loss, valid_acc = validate(model, valid_dataLoader, device, input_size, loss_fn)
        
        progress.set_description(f'Train Loss: {train_loss.item():.4f}. Train Accuracy: {train_acc:.4f}. Valid Loss: {valid_loss:.4f}. Valid Accuracy: {valid_acc:.4f}')
        
        # comment out code to return loss and acc for now
        # train_accs.append(train_acc)
        # val_losses.append(valid_loss)
        # val_accs.append(valid_acc)
        
        # Log the losses to TensorBoard
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Training Accuracy', train_acc, epoch)
        writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('Valid Accuracy', valid_acc, epoch)
        
        # Early stopping when train accuracy reaches 100%
        if stop_100 and train_acc >= 100:
            break
        
    print('Final Train Accuracy:', train_acc)
    print('Final Val Accuracy:', valid_acc)
    
    # comment out code to return loss and acc for now
    # return train_accs, val_accs, train_losses, val_losses, model
    return model

def trainPCVAE(pcvae, masked_data_loader, unmasked_data_loader, labeled_data_loader, epochs=20, lr=0.001, beta=1, 
               lambda_=1, l_weight=1, u_weight=1, run_name="default", device='cuda'):
    
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

        num_batches = len(masked_data_loader)
        
        # iterate through batches
        for batch_idx, ((x_masked, _), (x_unmasked, _)) in enumerate(zip(masked_data_loader, unmasked_data_loader)):
            x_masked = x_masked.to(device)
            x_unmasked = x_unmasked.to(device)

            optimizer.zero_grad()
            
            # process masked data
            mu_masked, logvar_masked, x_hat_masked, _ = pcvae(x_masked)
            
            # process unmasked data (target for posterior matching)
            mu_unmasked, logvar_unmasked, x_hat_unmasked, _ = pcvae(x_unmasked)
            
            # Reconstruction loss with unmasked data
            recon_loss_masked = criterion_recon(x_hat_masked, x_unmasked)
            kl_loss_masked = kl_divergence(mu_masked, logvar_masked)
            loss_masked = recon_loss_masked + beta * kl_loss_masked

            # Posterior matching: KL divergence between masked and unmasked posteriors
            matching_loss = kl_divergence(mu_masked, logvar_masked) - kl_divergence(mu_unmasked, logvar_unmasked)

            # Final loss
            total_loss = loss_masked + lambda_ * matching_loss
            total_loss.backward()
            optimizer.step()

        # Log the losses
        writer.add_scalar('Reconstruction Loss', recon_loss_masked, epoch)
        writer.add_scalar('KL Loss', kl_loss_masked, epoch)
        writer.add_scalar('Posterior Matching Loss', matching_loss, epoch)
        writer.add_scalar('Total Loss', total_loss, epoch)

        progress.set_description(f'Total Loss: {total_loss}, Recon: {recon_loss_masked}, KL: {kl_loss_masked}, Matching: {matching_loss}')
    
    return pcvae
