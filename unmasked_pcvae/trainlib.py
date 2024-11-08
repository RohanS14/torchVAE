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
            
            recon_loss = ((x - x_hat)**2).sum()
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

    with torch.no_gradvalid_dataLoader():
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

def trainPCVAE(pcvae, unlabeled_data_loader, labeled_data_loader, epochs=20, lr=0.001, beta=1, \
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

        # Create iterators, cycling the smaller one
        if len(unlabeled_data_loader) < len(labeled_data_loader):
            u_iter = itertools.cycle(unlabeled_data_loader)
            l_iter = iter(labeled_data_loader)
        else:
            l_iter = itertools.cycle(labeled_data_loader)
            u_iter = iter(unlabeled_data_loader)
        
        num_batches = max(len(unlabeled_data_loader), len(labeled_data_loader))
        
        # iterate through batches
        for _ in range(num_batches):
            x_u, _ = next(u_iter)
            x_l, y_l = next(l_iter)

            x_u = x_u.to(device)
            x_l, y_l = x_l.to(device), y_l.to(device)

            optimizer.zero_grad()
            
            # process unlabeled data
            mu_u, logvar_u, x_hat_u, _ = pcvae(x_u)
            
            recon_loss_u = criterion_recon(x_hat_u, x_u)
            kl_loss_u = kl_divergence(mu_u, logvar_u)
            
            loss_u = recon_loss_u + beta * kl_loss_u

            # process labeled data
            mu_l, logvar_l, x_hat_l, logits = pcvae(x_l)
            
            recon_loss_l = criterion_recon(x_hat_l, x_l)
            kl_loss_l = kl_divergence(mu_l, logvar_l)
            class_loss = criterion_class(logits, y_l)
            
            loss_l = recon_loss_l + beta * kl_loss_l + lambda_ * class_loss

            # combine all losses
            total_loss = u_weight * loss_u + l_weight * loss_l
            total_loss.backward()
            optimizer.step()
            
        # Log the losses
        with torch.no_grad():
            recon_loss = u_weight * recon_loss_u + l_weight * recon_loss_l
            kl_loss = u_weight * kl_loss_u +l_weight * kl_loss_l
            writer.add_scalar('Reconstruction Loss', recon_loss, epoch)
            writer.add_scalar('KL Loss', kl_loss, epoch)
            writer.add_scalar('Classifier Loss', class_loss, epoch)
            writer.add_scalar('Total Loss', total_loss, epoch)
        
        progress.set_description(f'Total Loss: {total_loss}, Recon: {recon_loss}, KL: {kl_loss}, Classifier: {class_loss}')
    
    return pcvae