import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
#import #wandb
#TODO: future : inpaint, colorization, low rez-> high rez
import argparse
import json 


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to the config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

from torch.utils.data import Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

root_path = '/cs/cs152/individual/ishita/checkpoints/'
if root_path not in sys.path:
    sys.path.insert(1, root_path)
from models.pcvae import PredictionConstrainedVAE
from utils.vistools import plotPCAdist, stackedHist
from masked_pcvae.train_pcvae_blackout import loadData

def kl_divergence_posterior(post1_mu, post1_logvar, post2_mu, post2_logvar): 
    
    kl = 0.5 * (post2_logvar - post1_logvar + (torch.exp(post1_logvar) + (post1_mu - post2_mu)**2) / torch.exp(post2_logvar) - 1) 
    return kl.sum(dim=1).mean().clone()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
masked_vae = PredictionConstrainedVAE(architecture=config['model']['architecture'], latent_dims=config['model']['latent_dims'], num_classes=config['model']['num_classes'])
unmasked_vae = PredictionConstrainedVAE(architecture=config['model']['architecture'], latent_dims=config['model']['latent_dims'], num_classes=config['model']['num_classes'])
m_file_path = '/cs/cs152/individual/ishita/checkpoints/pcvae_weighted-PCvae-MNIST-linear-20-1-0.001-2024-09-25-10-33-01'
u_file_path = '/cs/cs152/individual/ishita/checkpoints/pcvae_weighted-PCvae-MNIST-linear-20-1-0.001-2024-09-25-10-33-01'

state1 = torch.load(m_file_path)
masked_vae.load_state_dict(state1)
masked_vae.eval()
masked_vae.to(device)

state2 = torch.load(u_file_path)
unmasked_vae.load_state_dict(state2)
unmasked_vae.eval()
unmasked_vae.to(device)

DATASET_NAME = "MNIST"
NUM_TRAIN = config['dataset']['num_train']
BATCH_SIZE = config['training']['batch_size']


## TODO: add logging
# #wandb.init(project="posterior-matching", name="kl-loss-training", config={
#     "num_epochs": config['training']['num_epochs'],
#     "batch_size": config['training']['batch_size'],
#     "learning_rate": config['training']['learning_rate']
# })

masked_data_l, masked_data_u, unmasked_data_l,unmasked_data_u = loadData(DATASET_NAME, NUM_TRAIN)
masked_data_l = DataLoader(masked_data_l, batch_size=BATCH_SIZE, shuffle=False)
masked_data_u = DataLoader(masked_data_u, batch_size=BATCH_SIZE, shuffle=False) 
unmasked_data_l = DataLoader(unmasked_data_l, batch_size=BATCH_SIZE, shuffle=False)
unmasked_data_u = DataLoader(unmasked_data_u, batch_size=BATCH_SIZE, shuffle=False) 

# Freeze parameters of the unmasked encoder
for param in unmasked_vae.encoder.parameters():
    param.requires_grad = False

# added this
for param in unmasked_vae.decoder.parameters():
    param.requires_grad = False
    
# Define optimizer for masked VAE (we only optimize the masked VAE's encoder)
optimizer = optim.Adam(masked_vae.encoder.parameters(), lr=config['training']['learning_rate'])
# Training loop for posterior matching
num_epochs = config['training']['num_epochs']
broad_kl_loss = []
for epoch in range(num_epochs):
    kl_losses = []
    # l_masked = []
    # l_unmasked = [] 
    # for batch_data_m in masked_data_u:
    #     batch_data_m= batch_data_m[0]
    #     batch_data_m = batch_data_m.to(device)
    #     masked_mu, masked_logvar = masked_vae.encoder(batch_data_m)
    #     l_masked.append((masked_mu,masked_logvar))
    # for batch_data_u in unmasked_data_u:
    #     batch_data_u = batch_data_u[0]
    #     batch_data_u = batch_data_u.to(device)
        
    #     # added this
    #     with torch.no_grad():
    #         unmasked_mu, unmasked_logvar = unmasked_vae.encoder(batch_data_u)

    #     l_unmasked.append((unmasked_mu, unmasked_logvar))

    # for i in range(0,len(l_unmasked)):
    #     # Compute KL divergence between the two latent distributions
    #     masked_mu, masked_logvar = l_masked[i]
    #     unmasked_mu, unmasked_logvar = l_unmasked[i]
    #     #print(masked_mu,unmasked_mu)
        
    #     kl_loss = kl_divergence_posterior(masked_mu, masked_logvar, unmasked_mu, unmasked_logvar)
    #     kl_losses.append(kl_loss.item())
    #     # Backpropagation and optimization
    #     optimizer.zero_grad()
    #     kl_loss.backward()
    #     optimizer.step()
    
    for (m_batch, u_batch) in zip(masked_data_u, unmasked_data_u):
        m_batch = m_batch[0].to(device)
        u_batch = u_batch[0].to(device)

        masked_mu, masked_logvar = masked_vae.encoder(m_batch)

        # Freeze unmasked part
        with torch.no_grad():
            unmasked_mu, unmasked_logvar = unmasked_vae.encoder(u_batch)

        kl_loss = kl_divergence_posterior(masked_mu, masked_logvar,
                                        unmasked_mu, unmasked_logvar)

        optimizer.zero_grad()
        kl_loss.backward()
        optimizer.step()
        
        kl_losses.append(kl_loss.item())
    # Logging the KL loss per epoch
    avg_kl_loss = np.mean(kl_losses)
    broad_kl_loss.append(avg_kl_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, KL Loss: {avg_kl_loss}")
    print(f"average KL Loss: {avg_kl_loss}")
    #wandb.log({"KL Loss": avg_kl_loss, "epoch": epoch + 1})
    # # Visualize latent space every 10 epochs
    # if epoch % 10 == 0:
    # #TODO: this code does not have different or correct vars for unmasked or masked 
    #     # Collect latent distributions for visualization
    #     z_batches, y_batches = [], []
    #     for batch_data, labels in masked_data_l:
    #         batch_data = batch_data.to(device)
    #         with torch.no_grad():
    #             latent_z, _ = masked_vae.encoder(batch_data)
    #         z_batches.append(latent_z)
    #         y_batches.append(labels)
    #     # Convert lists to tensors for visualization
    #     z_batches = torch.cat(z_batches).cpu().numpy()
    #     y_batches = torch.cat(y_batches).cpu().numpy()
    #     # Plot latent distribution with PCA
    #     plotPCAdist(z_batches, y_batches)
    #     stackedHist(z_batches, y_batches)

torch.save(masked_vae.state_dict(), "saved_masked_rohan.sav")
torch.save(unmasked_vae.state_dict(), "saved_unmasked_rohan.sav")

import matplotlib.pyplot as plt
print(len(broad_kl_loss))

plt.plot( broad_kl_loss, label='KL Loss')
plt.xlabel('Epochs')
plt.ylabel('KL Loss')
plt.title('KL Loss during Posterior Matching Training')
plt.legend()
plt.grid(True)
#wandb.log({"KL Loss Plot": #wandb.Image(plt)})
#wandb.finish()
save_path = "/cs/cs152/individual/ishita/kl_loss_plot.png"
plt.savefig(save_path)

print("Posterior matching training complete!")
