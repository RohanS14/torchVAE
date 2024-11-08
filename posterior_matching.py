import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset

class SubsetWrapper(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Retrieve data and label from subset
        data = self.subset[idx]
        
        # Ensure data and label are converted to tensors
        if not isinstance(data, torch.Tensor):
            print(data.shape)
            data = torch.tensor(data, dtype=torch.float32)  # Convert data to float tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)  # Convert label to long tensor

        return data, label


# Set root path for importing modules
root_path = '/cs/cs152/individual/ishita/checkpoints/'
if root_path not in sys.path:
    sys.path.insert(1, root_path)
# Import custom modules
from models.pcvae import PredictionConstrainedVAE
from utils.vistools import plotPCAdist, stackedHist
from masked_pcvae.train_pcvae_blackout import loadData
def kl_divergence_posterior(post1_mu, post1_logvar, post2_mu, post2_logvar): # Direct calculation of KL divergence between two Gaussian posteriors # Here, post1 refers to q_ψ (masked) and post2 to q_θ (unmasked) 
    kl = 0.5 * (post2_logvar - post1_logvar + (torch.exp(post1_logvar) + (post1_mu - post2_mu)**2) / torch.exp(post2_logvar) - 1) 
    return kl.sum(dim=1).mean()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load the trained PCVAE
masked_vae = PredictionConstrainedVAE(architecture="linear", latent_dims=20, num_classes=10)
unmasked_vae = PredictionConstrainedVAE(architecture="linear", latent_dims=20, num_classes=10)
m_file_path = '/cs/cs152/individual/ishita/checkpoints/pcvae_weighted-PCvae-MNIST-linear-20-1-0.001-2024-09-25-10-33-01'
u_file_path = '/cs/cs152/individual/ishita/checkpoints/pcvae_weighted-PCvae-MNIST-linear-20-1-0.001-2024-10-07-20-34-09'
state1 = torch.load(m_file_path)
masked_vae.load_state_dict(state1)
masked_vae.eval()
masked_vae.to(device)

state2 = torch.load(u_file_path)
unmasked_vae.load_state_dict(state2)
unmasked_vae.eval()
unmasked_vae.to(device)
# Load Data
DATASET_NAME = "MNIST"
NUM_TRAIN = 100
BATCH_SIZE = 128
data_l, data_u = loadData(DATASET_NAME, NUM_TRAIN)
dataLoader_l = DataLoader(SubsetWrapper(data_l), batch_size=BATCH_SIZE, shuffle=True)
dataLoader_u = DataLoader(SubsetWrapper(data_u), batch_size=BATCH_SIZE, shuffle=False)

# Freeze parameters of the unmasked encoder
for param in unmasked_vae.encoder.parameters():
    param.requires_grad = False
# Define optimizer for masked VAE (we only optimize the masked VAE's encoder)
optimizer = optim.Adam(masked_vae.encoder.parameters(), lr=1e-4)
# Training loop for posterior matching
# num_epochs = 100
# for epoch in range(num_epochs):
#     kl_losses = []
#     for batch_data in dataLoader_u:
#         batch_data = batch_data.to(device)
#         # Forward pass through the encoders to get latent means and log variances
#         masked_mu, masked_logvar = masked_vae.encoder(batch_data)
#         unmasked_mu, unmasked_logvar = unmasked_vae.encoder(batch_data)
#         # Compute KL divergence between the two latent distributions
#         kl_loss = kl_divergence(masked_mu, masked_logvar, unmasked_mu, unmasked_logvar)
#         kl_losses.append(kl_loss.item())
#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         kl_loss.backward()
#         optimizer.step()
#     # Logging the KL loss per epoch
#     avg_kl_loss = np.mean(kl_losses)
#     print(f"Epoch {epoch+1}/{num_epochs}, KL Loss: {avg_kl_loss}")
#     # Visualize latent space every 10 epochs
#     if epoch % 10 == 0:
#         # Collect latent distributions for visualization
#         z_batches, y_batches = [], []
#         for batch_data, labels in dataLoader_u:
#             batch_data = batch_data.to(device)
#             with torch.no_grad():
#                 latent_z, _ = masked_vae.encoder(batch_data)
#             z_batches.append(latent_z)
#             y_batches.append(labels)
#         # Convert lists to tensors for visualization
#         z_batches = torch.cat(z_batches).cpu().numpy()
#         y_batches = torch.cat(y_batches).cpu().numpy()
#         # Plot latent distribution with PCA
#         plotPCAdist(z_batches, y_batches)
#         stackedHist(z_batches, y_batches)
# print("Posterior matching training complete!")





# Training loop for posterior matching
num_epochs = 100
for epoch in range(num_epochs):
    # total_kl_loss = 0  # Accumulate KL losses across all batches
    for batch_data in dataLoader_u:
        batch_data = batch_data.to(device)
        
        # Forward pass through both encoders to get latent means and log variances
        masked_mu, masked_logvar = masked_vae.encoder(batch_data)
        unmasked_mu, unmasked_logvar = unmasked_vae.encoder(batch_data)
        
        # Compute KL divergence between the two posterior distributions
        kl_loss = kl_divergence_posterior(masked_mu, masked_logvar, unmasked_mu, unmasked_logvar)
        # total_kl_loss += kl_loss.item()  # Accumulate loss
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        kl_loss.backward()
        optimizer.step()

    # Logging the total KL loss per epoch
    print(f"Epoch {epoch+1}/{num_epochs}, KL Loss: {kl_loss}")

    # Visualize latent space every 10 epochs
    if epoch % 10 == 0:
        # Collect latent distributions for visualization
        z_batches, y_batches = [], []
        for batch_data, labels in dataLoader_u:
            batch_data = batch_data.to(device)
            with torch.no_grad():
                latent_z, _ = masked_vae.encoder(batch_data)
            z_batches.append(latent_z)
            y_batches.append(labels)

        # Convert lists to tensors for visualization
        z_batches = torch.cat(z_batches).cpu().numpy()
        y_batches = torch.cat(y_batches).cpu().numpy()

        # Plot latent distribution with PCA
        plotPCAdist(z_batches, y_batches)
        stackedHist(z_batches, y_batches)

print("Posterior matching training complete!")
