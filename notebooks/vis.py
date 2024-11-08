#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

root_path = '/cs/cs152/individual/rohan/semisupervisedVAE/'

if root_path not in sys.path:
    sys.path.insert(1, root_path)
    
from models.pcvae import PredictionConstrainedVAE
from utils.vistools import * 
from scripts.train_pcvae import loadData

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ### Load trained PCVAE

# In[3]:


pcvae = PredictionConstrainedVAE(architecture="linear", latent_dims=20, num_classes=10)


# In[6]:


saved_model = "/mnt/home/ijain/cs152/torchVAE/checkpoints/pcvae_weighted-PCvae-MNIST-linear-20-1-0.001-2024-09-25-10-33-01"


# In[7]:


state_dict = torch.load(saved_model)
pcvae.load_state_dict(state_dict)
pcvae.eval()
pcvae.to(device)


# In[8]:


pcvae.classifier


# ### Get data

# In[9]:


DATASET_NAME = "MNIST"
NUM_TRAIN = 100
BATCH_SIZE = 128
data_l, data_u = loadData(DATASET_NAME, NUM_TRAIN)
dataLoader_l = DataLoader(data_l, batch_size=BATCH_SIZE, shuffle=True)
dataLoader_u = DataLoader(data_u, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:





# In[10]:


class EncodedDataset(Dataset):
    def __init__(self, dataset, encode_fn, device):
        self.dataset = dataset
        self.encode_fn = encode_fn
        self.device = device
        self.encoded_images = []
        self.labels = []

        # Encode all images beforehand and store them
        for image, label in tqdm(dataset):
            encoded_image = self.encode_fn(image, device)
            self.encoded_images.append(encoded_image)
            self.labels.append(label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.encoded_images[idx], self.labels[idx]

def encode(x, device):
    with torch.no_grad():
        pcvae.encoder.eval()
        x = x.to(device)
        mean, logvar = pcvae.encoder(x.to(device))
        
        # Sample an image from the distribution
        z = pcvae.encoder.sample(mean, logvar)
        z = z.detach()
        
        return z
    
encoded_train_data = EncodedDataset(data_l, encode, device)
encoded_val_data = EncodedDataset(data_u, encode, device)
print(f'Encoded train data shape: {encoded_train_data[0][0].shape}')

train_dataLoader = DataLoader(encoded_train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_dataLoader = DataLoader(encoded_val_data, batch_size=BATCH_SIZE, shuffle=False)


# ### Compute accuracy of classifier 

# In[11]:


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


# In[12]:


validate(pcvae.classifier, train_dataLoader, device=device, input_size=20, loss_fn=nn.CrossEntropyLoss())


# In[13]:


validate(pcvae.classifier, valid_dataLoader, device=device, input_size=20, loss_fn=nn.CrossEntropyLoss())


# ### Visualize latent distribution

# In[14]:


z_batches, y_batches = get_latent(pcvae, dataLoader_u, num_batches=100, latent_dims=20, device='cuda')


# In[15]:


z_batches.shape


# In[16]:


y_batches.shape


# In[17]:


plotPCAdist(z_batches, y_batches)


# In[18]:


stackedHist(z_batches, y_batches)


# ### Generate image samples

# In[19]:


sample_random(pcvae, n=10)


# ## generate image representation w/o latent

# In[ ]:





# In[ ]:




