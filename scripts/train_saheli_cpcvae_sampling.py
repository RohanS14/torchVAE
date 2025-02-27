import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch.utils.data import Dataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import wandb

wandb.init(project="torchVAE", name="saheli-conv-run-kl-log-sampling" ,entity="hopelab-hmc")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(ConvEncoder, self).__init__()
        init_channels = 16
        self.conv1 = nn.Conv2d(1, init_channels, kernel_size=3, padding=1, stride=2)
        self.skip_conv1 = nn.Conv2d(1, init_channels, kernel_size=3, padding=1, stride=2)
        self.conv2 =  nn.Conv2d(init_channels, init_channels*2, kernel_size=3, padding=1, stride=2)
        self.skip_conv2 =  nn.Conv2d(init_channels, init_channels*2, kernel_size=3, padding=1, stride=2)
        self.conv3 =  nn.Conv2d(init_channels*2, init_channels*4, kernel_size=3, padding=1, stride=2)
        self.skip_conv3 =  nn.Conv2d(init_channels*2, init_channels*4, kernel_size=3, padding=1, stride=2)
        self.activation = nn.ReLU()

        self.linear_mu = nn.Linear(init_channels*4*4*4,latent_dims)
        self.linear_sigma = nn.Linear(init_channels*4*4*4,latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.activation(self.conv1(x))+self.skip_conv1(x)
        x = self.activation(self.conv2(x))+self.skip_conv2(x)
        x = self.activation(self.conv3(x))+self.skip_conv3(x)
        x = torch.flatten(x, start_dim=1)
        mu =  self.linear_mu(x)
        logvar = self.linear_sigma(x)
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().pow(2))
        sigma = torch.exp(logvar/2)
        return mu, sigma
    
    def sample(self, mu, sigma):
        N = torch.distributions.Normal(0, 1)
        z = mu + sigma * self.N.sample(mu.shape)
        return z
    
class ConvDecoder(nn.Module):
    """Convolutional decoder for VAE on MNIST with Gaussian distribution."""

    def __init__(self, latent_dims):
        super(ConvDecoder, self).__init__()
        
        init_channels = 16
        self.distn = "norm"
        
        self.linear = nn.Linear(latent_dims, init_channels*4*4*4)
        self.convT1 = nn.ConvTranspose2d(init_channels*4,init_channels*2,kernel_size=2,stride=2)
        self.skip_convT1 = nn.ConvTranspose2d(init_channels*4,init_channels*2,kernel_size=2,stride=2)
        self.convT2 = nn.ConvTranspose2d(init_channels*2,init_channels,kernel_size=2,stride=2)
        self.skip_convT2 = nn.ConvTranspose2d(init_channels*2,init_channels,kernel_size=2,stride=2)
        self.convT3 = nn.ConvTranspose2d(init_channels,1,kernel_size=2,stride=2,padding=2)
        self.skip_convT3 = nn.ConvTranspose2d(init_channels,1,kernel_size=2,stride=2,padding=2)
        self.activation = nn.ReLU()
        
        self.fixed_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        
        init_channels = 16
        
        z = self.activation(self.linear(z))
        z = z.reshape((-1, init_channels * 4, 4, 4))
        z = self.activation(self.convT1(z)) + self.skip_convT1(z)
        z = self.activation(self.convT2(z)) + self.skip_convT2(z)
        z = self.activation(self.convT3(z)) + self.skip_convT3(z)

        mu = torch.sigmoid(z)
        logvar = self.fixed_scale.expand_as(mu)
        return mu, logvar

    def sample(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        normal_dist = torch.distributions.Normal(mu, sigma)
        xhat = normal_dist.rsample()
        return xhat.reshape((-1, 1, 28, 28))
    

class LogisticRegression(nn.Module):
    def __init__(self, latent_dims, classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(latent_dims, classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x=torch.flatten(x,start_dim=1)
        return self.softmax(self.linear(x))
    

class VAEClassifer(nn.Module):
    def __init__(self, latent_dims,classes):
        super(VAEClassifer, self).__init__()
        self.encoder = ConvEncoder(latent_dims)
        self.decoder = ConvDecoder(latent_dims)
        self.classifier = LogisticRegression(latent_dims,classes)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.encoder.sample(mu, sigma)
        mu, logvar = self.decoder(z)
        xhat = self.decoder.sample(mu, logvar)
        return xhat, self.classifier(z)

# Log confusion matrix to wandb
def log_confusion_matrix(cm, epoch):
    class_names = list(range(10))
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix")
    wandb.log({f"Confusion Matrix/{epoch}": wandb.Image(fig)})
    plt.close(fig)

def log_label_dist(preds, epoch):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.bar(list(range(10)), preds)
    plt.title("Label Distribution")
    wandb.log({f"Label Distribution/{epoch}": wandb.Image(fig)})
    plt.close(fig)

def train(VAE, unlabelled_train_data, labelled_train_data, valid_data, epochs=20):
    opt = torch.optim.Adam(VAE.parameters(), weight_decay=1e-2)
    pred_loss_func = nn.CrossEntropyLoss()
    highest_accuracy = 0
    
    for epoch in tqdm(range(epochs)):
        avg_loss_recon_u, avg_loss_kl_u = [], []
        avg_loss_recon_l, avg_loss_kl_l = [], []
        avg_loss_recon, avg_loss_kl = [], []
        avg_loss, avg_loss_vae, avg_loss_classifer = [], [], []
        avg_label_consistency, avg_loss_consistency = [], []
        avg_hist = np.array([0.0]*10)
        accuracy, num_data = 0, 0

        for unlabelled, labelled in zip(unlabelled_train_data, labelled_train_data):
            unlabelled_x, _ = unlabelled
            labelled_x, y = labelled
            unlabelled_x, labelled_x, y = unlabelled_x.to(device), labelled_x.to(device), y.to(device)
            opt.zero_grad()
            
            # unlabelled 
            x_hat, y_pred = VAE(unlabelled_x)
            x_hat_hat, y_hat_pred = VAE(x_hat)
            total_kl = VAE.encoder.kl
             
            loss_recon_u = ((unlabelled_x - x_hat_hat)**2).sum() + ((unlabelled_x - x_hat)**2).sum()
            loss_kl_u = VAE.encoder.kl
            loss_vae = (loss_recon_u + loss_kl_u) / (len(unlabelled_x) + len(labelled_x))
            
            consistency_loss = 20 * pred_loss_func(y_hat_pred, y_pred)
            histogram = torch.sum(y_hat_pred, dim=0)
            
            # labelled
            x_hat, y_pred = VAE(labelled_x)
            x_hat_hat, y_hat_pred = VAE(x_hat)
            
            loss_recon_l = ((labelled_x - x_hat_hat)**2).sum() + ((labelled_x - x_hat)**2).sum()
            loss_kl_l = VAE.encoder.kl
            loss_vae += (loss_recon_l + loss_kl_l) / (len(unlabelled_x) + len(labelled_x))
            
            consistency_loss += 20 * pred_loss_func(y_hat_pred, y_pred)
            histogram += torch.sum(y_hat_pred, dim=0)
            loss_classifer = 100 * pred_loss_func(y_hat_pred, y)
            avg_hist += histogram.cpu().detach().numpy()
            histogram = histogram / torch.sum(histogram)
            aggregate_label_consistency = 3200 * pred_loss_func(histogram, torch.tensor([.1]*10).to(device))
            loss = loss_vae + loss_classifer + consistency_loss + aggregate_label_consistency
            loss.backward()
            opt.step()
            
            avg_loss.append(loss.item())
            avg_loss_vae.append(loss_vae.item())
            avg_loss_classifer.append(loss_classifer.item())
            avg_label_consistency.append(aggregate_label_consistency.item())
            avg_loss_consistency.append(consistency_loss.item())
            
            avg_loss_recon.append(loss_recon_u.item() + loss_recon_l.item())
            avg_loss_kl.append(loss_kl_u.item() + loss_kl_l.item())
            avg_loss_recon_u.append(loss_recon_u.item())
            avg_loss_kl_u.append(loss_kl_u.item())
            avg_loss_recon_l.append(loss_recon_l.item())
            avg_loss_kl_l.append(loss_kl_l.item())
            
            accuracy += sum(torch.argmax(y_hat_pred, dim=1) == y).item()
            num_data += len(y)

        wandb.log({
            "Loss/train": np.mean(avg_loss),
            "Loss_VAE/train": np.mean(avg_loss_vae),
            "Loss_Classifier/train": np.mean(avg_loss_classifer),
            "Loss_Consistency/train": np.mean(avg_loss_consistency),
            "Aggregate_Label_Consistency/train": np.mean(avg_label_consistency),
            "Loss_Recon/train": np.mean(avg_loss_recon),
            "Loss_KL/train": np.mean(avg_loss_kl),
            "Loss_Recon_U/train": np.mean(avg_loss_recon_u),
            "Loss_KL_U/train": np.mean(avg_loss_kl_u),
            "Loss_Recon_L/train": np.mean(avg_loss_recon_l),
            "Loss_KL_L/train": np.mean(avg_loss_kl_l),
            
            "Accuracy/train": accuracy / num_data
        })
        
        # log_label_dist(avg_hist / sum(avg_hist), epoch)
        
        if epoch % 100 == 0:
            with torch.no_grad():
                highest_accuracy = max(highest_accuracy, validate(VAE, valid_data, epoch, highest_accuracy))
    return VAE

def validate(VAE, data, epoch, highest_accuracy):
    avg_loss_vae, avg_loss_classifer = [], []
    accuracy, num_data = 0, 0
    # cm = np.zeros((10,10))
    pred_loss_func = nn.CrossEntropyLoss()
    
    for x, y in data:
        x, y = x.to(device), y.to(device)
        x_hat, y_pred = VAE(x)
        x_hat_hat, y_hat_pred = VAE(x_hat)
        loss_vae = (((x - x_hat)**2).sum() + ((x - x_hat_hat)**2).sum() + VAE.encoder.kl) / batch_size
        loss_classifer = 50 * pred_loss_func(y_hat_pred, y) + ((y_pred - y_hat_pred)**2).sum()
        avg_loss_vae.append(loss_vae.item())
        avg_loss_classifer.append(loss_classifer.item())
        accuracy += sum(torch.argmax(y_hat_pred, dim=1) == y).item()
        num_data += len(y)
        
        # cm += confusion_matrix(y.cpu().numpy(), torch.argmax(y_hat_pred, dim=1).cpu().numpy())
    
    wandb.log({
        "Loss/valid": np.mean(avg_loss_vae) + np.mean(avg_loss_classifer),
        "Loss_VAE/valid": np.mean(avg_loss_vae),
        "Loss_Classifier/valid": np.mean(avg_loss_classifer),
        "Accuracy/valid": accuracy / num_data
    })
    
    # log_confusion_matrix(cm, epoch)
    
    if highest_accuracy and accuracy / num_data > highest_accuracy:
        torch.save(VAE.state_dict(), "mnist_model100labels_aggregate_label_long.sav")
        highest_accuracy = accuracy / num_data
    return accuracy / num_data

# Training

batch_size = 50
valid_batch_size = 6000

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

labelled_train_data = torch.utils.data.DataLoader(SimpleDataset(labelled_images, labelled_labels),batch_size=batch_size,shuffle=True)
unlabelled_train_data = torch.utils.data.DataLoader(SimpleDataset(unlabelled_images, unlabelled_labels),batch_size=batch_size,shuffle=True)
valid_data = torch.utils.data.DataLoader(SimpleDataset(valid_images, valid_labels),batch_size=valid_batch_size,shuffle=True)

test_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./mnist_data',
               transform=torchvision.transforms.ToTensor(),
               download=True, train=False),
        batch_size=1000,
        shuffle=True)

latent_dims = 10
VAE = VAEClassifer(latent_dims,classes=10).to(device) # GPU
# VAE = train(VAE, unlabelled_train_data, labelled_train_data, valid_data, epochs=20)
VAE = train(VAE, unlabelled_train_data, labelled_train_data, valid_data,epochs=20000)

# loss_weights={"consistency": 40, "classifer": 100, "aggregate": 3200}
# ,sav_file="mnist_model_100labels10.sav"