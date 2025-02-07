import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
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

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        init_channels = 16
        self.linear = nn.Linear(latent_dims, init_channels*4*4*4)
        self.convT1 = nn.ConvTranspose2d(init_channels*4,init_channels*2,kernel_size=2,stride=2)
        self.skip_convT1 = nn.ConvTranspose2d(init_channels*4,init_channels*2,kernel_size=2,stride=2)
        self.convT2 = nn.ConvTranspose2d(init_channels*2,init_channels,kernel_size=2,stride=2)
        self.skip_convT2 = nn.ConvTranspose2d(init_channels*2,init_channels,kernel_size=2,stride=2)
        self.convT3 = nn.ConvTranspose2d(init_channels,1,kernel_size=2,stride=2,padding=2)
        self.skip_convT3 = nn.ConvTranspose2d(init_channels,1,kernel_size=2,stride=2,padding=2)
        self.activation = nn.ReLU()
        self.fixed_scale = nn.Parameter(torch.tensor(0.))

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

    def forward(self, z):
        init_channels = 16
        z = self.activation(self.linear(z))
        z = z.reshape((-1, init_channels*4, 4, 4))
        z = self.activation(self.convT1(z))+self.skip_convT1(z)
        z = self.activation(self.convT2(z))+self.skip_convT2(z)
        
        mu = self.activation(self.convT3(z))+self.skip_convT3(z)
        logvar = self.fixed_scale.expand_as(mu)
        sigma = torch.exp(logvar / 2)
        
        return mu, sigma

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
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
        self.classifier = LogisticRegression(latent_dims,classes)

    def forward(self, x):
        mu_z, sigma_z = self.encoder(x)
        z = mu_z + sigma_z*self.encoder.N.sample(mu_z.shape)
        mu_x, sigma_x = self.decoder(z)
        x_hat = mu_x + sigma_x*self.decoder.N.sample(mu_x.shape)
        return z, x_hat, self.classifier(z)