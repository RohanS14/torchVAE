
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class LinearEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(LinearEncoder, self).__init__()
        
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims) # outputs mu
        self.linear3 = nn.Linear(512, latent_dims) # outputs logvar
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        
        # self.kl = 0
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        logvar = self.linear3(x)
        
        # change this NOT
        # self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # sigma = torch.exp(logvar / 2)
        # return mu, sigma
        return mu, logvar

    def sample(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        z = mu + sigma*self.N.sample(mu.shape)
        return z
    
class LinearDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(LinearDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2_mu = nn.Linear(512, 784)
        self.fixed_scale = nn.Parameter(torch.tensor(0.)) # logvar
        
    def forward(self, z):
        h = F.relu(self.linear1(z))
        mu = torch.sigmoid(self.linear2_mu(h))
        # expand to match the shape of mu
        # sigma = torch.exp(self.fixed_scale.expand_as(mu) / 2)
        # return mu, sigma
        
        logvar = self.fixed_scale.expand_as(mu)
        return mu, logvar

    def sample(self, mu, logvar):
        # Using torch.distributions to create a normal distribution and sample from it
        # no need to expand due to broadcasting
        sigma = torch.exp(logvar / 2)
        normal_dist = torch.distributions.Normal(mu, sigma)
        xhat = normal_dist.rsample()
        # for now, specific to MNIST
        return xhat.reshape((-1, 1, 28, 28))

class ConvEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(ConvEncoder, self).__init__()
        
        init_channels = 64
        self.conv1 = nn.Conv2d(3, init_channels, kernel_size=3, padding=1, stride=2)
        self.skip_conv1 = nn.Conv2d(3, init_channels, kernel_size=3, padding=1, stride=2)
        self.conv2 =  nn.Conv2d(init_channels, init_channels*2, kernel_size=3, padding=1, stride=2)
        self.skip_conv2 =  nn.Conv2d(init_channels, init_channels*2, kernel_size=3, padding=1, stride=2)
        self.conv3 =  nn.Conv2d(init_channels*2, init_channels*4, kernel_size=3, padding=1, stride=2)
        self.skip_conv3 =  nn.Conv2d(init_channels*2, init_channels*4, kernel_size=3, padding=1, stride=2)
        self.activation = nn.ReLU()

        self.linear_mu = nn.Linear(init_channels*4*4*4,latent_dims) # outputs mu
        self.linear_logvar = nn.Linear(init_channels*4*4*4,latent_dims)  # outputs logvar
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        
        # self.kl = 0

    def forward(self, x):
        x = self.activation(self.conv1(x))+self.skip_conv1(x)
        x = self.activation(self.conv2(x))+self.skip_conv2(x)
        x = self.activation(self.conv3(x))+self.skip_conv3(x)
        x = torch.flatten(x, start_dim=1)
        mu =  self.linear_mu(x)
        logvar = self.linear_logvar(x)
        # self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # sigma = torch.exp(logvar / 2)
        # return mu, sigma
        return mu, logvar
    
    def sample(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        z = mu + sigma*self.N.sample(mu.shape)
        return z
    
class ConvDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(ConvDecoder, self).__init__()
        init_channels = 64
        self.linear = nn.Linear(latent_dims, init_channels*4*4*4)
        self.convT1 = nn.ConvTranspose2d(init_channels*4,init_channels*2,kernel_size=2,stride=2)
        self.skip_convT1 = nn.ConvTranspose2d(init_channels*4,init_channels*2,kernel_size=2,stride=2)
        self.convT2 = nn.ConvTranspose2d(init_channels*2,init_channels,kernel_size=2,stride=2)
        self.skip_convT2 = nn.ConvTranspose2d(init_channels*2,init_channels,kernel_size=2,stride=2)
        self.convT3 = nn.ConvTranspose2d(init_channels,3,kernel_size=2,stride=2)
        self.skip_convT3 = nn.ConvTranspose2d(init_channels,3,kernel_size=2,stride=2)
        self.activation = nn.ReLU()
        
        self.fixed_scale = nn.Parameter(torch.tensor(0.)) # logvar

    def forward(self, z):
        init_channels = 64
        z = self.activation(self.linear(z))
        z = z.reshape((-1, init_channels*4, 4, 4))
        z = self.activation(self.convT1(z))+self.skip_convT1(z)
        z = self.activation(self.convT2(z))+self.skip_convT2(z)
        z = self.activation(self.convT3(z))+self.skip_convT3(z)
        
        mu = torch.sigmoid(z)
        # sigma = torch.exp(self.fixed_scale.expand_as(mu) / 2)
        # return mu, sigma
        logvar = self.fixed_scale.expand_as(mu)
        return mu, logvar
    
    def sample(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        normal_dist = torch.distributions.Normal(mu, sigma)
        xhat = normal_dist.rsample()
        # for now, specific to CIFAR
        return xhat.reshape((-1, 3, 32, 32))

class VariationalAutoencoder(nn.Module):
    def __init__(self, architecture="linear", latent_dims=20):
        super(VariationalAutoencoder, self).__init__()
        
        if architecture == "linear":
            self.encoder = LinearEncoder(latent_dims)
            self.decoder = LinearDecoder(latent_dims)
        if architecture == "conv":
            self.encoder = ConvEncoder(latent_dims)
            self.decoder = ConvDecoder(latent_dims) 
    
    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = self.encoder.sample(mu_z, logvar_z)
        mu_x, logvar_x = self.decoder(z)
        xhat = self.decoder.sample(mu_x, logvar_x)
        return mu_z, logvar_z, xhat