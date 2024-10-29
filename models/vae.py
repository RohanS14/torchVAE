"""Variational Autoencoder"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class LinearEncoder(nn.Module):
    """Simplest possible encoder for VAE with Gaussian distribution and fully connected layers."""

    def __init__(self, latent_dims, input_dims=784):
        super(LinearEncoder, self).__init__()

        self.input_dims = input_dims

        self.linear1 = nn.Linear(input_dims, 512)
        self.linear2 = nn.Linear(512, latent_dims)  # outputs mu
        self.linear3 = nn.Linear(512, latent_dims)  # outputs logvar

        self.N = torch.distributions.Normal(0, 1)

        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        logvar = self.linear3(x)
        return mu, logvar

    def sample(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        z = mu + sigma * self.N.sample(mu.shape)
        return z


class LinearDecoder(nn.Module):
    """
    Simplest possible decoder for VAE with Gaussian distribution, fixed variance across pixels and fully connected layers.
    """

    def __init__(self, latent_dims, output_dims=784):
        super(LinearDecoder, self).__init__()

        self.output_dims = output_dims

        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2_mu = nn.Linear(512, output_dims)

        # fixed logvar across all params
        self.fixed_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        h = F.relu(self.linear1(z))
        mu = torch.sigmoid(self.linear2_mu(h))
        logvar = self.fixed_scale.expand_as(mu)
        return mu, logvar

    def sample(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        normal_dist = torch.distributions.Normal(mu, sigma)
        xhat = normal_dist.rsample()
        image_size = math.isqrt(self.output_dims)
        return xhat.reshape((-1, 1, image_size, image_size))


class FCEncoder(nn.Module):
    """Fully connected encoder for VAE with Gaussian distribution, deeper than LinearEncoder."""

    def __init__(self, latent_dims=50, input_dims=784):
        super(FCEncoder, self).__init__()

        self.input_dims = input_dims

        self.linear1 = nn.Linear(input_dims, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, latent_dims)  # outputs mu
        self.linear4 = nn.Linear(512, latent_dims)  # outputs logvar

        self.N = torch.distributions.Normal(0, 1)

        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.linear3(x)
        logvar = self.linear4(x)

        return mu, logvar

    def sample(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        z = mu + sigma * self.N.sample(mu.shape)
        return z


class FCDecoder(nn.Module):
    """Fully connected decoder with fixed variance across pixels.

    Deeper than LinearDecoder and supports Continuous Bernoulli and Gaussian distributions.
    """

    def __init__(self, latent_dims=50, distn="bern", output_dims=784):
        super(FCDecoder, self).__init__()

        self.output_dims = output_dims
        self.distn = distn

        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3_mu = nn.Linear(512, output_dims)

        if distn != "bern":
            # fixed logvar
            self.fixed_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        h = F.relu(self.linear1(z))
        h = F.relu(self.linear2(h))
        mu = torch.sigmoid(self.linear3_mu(h))

        if self.distn == "bern":
            return mu
        else:
            logvar = self.fixed_scale.expand_as(mu)
            return mu, logvar

    def sample(self, mu, logvar=None):
        if self.distn == "bern":
            assert mu.all() <= 1 and mu.all() >= 0
            bern = torch.distributions.ContinuousBernoulli(probs=mu)
            xhat = bern.rsample()
        else:
            # Using torch.distributions to create a normal distribution and sample from it
            # no need to expand due to broadcasting
            sigma = torch.exp(logvar / 2)
            normal_dist = torch.distributions.Normal(mu, sigma)
            xhat = normal_dist.rsample()
        image_size = math.isqrt(self.output_dims)
        return xhat.reshape((-1, 1, image_size, image_size))


class ConvEncoder(nn.Module):
    """Convolutional encoder for VAE with Gaussian distribution."""

    def __init__(self, latent_dims):
        super(ConvEncoder, self).__init__()

        init_channels = 64
        self.conv1 = nn.Conv2d(3, init_channels, kernel_size=3, padding=1, stride=2)
        self.skip_conv1 = nn.Conv2d(
            3, init_channels, kernel_size=3, padding=1, stride=2
        )
        self.conv2 = nn.Conv2d(
            init_channels, init_channels * 2, kernel_size=3, padding=1, stride=2
        )
        self.skip_conv2 = nn.Conv2d(
            init_channels, init_channels * 2, kernel_size=3, padding=1, stride=2
        )
        self.conv3 = nn.Conv2d(
            init_channels * 2, init_channels * 4, kernel_size=3, padding=1, stride=2
        )
        self.skip_conv3 = nn.Conv2d(
            init_channels * 2, init_channels * 4, kernel_size=3, padding=1, stride=2
        )
        self.activation = nn.ReLU()

        self.linear_mu = nn.Linear(init_channels * 4 * 4 * 4, latent_dims)  # outputs mu
        self.linear_logvar = nn.Linear(
            init_channels * 4 * 4 * 4, latent_dims
        )  # outputs logvar

        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = self.activation(self.conv1(x)) + self.skip_conv1(x)
        x = self.activation(self.conv2(x)) + self.skip_conv2(x)
        x = self.activation(self.conv3(x)) + self.skip_conv3(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        z = mu + sigma * self.N.sample(mu.shape)
        return z


class ConvDecoder(nn.Module):
    """Convolutional decoder for VAE with Gaussian distribution."""

    def __init__(self, latent_dims):
        super(ConvDecoder, self).__init__()
        init_channels = 64

        self.linear = nn.Linear(latent_dims, init_channels * 4 * 4 * 4)
        self.convT1 = nn.ConvTranspose2d(
            init_channels * 4, init_channels * 2, kernel_size=2, stride=2
        )
        self.skip_convT1 = nn.ConvTranspose2d(
            init_channels * 4, init_channels * 2, kernel_size=2, stride=2
        )
        self.convT2 = nn.ConvTranspose2d(
            init_channels * 2, init_channels, kernel_size=2, stride=2
        )
        self.skip_convT2 = nn.ConvTranspose2d(
            init_channels * 2, init_channels, kernel_size=2, stride=2
        )
        self.convT3 = nn.ConvTranspose2d(init_channels, 3, kernel_size=2, stride=2)
        self.skip_convT3 = nn.ConvTranspose2d(init_channels, 3, kernel_size=2, stride=2)
        self.activation = nn.ReLU()

        self.fixed_scale = nn.Parameter(torch.tensor(0.0))  # logvar

    def forward(self, z):
        init_channels = 64
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
        return xhat.reshape((-1, 3, 32, 32))


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder with different architectures.

    Args:
        architecture (str): Architecture type ("linear", "fc", or "conv").
        latent_dims (int): Dimensionality of the latent space.
        input_dims (int): Dimensionality of the input data.

    Attributes:
        architecture (str): Architecture type.
        latent_dims (int): Dimensionality of the latent space.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
    """

    def __init__(self, architecture="fc", latent_dims=20, input_dims=784, distn="bern"):
        super(VariationalAutoencoder, self).__init__()

        self.architecture = architecture
        self.latent_dims = latent_dims

        if architecture == "linear":
            self.encoder = LinearEncoder(latent_dims, input_dims=input_dims)
            self.decoder = LinearDecoder(latent_dims, output_dims=input_dims)
        if architecture == "fc":
            self.encoder = FCEncoder(latent_dims, input_dims=input_dims)
            self.decoder = FCDecoder(latent_dims, distn=distn, output_dims=input_dims)
        if architecture == "conv":
            self.encoder = ConvEncoder(latent_dims)
            self.decoder = ConvDecoder(latent_dims)

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = self.encoder.sample(mu_z, logvar_z)

        if self.architecture == "fc" and self.decoder.distn == "bern":
            image_size = math.isqrt(self.decoder.output_dims)
            probs_xhat = self.decoder(z).reshape((-1, 1, image_size, image_size))
            xhat = self.decoder.sample(probs_xhat)
        else:
            mu_x, sigma_x = self.decoder(z)
            xhat = self.decoder.sample(mu_x, sigma_x)

        return mu_z, logvar_z, xhat
