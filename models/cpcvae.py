"""Consistency Constrained VAE."""

import math
from models.vae import VariationalAutoencoder
from models.logreg import LogisticRegression


class ConsistencyConstrainedVAE(VariationalAutoencoder):
    """Variational Autoencoder with consistency and prediction constraints.

    Args:
        architecture (str): The architecture type of the VAE.
        latent_dims (int): The number of latent dimensions.
        num_classes (int): The number of classes for the classifier.
        input_dims (int): The size of the input. Default is 784 for MNIST.

    Attributes:
        classifier (LogisticRegression): The classifier for the latent space.
    """

    def __init__(
        self, architecture="fc", latent_dims=20, num_classes=10, input_dims=784, distn="bern"
    ):
        super(ConsistencyConstrainedVAE, self).__init__(
            architecture, latent_dims, input_dims, distn
        )

        self.architecture = architecture
        self.classifier = LogisticRegression(latent_dims, num_classes)

    def forward(self, x):
        """Encodes, decodes, re-encodes the image. Classifies the input x."""

        mu_z, sigma_z = self.encoder(x)
        z = self.encoder.sample(mu_z, sigma_z)

        if self.architecture == "fc" and self.decoder.distn == "bern":
            # CHANGED: made this size agnostic
            image_size = math.isqrt(self.decoder.output_dims)
            mu_x = self.decoder(z).reshape((-1, 1, image_size, image_size))
            xhat = self.decoder.sample(mu_x)
        else:
            mu_x, sigma_x = self.decoder(z)
            xhat = self.decoder.sample(mu_x, sigma_x)

        # make classifications based on latent variable z
        logits_z = self.classifier(z)

        # Re-encode xhat to zhat
        mu_zhat, sigma_zhat = self.encoder(xhat)
        zhat = self.encoder.sample(mu_zhat, sigma_zhat)

        # make secondary classifications from zhat
        logits_zhat = self.classifier(zhat)

        return mu_z, sigma_z, mu_x, xhat, logits_z, logits_zhat
