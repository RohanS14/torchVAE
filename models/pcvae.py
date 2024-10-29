"""Prediction-constrained VAE."""

import math

from models.vae import VariationalAutoencoder
from models.logreg import LogisticRegression


class PredictionConstrainedVAE(VariationalAutoencoder):
    """Variational Autoencoder with prediction constraint.

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
        super(PredictionConstrainedVAE, self).__init__(
            architecture, latent_dims, input_dims, distn
        )

        self.architecture = architecture
        self.classifier = LogisticRegression(latent_dims, num_classes)

    def forward(self, x):
        """Encodes and decodes the image. Classifies the input x based on its latent representation z."""
        mu_z, sigma_z = self.encoder(x)
        z = self.encoder.sample(mu_z, sigma_z)
        
        if self.architecture == "fc" and self.decoder.distn == "bern":
            image_size = math.isqrt(self.decoder.output_dims)
            probs_xhat = self.decoder(z).reshape((-1, 1, image_size, image_size))
            xhat = self.decoder.sample(probs_xhat)
        else:
            mu_x, sigma_x = self.decoder(z)
            xhat = self.decoder.sample(mu_x, sigma_x)

        # make classifications based on latent variable
        logits = self.classifier(z)

        return mu_z, sigma_z, xhat, logits
