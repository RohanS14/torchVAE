"Basic tests for models."

from models.logreg import LogisticRegression
from models.vae import VariationalAutoencoder
from models.pcvae import PredictionConstrainedVAE
from models.cpcvae import ConsistencyConstrainedVAE


def test_logreg():
    model = LogisticRegression(input_size=784, num_classes=10)
    assert model.num_classes == 10
    assert model.linear.in_features == 784


def test_vae():
    model = VariationalAutoencoder(architecture="conv", latent_dims=50, input_dims=784)
    assert model.architecture == "conv"
    assert model.latent_dims == 50


def test_pcvae():
    model = PredictionConstrainedVAE(
        architecture="linear", latent_dims=50, num_classes=10, input_dims=784
    )
    assert model.architecture == "linear"
    assert model.classifier.num_classes == 10


def test_cpcvae():
    model = ConsistencyConstrainedVAE(
        architecture="fc", latent_dims=50, num_classes=10, input_dims=784, distn="bern"
    )
    assert model.architecture == "fc"
    assert model.decoder.distn == "bern"
    assert model.classifier.num_classes == 10
