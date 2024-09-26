from models.vae import VariationalAutoencoder
from models.logreg import LogisticRegression

class ConsistencyConstrainedVAE(VariationalAutoencoder):
    def __init__(self, architecture="linear", latent_dims=20, num_classes=10):
        super(ConsistencyConstrainedVAE, self).__init__(architecture, latent_dims)
    
        self.classifier = LogisticRegression(latent_dims, num_classes)
        
    def forward(self, x):
        
        # encode, sample, decode
        mu_z, sigma_z = self.encoder(x)
        z = self.encoder.sample(mu_z, sigma_z)
        mu_x, sigma_x = self.decoder(z)
        xhat = self.decoder.sample(mu_x, sigma_x)
        
        # make classifications based on latent variable z
        logits_z = self.classifier(z)
        
        # Re-encode xhat to zhat
        mu_zhat, sigma_zhat = self.encoder(xhat)
        zhat = self.encoder.sample(mu_zhat, sigma_zhat)
        
        # make secondary classifications from zhat
        logits_zhat = self.classifier(zhat)
        
        return mu_z, sigma_z, xhat, logits_z, logits_zhat