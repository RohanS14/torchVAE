import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

from models.vae import VariationalAutoencoder
from training.trainlib import trainVAE
from utils.datatools import load_data, blackout_dataloader

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def kl_gaussian(mu_target, logvar_target, mu_source, logvar_source):
    sigma_t_sq = torch.exp(logvar_target)
    sigma_s_sq = torch.exp(logvar_source)
    term1 = logvar_source - logvar_target
    term2 = (sigma_t_sq + (mu_target - mu_source) ** 2) / sigma_s_sq
    kl = 0.5 * torch.sum(term1 + term2 - 1, dim=1)
    return kl.mean()

def train_one_shot(vae1, vae2, decoder, full_loader, blackout_loader, config, device):
    print("Starting one-shot training")
    wandb.init(project="torchVAE", name=config["run_name"] + "_oneshot", config=config, entity="hopelab-hmc")

    optimizer = torch.optim.Adam(list(vae1.encoder.parameters()) + list(vae2.encoder.parameters()) + list(decoder.parameters()), lr=config["training"]["oneshot_lr"])
    epochs = config["training"]["oneshot_epochs"]
    beta = config["model"]["beta"]
    pm_weight = config["model"].get("pm_weight", 1.0)
    progress = tqdm.trange(epochs)

    for epoch in progress:
        epoch_loss = 0.0
        vae1.train()
        vae2.train()
        decoder.train()

        for (xb, _), (xf, _) in zip(blackout_loader, full_loader):
            xb, xf = xb.to(device), xf.to(device)
            optimizer.zero_grad()

            mu1, logvar1 = vae1.encoder(xf)
            mu2, logvar2 = vae2.encoder(xb)

            std2 = torch.exp(0.5 * logvar2)
            eps = torch.randn_like(std2)
            z2 = mu2 + eps * std2

            recon2, _ = decoder(z2)

            recon_loss = ((xb.view(xb.size(0), -1) - recon2) ** 2).sum() / xb.size(0)
            kl_loss = kl_divergence(mu2, logvar2)
            elbo = recon_loss + beta * kl_loss
            pm = kl_gaussian(mu2, logvar2, mu1.detach(), logvar1.detach())
            total_loss = elbo + pm_weight * pm

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(full_loader)
        progress.set_description(f"Epoch [{epoch+1}/{epochs}] OneShotLoss: {avg_loss:.4f}")
        wandb.log({
            "OneShot Total Loss": avg_loss,
            "ELBO Loss": elbo.item(),
            "Reconstruction Loss": recon_loss.item(),
            "KL Loss": kl_loss.item(),
            "Posterior Matching Loss": pm.item(),
        })

    wandb.finish()

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, _, input_size = load_data("MNIST", config["dataset"]["num_train"])
    vae1 = VariationalAutoencoder(
        architecture=config["model"]["architecture"],
        latent_dims=config["model"]["latent_dims"],
        input_dims=input_size,
    ).to(device)
    vae2 = VariationalAutoencoder(
        architecture=config["model"]["architecture"],
        latent_dims=config["model"]["latent_dims"],
        input_dims=input_size,
    ).to(device)

    (blackout_l, blackout_u), (full_l, full_u) = blackout_dataloader("MNIST", config["dataset"]["num_train"])
    blackout_loader = torch.utils.data.DataLoader(blackout_u, batch_size=config["training"]["batch_size"], shuffle=False)
    full_loader = torch.utils.data.DataLoader(full_u, batch_size=config["training"]["batch_size"], shuffle=False)

    train_one_shot(
        vae1=vae1,
        vae2=vae2,
        decoder=vae1.decoder,
        full_loader=full_loader,
        blackout_loader=blackout_loader,
        config=config,
        device=device
    )

    # Final image logging
    print("Logging reconstructions for visual inspection")
    vae2.eval()
    vae1.decoder.eval()
    xb_vis, _ = next(iter(blackout_loader))
    xb_vis = xb_vis.to(device)

    with torch.no_grad():
        mu_vis, logvar_vis = vae2.encoder(xb_vis)
        std_vis = torch.exp(0.5 * logvar_vis)
        eps = torch.randn_like(std_vis)
        z_vis = mu_vis + eps * std_vis
        recon_vis, _ = vae1.decoder(z_vis)
        recon_vis = recon_vis.clamp(0, 1)

    logged_images = []
    for i in range(min(8, xb_vis.shape[0])):
        recon_img = recon_vis[i].view(1, 28, 28)
        stacked = torch.cat((xb_vis[i], recon_img), dim=2)
        logged_images.append(wandb.Image(stacked.cpu().numpy(), caption=f"Blackout | Recon {i}"))

    wandb.init(project="torchVAE", name=config["run_name"] + "_eval", config=config, entity="hopelab-hmc")
    wandb.log({"Reconstructions": logged_images})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
