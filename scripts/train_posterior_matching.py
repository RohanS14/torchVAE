import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

# Importing existing modules
from models.vae import VariationalAutoencoder
from training.trainlib import trainVAE
from utils.datatools import load_data, blackout_dataloader


def kl_gaussian(mu_target, logvar_target, mu_source, logvar_source):
    """
    Computes KL divergence: KL(N(target) || N(source)).
    """
    sigma_t_sq = torch.exp(logvar_target)
    sigma_s_sq = torch.exp(logvar_source)
    term1 = logvar_source - logvar_target
    term2 = (sigma_t_sq + (mu_target - mu_source) ** 2) / sigma_s_sq
    kl = 0.5 * torch.sum(term1 + term2 - 1, dim=1)  # Sum over latent dims
    return kl.mean()  # Mean over batch


def train_posterior_encoder(old_encoder, new_encoder, full_loader, blackout_loader, epochs, lr, run_name, device):
    """Train a new encoder using KL loss to match old encoder."""
    print(f"Starting posterior matching training: {run_name}")
    wandb.init(project="torchVAE", name=run_name, entity="hopelab-hmc")

    for param in old_encoder.parameters():
        param.requires_grad = False
    old_encoder.eval()
    optimizer = torch.optim.Adam(new_encoder.parameters(), lr=lr)

    progress = tqdm.trange(epochs)
    for epoch in progress:
        epoch_loss = 0.0
        new_encoder.train()

        for (xb, _), (xf, _) in zip(blackout_loader, full_loader):
            xb, xf = xb.to(device), xf.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                mu_old, logvar_old = old_encoder(xf)
            
            mu_new, logvar_new = new_encoder(xb)
            loss = kl_gaussian(mu_new, logvar_new, mu_old, logvar_old)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(full_loader)
        progress.set_description(f"Epoch [{epoch+1}/{epochs}] KL: {avg_loss:.4f}")
        wandb.log({"Posterior KL": avg_loss})

    wandb.finish()
    return new_encoder


def main(config):
    """Main pipeline for posterior matching training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1: Train Standard VAE
    run_name_vae = config["run_name"] + "_vae"
    print(f"Training standard VAE: {run_name_vae}")
    
    train_data, _, input_size = load_data("MNIST", config["dataset"]["num_train"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)
    
    old_vae = VariationalAutoencoder(
        architecture=config["model"]["architecture"],
        latent_dims=config["model"]["latent_dims"],
        input_dims=input_size,
    ).to(device)
    
    trainVAE(
        old_vae,
        train_loader,
        config["model"]["beta"],
        config["training"]["num_epochs"],
        config["training"]["learning_rate"],
        run_name_vae,
        device,
        config=config,
    )
    
    if config["training"].get("save_model", False):
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(old_vae.state_dict(), f"./checkpoints/{run_name_vae}.pt")
        print(f"Saved trained VAE: {run_name_vae}")

    old_vae.eval()
    for param in old_vae.parameters():
        param.requires_grad = False

    # Stage 2: Train New Encoder for Posterior Matching
    run_name_enc = config["run_name"] + "_encoder"
    print(f"Training new encoder: {run_name_enc}")

    new_vae = VariationalAutoencoder(
        architecture=config["model"]["architecture"],
        latent_dims=config["model"]["latent_dims"],
        input_dims=input_size,
    ).to(device)

    for param in new_vae.decoder.parameters():
        param.requires_grad = False  # Only train encoder

    torch.manual_seed(0)
    (blackout_l, blackout_u), (full_l, full_u) = blackout_dataloader("MNIST", config["dataset"]["num_train"])
    blackout_loader = torch.utils.data.DataLoader(blackout_u, batch_size=config["training"]["batch_size"], shuffle=False)
    full_loader = torch.utils.data.DataLoader(full_u, batch_size=config["training"]["batch_size"], shuffle=False)

    train_posterior_encoder(
        old_encoder=old_vae.encoder,
        new_encoder=new_vae.encoder,
        full_loader=full_loader,
        blackout_loader=blackout_loader,
        epochs=config["training"]["posterior_epochs"],
        lr=config["training"]["posterior_lr"],
        run_name=run_name_enc,
        device=device,
    )

    # Stage 3: Evaluate Reconstruction
    print("Evaluating new encoder with blacked-out images.")
    new_vae.encoder.eval()
    old_vae.decoder.eval()
    
    xb, _ = next(iter(blackout_loader))
    xb = xb.to(device)
    
    with torch.no_grad():
        mu_new, logvar_new = new_vae.encoder(xb)
        std_new = torch.exp(0.5 * logvar_new)
        eps = torch.randn_like(std_new)
        z_new = mu_new + eps * std_new
        recon, _ = old_vae.decoder(z_new)
        # recon, _ = old_vae.decoder.sample(mu_new, logvar_new)
    
    recon = recon.clamp(0, 1)

    logged_images = []
    for i in range(min(8, xb.shape[0])):
        recon_img = recon[i].view(1, 28, 28)  # reshape to match input
        stacked = torch.cat((xb[i], recon_img), dim=2)  # [1, 28, 56]
        logged_images.append(wandb.Image(stacked.cpu().numpy(), caption=f"Blackout | Recon {i}"))


    # Log images
    wandb.init(project="torchVAE", name=config["run_name"] + "_eval", config=config, entity="hopelab-hmc")
    wandb.log({"Blackout vs Reconstruction": logged_images})
    wandb.finish()

    print("Evaluation complete. See WandB logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)