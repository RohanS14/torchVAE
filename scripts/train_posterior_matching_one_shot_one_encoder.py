import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import tqdm
import wandb

from models.vae import VariationalAutoencoder
from utils.datatools import load_data, blackout_dataloader


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard KL( N(mu, sigma) || N(0, I) ) summed over latent dims and
    averaged over the batch."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def kl_gaussian(mu_t: torch.Tensor, logvar_t: torch.Tensor,
                 mu_s: torch.Tensor, logvar_s: torch.Tensor) -> torch.Tensor:
    """KL divergence KL( q_t || q_s ) where both are diagonal Gaussians.
    Used as the posterior‑matching term."""
    sigma_t2 = torch.exp(logvar_t)
    sigma_s2 = torch.exp(logvar_s)
    term1 = logvar_s - logvar_t
    term2 = (sigma_t2 + (mu_t - mu_s) ** 2) / sigma_s2
    return 0.5 * torch.sum(term1 + term2 - 1, dim=1).mean()


# ---------------------------------------------------------------------------
# Training routine with a **shared encoder**
# ---------------------------------------------------------------------------

def train_one_shot_shared(vae: VariationalAutoencoder,
                          full_loader: torch.utils.data.DataLoader,
                          blackout_loader: torch.utils.data.DataLoader,
                          config: dict,
                          device: torch.device) -> None:
    print("Starting one‑shot training with shared encoder")
    wandb.init(project="torchVAE",
               name=f"{config['run_name']}_oneshot_shared",
               config=config,
               entity="hopelab-hmc")

    encoder, decoder = vae.encoder, vae.decoder
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config["training"]["oneshot_lr"],
    )

    epochs = config["training"]["oneshot_epochs"]
    beta = config["model"]["beta"]
    pm_weight = config["model"].get("pm_weight", 1.0)

    progress = tqdm.trange(epochs, desc="Training")
    for epoch in progress:
        vae.train()
        running_loss = 0.0

        for (xb, _), (xf, _) in zip(blackout_loader, full_loader):
            xb, xf = xb.to(device), xf.to(device)
            optimizer.zero_grad()

            # --- shared encoder forward passes ---
            mu_mask, logvar_mask = encoder(xb)
            mu_full, logvar_full = encoder(xf)

            # Reparameterisation trick
            z_mask = mu_mask + torch.randn_like(mu_mask) * torch.exp(0.5 * logvar_mask)
            z_full = mu_full + torch.randn_like(mu_full) * torch.exp(0.5 * logvar_full)

            # Reconstructions
            recon_mask, _ = decoder(z_mask)
            recon_full, _ = decoder(z_full)

            # Loss components --------------------------------------------------
            rec_mask = F.mse_loss(recon_mask, xb.view(xb.size(0), -1), reduction="sum") / xb.size(0)
            rec_full = F.mse_loss(recon_full, xf.view(xf.size(0), -1), reduction="sum") / xf.size(0)

            kl_mask = kl_divergence(mu_mask, logvar_mask)
            kl_full = kl_divergence(mu_full, logvar_full)

            elbo_mask = rec_mask + beta * kl_mask
            elbo_full = rec_full + beta * kl_full  # stabilises encoder

            pm_loss = kl_gaussian(mu_mask, logvar_mask, mu_full.detach(), logvar_full.detach())

            total_loss = elbo_mask + elbo_full + pm_weight * pm_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        avg_loss = running_loss / len(full_loader)
        progress.set_description(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")
        wandb.log({
            "Total Loss": avg_loss,
            "ELBO Mask": elbo_mask.item(),
            "ELBO Full": elbo_full.item(),
            "Recon Mask": rec_mask.item(),
            "Recon Full": rec_full.item(),
            "KL Mask": kl_mask.item(),
            "KL Full": kl_full.item(),
            "PM Loss": pm_loss.item(),
        })

    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data ------------------------------------------------------------------
    _, _, input_dims = load_data("MNIST", config["dataset"]["num_train"])
    (blackout_l, blackout_u), (full_l, full_u) = blackout_dataloader(
        "MNIST", config["dataset"]["num_train"],
    )
    blackout_loader = torch.utils.data.DataLoader(
        blackout_u, batch_size=config["training"]["batch_size"], shuffle=False,
    )
    full_loader = torch.utils.data.DataLoader(
        full_u, batch_size=config["training"]["batch_size"], shuffle=False,
    )

    # Model -----------------------------------------------------------------
    vae = VariationalAutoencoder(
        architecture=config["model"]["architecture"],
        latent_dims=config["model"]["latent_dims"],
        input_dims=input_dims,
    ).to(device)

    # Training --------------------------------------------------------------
    train_one_shot_shared(
        vae=vae,
        full_loader=full_loader,
        blackout_loader=blackout_loader,
        config=config,
        device=device,
    )

    # ----------------------------------------------------------------------
    # Qualitative evaluation: log a small batch of blackout → reconstruction
    # ----------------------------------------------------------------------
    print("Logging reconstructions for visual inspection…")
    xb_vis, _ = next(iter(blackout_loader))
    xb_vis = xb_vis.to(device)
    vae.eval()
    with torch.no_grad():
        mu_v, logvar_v = vae.encoder(xb_vis)
        z_v = mu_v + torch.randn_like(mu_v) * torch.exp(0.5 * logvar_v)
        recon_v, _ = vae.decoder(z_v)
        recon_v = recon_v.clamp(0, 1)

    images = []
    for i in range(min(8, xb_vis.size(0))):
        stacked = torch.cat((xb_vis[i], recon_v[i].view(1, 28, 28)), dim=2)
        images.append(wandb.Image(stacked.cpu().numpy(), caption=f"Blackout | Recon {i}"))

    wandb.init(project="torchVAE",
               name=f"{config['run_name']}_eval_shared",
               config=config,
               entity="hopelab-hmc")
    wandb.log({"Reconstructions": images})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    main(cfg)
