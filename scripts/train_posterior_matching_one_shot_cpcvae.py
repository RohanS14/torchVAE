import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import tqdm
import wandb

from models.cpcvae import ConsistencyConstrainedVAE
from utils.datatools import load_data, blackout_dataloader


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def kl_gaussian(mu_t: torch.Tensor, logvar_t: torch.Tensor,
                mu_s: torch.Tensor, logvar_s: torch.Tensor) -> torch.Tensor:
    sigma_t2 = torch.exp(logvar_t)
    sigma_s2 = torch.exp(logvar_s)
    term1 = logvar_s - logvar_t
    term2 = (sigma_t2 + (mu_t - mu_s) ** 2) / sigma_s2
    return 0.5 * torch.sum(term1 + term2 - 1, dim=1).mean()


def train_one_shot_cpcvae(cpcvae: ConsistencyConstrainedVAE,
                          full_loader: torch.utils.data.DataLoader,
                          blackout_loader: torch.utils.data.DataLoader,
                          config: dict,
                          device: torch.device) -> None:
    print("Starting one‑shot training with CPCVAE (shared encoder)")
    wandb.init(project="torchVAE",
               name=f"{config['run_name']}_oneshot_cpcvae",
               config=config,
               entity="hopelab-hmc")

    lr = float(config["training"]["oneshot_lr"])
    optimizer = torch.optim.Adam(cpcvae.parameters(), lr=lr)
    epochs = config["training"]["oneshot_epochs"]
    beta = config["model"]["beta"]
    pm_weight = config["model"].get("pm_weight", 1.0)
    consistency_weight = config["model"].get("consistency_weight", 1.0)
    classification_weight = config["model"].get("classification_weight", 1.0)

    progress = tqdm.trange(epochs, desc="Training")
    for epoch in progress:
        cpcvae.train()
        running_loss = 0.0

        for (xb, yb), (xf, yf) in zip(blackout_loader, full_loader):
            xb, xf = xb.to(device), xf.to(device)
            yb, yf = yb.to(device), yf.to(device)
            optimizer.zero_grad()

            mu_mask, logvar_mask, _, xhat_mask, logits_z_mask, logits_zhat_mask, xhat2_mask = cpcvae(xb)
            mu_full, logvar_full, _, xhat_full, logits_z_full, logits_zhat_full, xhat2_full = cpcvae(xf)

            z_mask = cpcvae.encoder.sample(mu_mask, logvar_mask)
            z_full = cpcvae.encoder.sample(mu_full, logvar_full)

            rec_mask = F.mse_loss(xhat_mask, xb, reduction="sum") / xb.size(0)
            rec_full = F.mse_loss(xhat_full, xf, reduction="sum") / xf.size(0)
            kl_mask = kl_divergence(mu_mask, logvar_mask)
            kl_full = kl_divergence(mu_full, logvar_full)

            elbo_mask = rec_mask + beta * kl_mask
            elbo_full = rec_full + beta * kl_full

            pm_loss = kl_gaussian(mu_mask, logvar_mask, mu_full.detach(), logvar_full.detach())
            consistency_loss = F.mse_loss(z_mask, cpcvae.encoder.sample(*cpcvae.encoder(xhat_mask)))
            classification_loss = (
                F.cross_entropy(logits_z_mask, yb) +
                F.cross_entropy(logits_zhat_mask, yb)
            ) / 2.0

            total_loss = (
                elbo_mask + elbo_full +
                pm_weight * pm_loss +
                consistency_weight * consistency_loss +
                classification_weight * classification_loss
            )

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        avg_loss = running_loss / len(full_loader)
        progress.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        wandb.log({
            "Total Loss": avg_loss,
            "Recon Mask": rec_mask.item(),
            "Recon Full": rec_full.item(),
            "KL Mask": kl_mask.item(),
            "KL Full": kl_full.item(),
            "PM Loss": pm_loss.item(),
            "Consistency Loss": consistency_loss.item(),
            "Classification Loss": classification_loss.item(),
        })

    wandb.finish()


def main(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    cpcvae = ConsistencyConstrainedVAE(
        architecture=config["model"]["architecture"],
        latent_dims=config["model"]["latent_dims"],
        num_classes=10,
        input_dims=input_dims,
    ).to(device)

    train_one_shot_cpcvae(
        cpcvae=cpcvae,
        full_loader=full_loader,
        blackout_loader=blackout_loader,
        config=config,
        device=device,
    )

    # Qualitative recon logging
    print("Logging reconstructions for visual inspection…")
    xb_vis, yb_vis = next(iter(blackout_loader))
    xb_vis = xb_vis.to(device)
    cpcvae.eval()
    with torch.no_grad():
        mu_v, logvar_v, _, xhat, _, _, _ = cpcvae(xb_vis)
        recon_v = xhat.clamp(0, 1)

    images = []
    for i in range(min(8, xb_vis.size(0))):
        stacked = torch.cat((xb_vis[i], recon_v[i]), dim=2)
        images.append(wandb.Image(stacked.cpu().numpy(), caption=f"Blackout | Recon {i}"))

    wandb.init(project="torchVAE",
               name=f"{config['run_name']}_eval_cpcvae",
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
