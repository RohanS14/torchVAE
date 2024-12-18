"""Training loops for VAEs and classification models."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import itertools

import wandb


def kl_divergence(mu, logvar):
    """
    Compute the KL divergence between a normal distribution with mean mu and log variance logvar
    and a standard normal distribution.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def trainVAE(vae, data, beta, epochs, lr, run_name, device="cuda", config=None):
    """Unsupervised training loop for a VAE."""
    recon_losses = []
    kl_losses = []

    print(f"Logging to {run_name}")

    wandb.init(project="torchVAE", name=run_name, config=config, entity="hopelab-hmc")

    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    progress = tqdm.trange(epochs)

    for epoch in progress:
        print(epoch)
        for x, _ in data:
            x = x.to(device)  # GPU

            opt.zero_grad()
            mu_z, logvar_z, x_hat = vae(x)

            kl_loss = kl_divergence(mu_z, logvar_z)

            recon_loss = ((x - x_hat) ** 2).sum()
            loss = recon_loss + beta * kl_loss
            loss.backward()

            recon_losses.append(recon_loss)
            kl_losses.append(kl_loss)
            opt.step()
        progress.set_description(f"Loss: {loss}, Recon: {recon_loss}, KL: {kl_loss}")

        # Log the losses to WandB
        wandb.log(
            {
                "Reconstruction Loss": recon_loss.item(),
                "KL Loss": kl_loss.item(),
                "Total Loss": loss.item(),
            }
        )

    wandb.finish()

    return recon_losses, kl_losses, vae


def validate(model, dataLoader, device, input_size, loss_fn):
    """Validate classification on raw images."""
    model.eval()
    accurate = 0
    total = 0
    dl_losses = []

    with torch.no_grad():
        for images, labels in dataLoader:
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)

            output = model(images)
            _, predicted = torch.max(output.data, 1)

            # compute loss
            dl_losses.append(loss_fn(output, labels).item())

            # total labels
            total += labels.size(0)

            # total correct predictions
            accurate += (predicted == labels).sum().item()

    accuracy_score = 100 * accurate / total
    return np.mean(dl_losses), accuracy_score


def train_logreg(model, train_dataLoader, valid_dataLoader, device, optimizer, loss_fn, \
                 learning_rate, num_epochs, input_size, regularization, lambda_, stop_100, run_name, config=None):
    """Supervised training loop for a logistic regression model."""
    # summary writer with custom run name
    print(f"Logging to {run_name}")
    wandb.init(project="torchVAE", name=run_name, config=config, entity="hopelab-hmc")

    progress = tqdm.trange(num_epochs)

    for epoch in progress:
        model.train()
        accurate = 0
        total = 0

        print("Epoch", epoch)
        for images, labels in train_dataLoader:
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            train_loss = loss_fn(output, labels)

            # Adding Regularization
            if regularization == "l2":
                l2_reg = lambda_ * sum(p.pow(2.0).sum() for p in model.parameters())
                train_loss += l2_reg
            elif regularization == "l1":
                l1_reg = lambda_ * sum(p.abs().sum() for p in model.parameters())
                train_loss += l1_reg

            train_loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            accurate += (predicted == labels).sum().item()

        train_acc = 100 * accurate / total
        valid_loss, valid_acc = validate(
            model, valid_dataLoader, device, input_size, loss_fn
        )

        progress.set_description(
            f"Train Loss: {train_loss.item():.4f}. Train Accuracy: {train_acc:.4f}. Valid Loss: {valid_loss:.4f}. Valid Accuracy: {valid_acc:.4f}"
        )

        # Log the losses to WandB
        wandb.log(
            {
                "Training Loss": train_loss.item(),
                "Training Accuracy": train_acc,
                "Valid Loss": valid_loss,
                "Valid Accuracy": valid_acc,
            }
        )

        # Early stopping when train accuracy reaches 100%
        if stop_100 and train_acc >= 100:
            break

    wandb.finish()

    print("Final Train Accuracy:", train_acc)
    print("Final Val Accuracy:", valid_acc)

    return model


def validate_model(model, dataLoader, device, input_size=None):
    """Compute accuracy for constrained VAE."""
    model.eval()
    accurate = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)

            mus, logvars = model.encoder(images)

            encoded_images = model.encoder.sample(mus, logvars)

            output = model.classifier(encoded_images)
            _, predicted = torch.max(output.data, 1)

            # total labels
            total += labels.size(0)

            # total correct predictions
            accurate += (predicted == labels).sum().item()

    return 100 * accurate / total


def trainPCVAE(pcvae, unlabeled_data_loader, labeled_data_loader, epochs=20, lr=0.001, beta=1, \
    lambda_=1, l_weight=1, u_weight=1, run_name="default", device="cuda", config=None,
):
    """Semi-supervised training loop for a Prediction Constrained VAE."""

    criterion_recon = nn.MSELoss(reduction="sum")
    criterion_class = nn.CrossEntropyLoss()

    # set up wandb logging
    print(f"Logging to {run_name}")

    wandb.init(project="torchVAE", name=run_name, config=config, entity="hopelab-hmc")

    optimizer = torch.optim.Adam(pcvae.parameters(), lr=lr)
    progress = tqdm.trange(epochs)

    for epoch in progress:
        print(epoch)

        pcvae.train()

        # Create iterators, cycling the smaller one
        if len(unlabeled_data_loader) < len(labeled_data_loader):
            u_iter = itertools.cycle(unlabeled_data_loader)
            l_iter = iter(labeled_data_loader)
        else:
            l_iter = itertools.cycle(labeled_data_loader)
            u_iter = iter(unlabeled_data_loader)

        num_batches = max(len(unlabeled_data_loader), len(labeled_data_loader))

        # iterate through batches
        for _ in range(num_batches):
            x_u, _ = next(u_iter)
            x_l, y_l = next(l_iter)

            x_u = x_u.to(device)
            x_l, y_l = x_l.to(device), y_l.to(device)

            optimizer.zero_grad()

            # process unlabeled data
            mu_u, logvar_u, x_hat_u, _ = pcvae(x_u)

            recon_loss_u = criterion_recon(x_hat_u, x_u)
            kl_loss_u = kl_divergence(mu_u, logvar_u)

            loss_u = recon_loss_u + beta * kl_loss_u

            # process labeled data
            mu_l, logvar_l, x_hat_l, logits = pcvae(x_l)

            recon_loss_l = criterion_recon(x_hat_l, x_l)
            kl_loss_l = kl_divergence(mu_l, logvar_l)
            class_loss = criterion_class(logits, y_l)

            loss_l = recon_loss_l + beta * kl_loss_l + lambda_ * class_loss

            # combine all losses
            total_loss = u_weight * loss_u + l_weight * loss_l
            total_loss.backward()
            optimizer.step()

        # Log the losses
        with torch.no_grad():
            recon_loss = u_weight * recon_loss_u + l_weight * recon_loss_l
            kl_loss = u_weight * kl_loss_u + l_weight * kl_loss_l

            labeled_accuracy = validate_model(pcvae, labeled_data_loader, device)
            unlabeled_accuracy = validate_model(pcvae, unlabeled_data_loader, device)

            wandb.log(
                {
                    "Reconstruction Loss": recon_loss.item(),
                    "KL Loss": kl_loss.item(),
                    "Classifier Loss": class_loss.item(),
                    "Total Loss": total_loss.item(),
                    "Train Accuracy": labeled_accuracy,
                    "Test Accuracy": unlabeled_accuracy,
                }
            )

        progress.set_description(
            f"Total Loss: {total_loss}, Recon: {recon_loss}, KL: {kl_loss}, Classifier: {class_loss}, "
            f'Train Accuracy: {labeled_accuracy}, "Test Accuracy": {unlabeled_accuracy}'
        )

    return pcvae


def trainCPCVAE(cpcvae, unlabeled_data_loader, labeled_data_loader, epochs=20, lr=0.001, beta=1, \
    lambda_=1, gamma=1, l_weight=1, u_weight=1, run_name="default", device="cuda", config=None,
):
    """Semi-supervised training loop for a Consistency Constrained VAE."""

    criterion_recon = nn.MSELoss(reduction="sum")
    criterion_class = nn.CrossEntropyLoss()
    criterion_consistency = nn.CrossEntropyLoss()
    criterion_aggregate = nn.CrossEntropyLoss()

    # set up wandb logging
    print(f"Logging to {run_name}")
    wandb.init(project="torchVAE", name=run_name, config=config, entity="hopelab-hmc")

    optimizer = torch.optim.Adam(cpcvae.parameters(), lr=lr)
    progress = tqdm.trange(epochs)

    for epoch in progress:
        print(epoch)

        cpcvae.train()

        # Create iterators, cycling the smaller one
        if len(unlabeled_data_loader) < len(labeled_data_loader):
            u_iter = itertools.cycle(unlabeled_data_loader)
            l_iter = iter(labeled_data_loader)
        else:
            l_iter = itertools.cycle(labeled_data_loader)
            u_iter = iter(unlabeled_data_loader)

        num_batches = max(len(unlabeled_data_loader), len(labeled_data_loader))

        # iterate through batches
        for i in range(num_batches):
            x_u, _ = next(u_iter)
            x_l, y_l = next(l_iter)

            x_u = x_u.to(device)
            x_l, y_l = x_l.to(device), y_l.to(device)

            optimizer.zero_grad()

            # process unlabeled data
            mu_u, logvar_u, probs_xhat_u, x_hat_u, logits_z_u, logits_zhat_u = cpcvae(
                x_u
            )

            assert (
                x_u.shape == x_hat_u.shape
            ), f"Shapes don't match: {x_u.shape} and {x_hat_u.shape}"

            assert x_u.all() <= 1 and x_u.all() >= 0, "x_u not in [0, 1]"
            assert x_hat_u.all() <= 1 and x_hat_u.all() >= 0, "x_u not in [0, 1]"

            if cpcvae.decoder.distn == "bern":
                bern = torch.distributions.ContinuousBernoulli(probs=probs_xhat_u)
                recon_loss_u = -bern.log_prob(x_u).sum()
            else:
                recon_loss_u = criterion_recon(x_hat_u, x_u)

            kl_loss_u = kl_divergence(mu_u, logvar_u)

            loss_u = recon_loss_u + beta * kl_loss_u

            # added consistency for ul too
            consistency_loss_u = criterion_consistency(
                logits_zhat_u, F.softmax(logits_z_u, dim=-1)
            )
            loss_u += gamma * consistency_loss_u

            # process labeled data
            mu_l, logvar_l, probs_xhat_l, x_hat_l, logits_z_l, logits_zhat_l = cpcvae(x_l)

            if cpcvae.decoder.distn == "bern":
                bern = torch.distributions.ContinuousBernoulli(probs=probs_xhat_l)
                recon_loss_l = -bern.log_prob(x_l).sum()
            else:
                recon_loss_l = criterion_recon(x_hat_l, x_l)

            kl_loss_l = kl_divergence(mu_l, logvar_l)
            class_loss = criterion_class(logits_z_l, y_l)
            consistency_loss_l = criterion_consistency(
                logits_zhat_l, F.softmax(logits_z_l, dim=-1)
            )

            # aggregate label consistency
            # cross entropy loss between the distribution of predicted labels and uniform target
            probs = F.softmax(logits_z_u, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            counts = torch.bincount(preds, minlength=cpcvae.classifier.num_classes)
            assert counts.shape == (
                cpcvae.classifier.num_classes,
            ), f"Counts shape is wrong: {counts.shape}"
            counts = counts / counts.sum()
            assert torch.isclose(counts.sum(), torch.tensor(1.0)), f"Counts don't sum to 1: {counts.sum()}"
            agg_loss = criterion_aggregate(
                counts, torch.ones_like(counts) / cpcvae.classifier.num_classes
            )

            loss_l = (
                recon_loss_l
                + beta * kl_loss_l
                + lambda_ * class_loss
                + gamma * consistency_loss_l
                + agg_loss
            )

            # combine all losses
            total_loss = u_weight * loss_u + l_weight * loss_l
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            # this currently logs the scaled losses
            recon_loss = u_weight * recon_loss_u + l_weight * recon_loss_l
            kl_loss = beta * (u_weight * kl_loss_u + l_weight * kl_loss_l)
            class_loss = lambda_ * l_weight * class_loss
            consistency_loss = gamma * (
                u_weight * consistency_loss_u + l_weight * consistency_loss_l
            )

            labeled_accuracy = validate_model(cpcvae, labeled_data_loader, device)
            unlabeled_accuracy = validate_model(cpcvae, unlabeled_data_loader, device)

            wandb.log(
                {
                    "Reconstruction Loss": recon_loss.item(),
                    "KL Loss": kl_loss.item(),
                    "Classifier Loss": class_loss.item(),
                    "Consistency Loss": consistency_loss.item(),
                    "Aggregate Loss": agg_loss.item(),
                    "Total Loss": total_loss.item(),
                    "Train Accuracy": labeled_accuracy,
                    "Test Accuracy": unlabeled_accuracy,
                }
            )

        progress.set_description(
            f"Total Loss: {total_loss:.3f}, Recon: {recon_loss:.3f}, KL: {kl_loss:.3f}, Class: {class_loss:.3f},"
            f"Consistency: {consistency_loss:.3f}, Aggregate: {agg_loss:.3f}, Train Acc: {labeled_accuracy:.3f}, Test Acc: {unlabeled_accuracy:.3f}"
        )

    return cpcvae
