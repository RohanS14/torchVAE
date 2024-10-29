import math
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_latent(vae, data, num_batches=100, latent_dims=20, device='cuda'):
    """Get the latent representations of data using a Variational Autoencoder (VAE).

    Args:
        vae (VAE): The Variational Autoencoder model.
        data (DataLoader): The data loader containing the input data.
        num_batches (int, optional): The number of batches to process. Defaults to 100.
        latent_dims (int, optional): The number of latent dimensions. Defaults to 20.

    Returns:
        torch.Tensor: The latent representations.
        torch.Tensor: The corresponding labels.
    """
    z_batches = torch.empty((0, latent_dims))
    y_batches = torch.empty((0, ))
    
    # data loader with batch size 128
    for i, (x, y) in enumerate(data):
        # Decode z to distribution parameters
        mean, sigma = vae.encoder(x.to(device))
        
        # Sample an image from the distribution
        z = vae.encoder.sample(mean, sigma)
        
        z = z.to('cpu').detach()
        
        z_batches = torch.cat((z_batches, z), dim=0)
        y_batches = torch.cat((y_batches, y), dim=0)
        
        if i >= num_batches - 1:
            break
    return z_batches, y_batches

def plotPCAdist(z_batches, y_batches):
    """Plot the 2D PCA representation of the latent representations.

    Args:
        z_batches (torch.Tensor): The latent representations.
        y_batches (torch.Tensor): The corresponding labels.
    """
    latent_representations = z_batches.numpy()
    classes = y_batches.numpy()

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_representations)

    # Plotting
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=classes, cmap='tab10')
    plt.title('2D PCA of Image Latent Representations')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.legend(*scatter.legend_elements(),
                        loc="best", title="Classes")

    plt.show()
    
def stackedHist(z_batches, y_batches):
    """Plot stacked histograms for each dimension of the latent representations.

    Args:
        z_batches (torch.Tensor): The latent representations.
        y_batches (torch.Tensor): The corresponding labels.
    """
    num_classes = len(np.unique(y_batches))
    num_dims = z_batches.shape[1]

    # Create histograms

    # CHANGED: make size agnostic
    ncol = 5
    nrow = math.ceil(num_dims / ncol)

    fig, axes = fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 4))
    axes = axes.flatten()

    for dim in range(num_dims):
        ax = axes[dim]
        # Plot a histogram for each class
        for class_id in range(num_classes):
            mask = y_batches == class_id
            ax.hist(z_batches[mask, dim], bins=30, alpha=0.5, label=f'{class_id}')

        ax.set_xlabel(f'Dimension {dim+1} Values')
        ax.set_ylabel('Frequency')
        if dim == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15), ncol=num_classes)

    # Hide any unused subplots
    for j in range(num_dims, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
def sample_random(vae, n=10, device='cuda'):
    """Generate and display random samples from the VAE decoder.

    Args:
        vae (VAE): The Variational Autoencoder model.
        n (int, optional): The number of samples to generate. Defaults to 10.
    """
    rows = cols = int(n**0.5)
    if rows * cols < n:
        cols += 1
        
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()
    
    for i in range(n):
        # CHANGED: make size agnostic
        random_z = torch.randn(vae.latent_dims)
        
        if vae.architecture == "fc" and vae.decoder.distn == "bern":
                # Continuous Bernoulli distribution
                mu = vae.decoder(random_z.to(device))
                x_hat = vae.decoder.sample(mu)
        else:
            # Decode z to distribution
            mean, sigma = vae.decoder(random_z.to(device))
            
            # Sample image from the distribution
            x_hat = vae.decoder.sample(mean, sigma)

        # Show image
        ax = axes[i]
        ax.imshow(x_hat.to('cpu').detach().squeeze(), cmap='gray')
        ax.axis('off')
        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def reconstruct_and_compare(cpcvae, dataLoader, device, n=10):
    """Display original images along with two separate reconstructions from the same CPCVAE,
    sampling twice from the Continuous Bernoulli decoder.

    Args:
        cpcvae: The CPCVAE model.
        dataLoader: The data loader for the dataset.
        device: The device to run the model on ('cuda' or 'cpu').
        n (int): The number of images to display. Defaults to 10.
    """
    cpcvae.eval()
    
    images_to_show = []
    reconstructions1 = []
    reconstructions2 = []

    # Collect n images and their reconstructions
    for images, _ in dataLoader:
        images = images.to(device)
        
        with torch.no_grad():
            # Encode the images to get latent variables
            mu_z, logvar_z = cpcvae.encoder(images)

            # First reconstruction: Sample from the Continuous Bernoulli decoder
            recon_images1_logits = cpcvae.decoder(mu_z)
            recon_images1 = cpcvae.decoder.sample(recon_images1_logits)

            # Second reconstruction: Sample again from the same latent variables (mu_z, logvar_z)
            recon_images2 = cpcvae.decoder.sample(recon_images1_logits)  # Same logits, new sample

        images_to_show.append(images.cpu())
        reconstructions1.append(recon_images1.cpu())
        reconstructions2.append(recon_images2.cpu())

        if len(images_to_show) * images.shape[0] >= n:
            break

    # Now display the images and their two reconstructions
    images_to_show = torch.cat(images_to_show)[:n]  # Collect exactly n images
    reconstructions1 = torch.cat(reconstructions1)[:n]
    reconstructions2 = torch.cat(reconstructions2)[:n]
    
    plot_original_and_reconstructions(images_to_show, reconstructions1, reconstructions2)

def plot_original_and_reconstructions(originals, recon1, recon2):
    """Plot the original images, and two reconstructions from the same CPCVAE.

    Args:
        originals: The original images.
        recon1: The first set of reconstructions.
        recon2: The second set of reconstructions.
    """
    fig, axs = plt.subplots(3, len(originals), figsize=(len(originals) * 2, 6))
    
    for i in range(len(originals)):
        # Original images
        axs[0, i].imshow(originals[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title("Original")

        # First reconstruction
        axs[1, i].imshow(recon1[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title("Recon 1")

        # Second reconstruction
        axs[2, i].imshow(recon2[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axs[2, i].axis('off')
        axs[2, i].set_title("Recon 2")

    plt.tight_layout()
    plt.show()


def generate_confusion_matrix(cpcvae, dataLoader, device):
    """Generate a confusion matrix for the CPCVAE classifier.
    
    Args:
        cpcvae: The CPCVAE model.
        dataLoader: The data loader for the dataset.
        device: The device to run the model on ('cuda' or 'cpu').
    """
    cpcvae.eval()
    
    all_preds = []
    all_labels = []

    # Go through the entire dataset and collect predictions and true labels
    with torch.no_grad():
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)

            # Get classifier predictions (logits) and take the argmax for predicted labels
            logits = cpcvae.classifier(images)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Store true labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix using seaborn for a heatmap
    plot_confusion_matrix(cm, class_names=range(10))  # Assuming 10 classes

def plot_confusion_matrix(cm, class_names):
    """Plots the confusion matrix.
    
    Args:
        cm: The confusion matrix.
        class_names: List of class names (or range of class indices).
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
