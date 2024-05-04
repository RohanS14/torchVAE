from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import torch
    
def get_latent(vae, data, num_batches=100, latent_dims=20, device='cuda'):
    """
    Get the latent representations of data using a Variational Autoencoder (VAE).

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
    """
    Plot the 2D PCA representation of the latent representations.

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
    """
    Plot stacked histograms for each dimension of the latent representations.

    Args:
        z_batches (torch.Tensor): The latent representations.
        y_batches (torch.Tensor): The corresponding labels.
    """
    num_classes = len(np.unique(y_batches))
    num_dims = z_batches.shape[1]

    # Create histograms

    fig, axes = plt.subplots(5, 4, figsize=(12, 20))
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
            ax.legend()

    plt.tight_layout()
    plt.show()
    
def sample_random(vae, n=10, device='cuda'):
    """
    Generate and display random samples from the VAE decoder.

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
        random_z = torch.randn(20)
        
        # Decode z to distribution
        mean, sigma = vae.decoder(random_z.to(device))
        
        # Sample image from the distribution
        x_hat = vae.decoder.sample(mean, sigma)
        
        # Show image
        ax = axes[i]
        ax.imshow(x_hat.to('cpu').detach().squeeze(), cmap='gray')
        ax.axis('off')  # Hide axes
        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()