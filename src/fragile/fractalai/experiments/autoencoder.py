import holoviews as hv
from sklearn.datasets import make_swiss_roll
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


# Use plotly backend for 3D plots (better for headless environments)
hv.extension("bokeh", "plotly")

# --- HYPERPARAMETERS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EPOCHS = 1500
BATCH_SIZE = 128
LEARNING_RATE = 0.5e-3
LATENT_DIM = 2
N_SAMPLES = 2000

# Geometric Loss Hyperparameters
LAMBDA_RECON = 1.0
LAMBDA_L1 = 0.01  # Sparsity penalty (L1 norm)
LAMBDA_L2 = 0.01  # Ridge penalty (L2 norm)
LAMBDA_RICCI = 0.1  # Encourages tightness (less important for this task)
LAMBDA_WEYL = 0.1  # Penalizes anisotropy (encourages flat, square patches)
LAMBDA_DIM = 0.01  # Penalizes dimensions > 2
LAMBDA_ENTROPY = 0.05  # Entropy regularization (encourages exploration)
RHO = 0.5  # Neighborhood scale


# --- GEOMETRIC LOSS FUNCTION (from previous example) ---
def geometric_loss_function(z, rho=0.5):
    """
    Fully vectorized geometric loss computation.

    Args:
        z: Latent vectors [N, d]
        rho: Neighborhood scale parameter

    Returns:
        ricci_loss: Local tightness measure (scalar)
        weyl_loss: Local anisotropy measure (scalar)
        dim_loss: Intrinsic dimension measure (scalar)
    """
    N, _d = z.shape

    # Compute pairwise squared distances [N, N]
    dist_sq = torch.cdist(z, z, p=2).pow(2)

    # Compute weights for all points simultaneously [N, N]
    weights = torch.exp(-dist_sq / (2 * rho**2))

    # Zero out self-weights (diagonal)
    weights *= 1 - torch.eye(N, device=z.device, dtype=z.dtype)

    # Normalize weights [N, N]
    weights /= weights.sum(dim=1, keepdim=True) + 1e-09

    # --- Ricci Proxy (Local Tightness) ---
    # Compute weighted sum of squared distances for each point [N]
    local_tightness = (weights * dist_sq).sum(dim=1)
    ricci_loss = local_tightness.mean()

    # --- Compute local covariance matrices for all points ---
    # Weighted mean for each point: z_mean[i] = sum_j weights[i,j] * z[j]
    # Shape: [N, d]
    z_mean = torch.matmul(weights, z)

    # Center z around each local mean
    # z_centered[i, j, :] = z[j, :] - z_mean[i, :]
    # Shape: [N, N, d]
    z_centered = z.unsqueeze(0) - z_mean.unsqueeze(1)

    # Compute weighted covariance for each point
    # cov[i] = sum_j weights[i,j] * z_centered[i,j] * z_centered[i,j]^T
    # We use einsum for efficient batched outer product
    # Shape: [N, d, d]
    weighted_z_centered = z_centered * weights.unsqueeze(-1)  # [N, N, d]
    cov = torch.matmul(weighted_z_centered.transpose(1, 2), z_centered)  # [N, d, d]

    # Compute eigenvalues for all covariance matrices
    # eigvals shape: [N, d]
    eigvals = torch.linalg.eigh(cov)[0]
    eigvals = torch.clamp(eigvals, min=1e-9)

    # --- Weyl Proxy (Local Anisotropy) ---
    # Normalize eigenvalues for each point [N, d]
    p_weyl = eigvals / (eigvals.sum(dim=1, keepdim=True) + 1e-9)
    # Compute variance across eigenvalues for each point [N]
    local_anisotropy = p_weyl.var(dim=1)
    weyl_loss = local_anisotropy.mean()

    # --- Dimension Proxy (Participation Ratio) ---
    # Normalize eigenvalues [N, d]
    p_dim = eigvals / (eigvals.sum(dim=1, keepdim=True) + 1e-9)
    # Participation ratio for each point [N]
    local_dim = 1.0 / (p_dim.pow(2).sum(dim=1) + 1e-9)
    dim_loss = local_dim.mean()

    return ricci_loss, weyl_loss, dim_loss


def entropy_loss_function(z, rho=0.5):
    """
    Vectorized entropy regularization based on local density estimation.

    Encourages the latent representation to spread out and explore the space
    by penalizing clustering (low entropy) and rewarding uniform distribution.

    Uses differential entropy approximation via k-nearest neighbors density.

    Args:
        z: Latent vectors [N, d]
        rho: Neighborhood scale parameter for density estimation

    Returns:
        entropy_loss: Negative entropy (scalar) - lower is more uniform/exploratory
    """
    N, _d = z.shape

    # Compute pairwise squared distances [N, N]
    dist_sq = torch.cdist(z, z, p=2).pow(2)

    # Compute local density using Gaussian kernel [N, N]
    # Density is high where points are clustered, low where spread out
    kernel = torch.exp(-dist_sq / (2 * rho**2))

    # Zero out self-similarity (diagonal)
    kernel *= 1 - torch.eye(N, device=z.device, dtype=z.dtype)

    # Local density for each point (sum of kernel weights) [N]
    local_density = kernel.sum(dim=1)

    # Normalize to form probability distribution
    p = local_density / (local_density.sum() + 1e-9)

    # Differential entropy: H = -sum(p * log(p))
    # We want to MAXIMIZE entropy (uniform spread), so we MINIMIZE negative entropy
    entropy = -(p * torch.log(p + 1e-9)).sum()

    # Return negative entropy as loss (minimizing this maximizes entropy)
    return -entropy


# --- MODEL ARCHITECTURE ---
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# --- DATA PREPARATION ---
X, color = make_swiss_roll(n_samples=N_SAMPLES, noise=0.05, random_state=42)
X = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalize
X_tensor = torch.FloatTensor(X).to(DEVICE)
dataset = TensorDataset(X_tensor)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- TRAINING FUNCTION ---
def train_ae(model, use_geometric_loss=False):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    recon_loss_fn = nn.MSELoss()
    print(
        f"\n--- Training "
        f"{'Geometric+Entropy' if use_geometric_loss else 'Standard'} "
        f"Autoencoder ---"
    )

    for epoch in range(EPOCHS):
        for data_batch in data_loader:
            points = data_batch[0].to(DEVICE)
            optimizer.zero_grad()

            x_recon, z = model(points)
            recon_loss = recon_loss_fn(x_recon, points)

            total_loss = LAMBDA_RECON * recon_loss

            if use_geometric_loss:
                l1_loss = torch.norm(z, 1) / z.shape[0]  # L1 on latent activations
                l2_loss = torch.norm(z, 2).pow(2) / z.shape[0]  # L2 on latent activations
                ricci_loss, weyl_loss, dim_loss = geometric_loss_function(z, rho=RHO)
                entropy_loss = entropy_loss_function(z, rho=RHO)
                total_loss += (
                    LAMBDA_L1 * l1_loss
                    + LAMBDA_L2 * l2_loss
                    + LAMBDA_RICCI * ricci_loss
                    + LAMBDA_WEYL * weyl_loss
                    + LAMBDA_DIM * dim_loss
                    + LAMBDA_ENTROPY * entropy_loss
                )

            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Recon Loss: {recon_loss.item():.4f}")


# --- VISUALIZATION ---
def plot_results(model, title, filename_prefix):
    model.eval()
    with torch.no_grad():
        _, z = model(X_tensor)
        z = z.cpu().numpy()
        x_recon, _ = model(X_tensor)
        x_recon = x_recon.cpu().numpy()

    # Original 3D Data - using Scatter3D with plotly backend
    original_3d = hv.Scatter3D(
        (X[:, 0], X[:, 1], X[:, 2], color), kdims=["x", "y", "z"], vdims=["color"]
    ).opts(
        color="color",
        cmap="viridis",
        size=3,
        width=500,
        height=400,
        title="Original 3D Swiss Roll",
        colorbar=True,
        backend="plotly",
    )

    # Learned 2D Latent Space - using bokeh backend
    # Create Points instead of Scatter to avoid kdims warning
    latent_2d = hv.Points(
        (z[:, 0], z[:, 1], color), kdims=["Latent Dim 1", "Latent Dim 2"], vdims=["color"]
    ).opts(
        color="color",
        cmap="viridis",
        size=5,
        width=500,
        height=400,
        title=title,
        aspect="equal",
        colorbar=True,
        show_grid=True,
        backend="bokeh",
    )

    # Reconstruction 3D Data - using plotly backend
    recon_3d = hv.Scatter3D(
        (x_recon[:, 0], x_recon[:, 1], x_recon[:, 2], color),
        kdims=["x", "y", "z"],
        vdims=["color"],
    ).opts(
        color="color",
        cmap="viridis",
        size=3,
        width=500,
        height=400,
        title="Reconstructed 3D Data",
        colorbar=True,
        backend="plotly",
    )

    # Save individual plots as HTML (portable and headless-friendly)
    hv.save(original_3d, f"{filename_prefix}_original_3d.html", backend="plotly")
    hv.save(latent_2d, f"{filename_prefix}_latent_2d.html", backend="bokeh")
    hv.save(recon_3d, f"{filename_prefix}_recon_3d.html", backend="plotly")

    print(f"Plots saved as {filename_prefix}_*.html")
    print(f"  - {filename_prefix}_original_3d.html (3D Swiss Roll)")
    print(f"  - {filename_prefix}_latent_2d.html (2D Latent Space)")
    print(f"  - {filename_prefix}_recon_3d.html (3D Reconstruction)")

    return original_3d, latent_2d, recon_3d


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Standard AE
    # standard_ae = SimpleAutoencoder(input_dim=3, latent_dim=LATENT_DIM).to(DEVICE)
    # train_ae(standard_ae, use_geometric_loss=False)
    # plot_results(standard_ae, "Latent Space of Standard AE", "standard_ae")

    # Geometric + L1 AE
    geometric_ae = SimpleAutoencoder(input_dim=3, latent_dim=LATENT_DIM).to(DEVICE)
    train_ae(geometric_ae, use_geometric_loss=True)
    plot_results(geometric_ae, "Latent Space of Geometric+L1 AE", "geometric_ae")
