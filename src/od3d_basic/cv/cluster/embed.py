import torch
from sklearn.manifold import TSNE


def pca(X, C=2, center=True, return_V=False):
    """
    Principal Component Analysis (PCA) is a linear dimensionality reduction
    Args:
        X (torch.Tensor): NxF

    Returns:
        X_embedded (torch.Tensor): NxC
    """

    # if q==None: q=min(6,N)
    batch_shape = X.shape[:-1]
    if X.dim() > 2:
        X_flat = X.clone().reshape(-1, X.shape[-1])
    else:
        X_flat = X

    _, _, pca_V = torch.pca_lowrank(X_flat, center=center, q=C)

    if return_V:
        return pca_V
    else:
        X_embedded = torch.mm(X_flat, pca_V[:, :C])
        if X.dim() > 2:
            X_embedded = X_embedded.reshape(*batch_shape, C)
        return X_embedded


def tsne(X, C=2):
    """
    t-distributed Stochastic Neighbor Embedding (t-SNE) is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.
    Args:
        X (torch.Tensor): NxF
    Returns:
        X_embedded (torch.Tensor): NxC
    """
    device = X.device

    X_embedded = torch.from_numpy(
        TSNE(n_components=C).fit_transform(X.detach().cpu().numpy()),
    ).to(device=device)
    return X_embedded
