from typing import Optional, Literal
import scipy.sparse as sp
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix


def normalize_counts(
    X: np.ndarray,
    target_sum: float = 1e4,
) -> np.ndarray:
    return target_sum * X / X.sum(axis=1, keepdims=True)


def corrupt_counts(
    adata: ad.AnnData,
    niche_indexes_key: str,
    niche_distances_key: str,
    use_layer: str | None = None,
    add_layer: str = "corrupted_counts",
    k_nn: int = 7,
    bandwidth: Literal["mean", "median", "none"] = "median",
    save_weights: bool = False,
    target_sum: float = 1e4,
    spatial_weight: float = 1.0,
    log1p: bool = False,
):
    distance_matrix = adata.obsm[niche_distances_key][:, :k_nn]
    idx = adata.obsm[niche_indexes_key][:, :k_nn]

    if bandwidth == "mean":
        bandwidth = np.mean(np.min(distance_matrix, axis=1))
    elif bandwidth == "median":
        # bandwidth = np.median(np.min(distance_matrix, axis=1))
        bandwidth = np.median(distance_matrix, axis=1)
    else:
        bandwidth = np.ones(distance_matrix.shape[0])

    if use_layer is not None:
        counts = adata.layers[use_layer].copy()
    else:
        counts = adata.X.copy()

    if sp.issparse(counts):
        counts = counts.toarray()

    _exp_argument = -((distance_matrix / bandwidth[:, None]) ** 2)
    _exp_argument = np.exp(_exp_argument)
    _exp_argument = _exp_argument / np.sum(_exp_argument, axis=1)[:, None]

    _normalized_counts = normalize_counts(counts, target_sum=target_sum)

    weighted_neighbors = np.einsum("ij,ijk->ik", _exp_argument, _normalized_counts[idx])

    # _normalized_neighbors_counts = normalize_counts(
    #     weighted_neighbors, target_sum=target_sum
    # )

    corrupted_counts = (
        _normalized_counts
        + spatial_weight * weighted_neighbors
        # _normalized_neighbors_counts
    )

    _normalized_corr_counts = normalize_counts(corrupted_counts, target_sum=target_sum)

    corrupted_counts = np.log1p(_normalized_corr_counts)
    unc_counts = np.log1p(_normalized_counts)

    adata.layers[add_layer] = csr_matrix(corrupted_counts)
    adata.layers["uncorrupted_counts"] = csr_matrix(unc_counts)

    if save_weights:
        adata.obsm["corruption_weights"] = _exp_argument

    return None
