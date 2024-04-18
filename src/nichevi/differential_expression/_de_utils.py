from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
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


def adjusted_nearest_neighbors(
    adata: ad.AnnData,
    cell_samples: np.array,
    cell_coordinates: np.array,
    cell_labels: np.array,
    radius: int | None = None,
    k_nn: int | None = None,
    return_sparse: bool = True,
):
    from scipy.sparse import block_diag
    from sklearn.neighbors import NearestNeighbors

    # cell_types = adata.obs[labels].copy().values
    # cell_coords = adata.obsm[cell_coordinates].copy()
    # cell_samples = adata.obs[samples].copy().values

    adjacency_matrices = []

    for sample in np.unique(cell_samples):
        mask = np.squeeze(cell_samples == sample, axis=1)
        sample_coord = cell_coordinates[mask]
        sample_cell_types = np.squeeze(cell_labels[mask], axis=1)

        # build a dict of masks for each cell type
        cell_type_masks = {
            cell_type: sample_cell_types != cell_type
            for cell_type in np.unique(sample_cell_types)
        }

        # make it a df
        cell_type_masks_df = pd.DataFrame(cell_type_masks).transpose()

        # then build the mask matrix of the sample
        mask_matrix = cell_type_masks_df.loc[sample_cell_types].values

        if radius is not None:
            nn = NearestNeighbors(radius=radius)
            nn.fit(sample_coord)
            A = nn.radius_neighbors_graph(sample_coord)
        elif k_nn is not None:
            nn = NearestNeighbors(n_neighbors=k_nn + 1)
            nn.fit(sample_coord)
            A = nn.kneighbors_graph(sample_coord)
        else:
            raise ValueError("Either radius or k_nn must be provided.")

        A_adjusted = A.multiply(mask_matrix)

        A_adjusted.eliminate_zeros()

        adjacency_matrices.append(A_adjusted.astype(bool, copy=False))

    adjacency_matrix = block_diag(adjacency_matrices, format="csr")

    if return_sparse:
        return adjacency_matrix

    return adjacency_matrix.toarray()


def _get_nonzero_indices_from_rows(csr_matrix, row_idx):
    return np.unique(csr_matrix[row_idx].indices)


def get_connectivity_distribution(csr_matrix):
    # Get the number of non-zero entries per row
    row_counts = np.diff(csr_matrix.indptr)

    # Create the histogram
    fig, ax = plt.subplots()
    ax.hist(row_counts, bins=np.max(row_counts) + 1, density=True)
    ax.set_xlabel("Number of non-zero entries")
    ax.set_ylabel("Number of rows")
    ax.set_title("Histogram of non-zero entries per row")
    plt.show()
