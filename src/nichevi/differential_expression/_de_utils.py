from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def normalize_counts(
    X: np.ndarray,
    target_sum: float = 1e4,
) -> np.ndarray:
    return target_sum * X / X.sum(axis=1, keepdims=True)


def probabilistic_rounding(array: np.ndarray) -> np.ndarray:
    """Round an array of floats to integers probabilistically."""
    uniform_random = np.random.rand(*array.shape)

    array_int = np.floor(array)

    array_float = array - array_int

    proba_mask = array_float > uniform_random

    array_int[proba_mask] += 1

    return array_int


def corrupt_counts(
    origin_counts: csr_matrix,
    neighbors_counts: csr_matrix,
    target_corruption: float = 0.1,
    rounding: Literal["ceil", "floor", "round", "random"] = "random",
) -> csr_matrix:
    x_uncorr_sum = origin_counts.sum(axis=1)
    x_niche1_sum = neighbors_counts.sum(axis=1)

    corruption_weights = np.divide(
        target_corruption * x_uncorr_sum,
        x_niche1_sum,
        out=np.zeros_like(x_niche1_sum, dtype=float),
        where=x_niche1_sum != 0,
    )

    x_corr = neighbors_counts.multiply(corruption_weights)

    if rounding == "ceil":
        x_corr = x_corr.ceil().astype(int)
    elif rounding == "floor":
        x_corr = x_corr.floor().astype(int)
    elif rounding == "round":
        x_corr.data = x_corr.data.round().astype(int)
    elif rounding == "random":
        x_corr.data = probabilistic_rounding(x_corr.data)

    x_corr = origin_counts + x_corr

    return x_corr


def adjusted_nearest_neighbors(
    cell_samples: np.array,
    cell_coordinates: np.array,
    cell_labels: np.array,
    radius: int | None = None,
    k_nn: int | None = None,
    return_sparse: bool = True,
):
    from scipy.sparse import block_diag
    from sklearn.neighbors import NearestNeighbors

    adjacency_matrices = []

    for sample in np.unique(cell_samples):
        mask = np.squeeze(cell_samples == sample, axis=1)  # n_cells
        sample_coord = cell_coordinates[mask]  # n_cells_sample_i x 2
        sample_cell_types = np.squeeze(cell_labels[mask], axis=1)  # n_cells_sample_i

        # build a dict of masks for each cell type
        cell_type_masks = {
            cell_type: sample_cell_types == cell_type for cell_type in np.unique(sample_cell_types)
        }  # change to equal!!

        # make it a df
        cell_type_masks_df = pd.DataFrame(
            cell_type_masks
        ).transpose()  # n_cell_types x n_cells_sample_i

        # # # then build the mask matrix of the sample
        # mask_matrix = cell_type_masks_df.loc[
        #     sample_cell_types
        # ].values  # n_cells_sample_i x n_cells_sample_i

        # get the size in MB of the mask matrix cell_type_masks_df.loc[sample_cell_types]:
        # print(f"Size of the mask matrix (dense): {mask_matrix.nbytes / 1e6:.2f} MB")

        # Size of the sparse matrix in MB
        # sparse_size_mb = (
        #     mask_matrix_sparse.data.nbytes  # Size of the non-zero data
        #     + mask_matrix_sparse.indptr.nbytes  # Size of the index pointer array
        #     + mask_matrix_sparse.indices.nbytes  # Size of the indices array
        # ) / (1024**2)
        # print(f"Size of the mask matrix (sparse): {sparse_size_mb:.2f} MB")

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

        A_adjusted = A.copy()
        # Process each label type at once using CSR format
        for ids in cell_type_masks_df.index:
            # Get the mask of rows we want to modify
            row_mask = np.isin(np.arange(A.shape[0]), ids)

            # Process only the affected rows
            for i in np.where(row_mask)[0]:
                # Get row slice
                row_start = A_adjusted.indptr[i]
                row_end = A_adjusted.indptr[i + 1]

                # Get columns that need to be zeroed (same label connections)
                cols_to_zero = np.isin(A_adjusted.indices[row_start:row_end], ids)

                # Zero out these connections
                A_adjusted.data[row_start:row_end][cols_to_zero] = 0

        # A_adjusted = A.multiply(mask_matrix)

        A_adjusted.eliminate_zeros()

        adjacency_matrices.append(A_adjusted.astype(bool, copy=False))

    adjacency_matrix = block_diag(adjacency_matrices, format="csr")

    row_counts = np.diff(adjacency_matrix.indptr)
    # print mean and std of number of neighbors with a sigma letter for the std, round to 2 decimals:
    print(f"Mean number of neighbors: {np.mean(row_counts):.1f} ± {np.std(row_counts):.1f}")

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


def _fdr_de_prediction(posterior_probas: pd.Series, fdr: float = 0.05) -> pd.Series:
    """Compute posterior expected FDR and tag features as DE."""
    if not posterior_probas.ndim == 1:
        raise ValueError("posterior_probas should be 1-dimensional")
    original_index = posterior_probas.index
    sorted_pgs = posterior_probas.sort_values(ascending=False)
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))
    d = (cumulative_fdr <= fdr).sum()
    is_pred_de = pd.Series(np.zeros_like(cumulative_fdr).astype(bool), index=sorted_pgs.index)
    is_pred_de.iloc[:d] = True
    is_pred_de = is_pred_de.loc[original_index]
    return is_pred_de
