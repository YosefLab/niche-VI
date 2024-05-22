from collections.abc import Iterable as IterableClass

import numpy as np
import pandas as pd
from scvi import REGISTRY_KEYS
from scvi.model.base._differential import DifferentialComputation
from scvi.model.base._utils import _fdr_de_prediction, _prepare_obs
from scvi.utils import track

from nichevi import NICHEVI_REGISTRY_KEYS

from ._de_utils import _get_nonzero_indices_from_rows, adjusted_nearest_neighbors


def _de_core(
    adata_manager,
    model_fn,
    representation_fn,
    groupby,
    group1,
    group2,
    idx1,
    idx2,
    all_stats,
    all_stats_fn,
    col_names,
    mode,
    batchid1,
    batchid2,
    delta,
    batch_correction,
    fdr,
    silent,
    ###### NicheSCVI specific ######
    # sample_key="sample",
    # cell_coordinates_key="spatial",
    # label_key="cell_type",
    radius=50,
    k_nn=None,
    count_corruption: float | None = None,
    **kwargs,
):
    """Internal function for DE interface."""
    adata = adata_manager.adata
    # adata = adata
    if group1 is None and idx1 is None:
        group1 = adata.obs[groupby].astype("category").cat.categories.tolist()
        if len(group1) == 1:
            raise ValueError("Only a single group in the data. Can't run DE on a single group.")

    if not isinstance(group1, IterableClass) or isinstance(group1, str):
        group1 = [group1]

    # make a temp obs key using indices
    temp_key = None
    if idx1 is not None:
        obs_col, group1, group2 = _prepare_obs(idx1, idx2, adata)
        temp_key = "_scvi_temp_de"
        adata.obs[temp_key] = obs_col
        groupby = temp_key

    cell_samples = adata_manager.get_from_registry(NICHEVI_REGISTRY_KEYS.SAMPLE_KEY)
    cell_labels = adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
    cell_coordinates = adata_manager.get_from_registry(NICHEVI_REGISTRY_KEYS.CELL_COORDINATES_KEY)

    # cell_samples = adata.obs[sample_key].values
    # cell_labels = adata.obs[label_key].values
    # cell_coordinates = adata.obsm[cell_coordinates_key]

    A = adjusted_nearest_neighbors(
        adata,
        cell_samples=cell_samples,
        cell_coordinates=cell_coordinates,
        cell_labels=cell_labels,
        radius=radius,
        k_nn=k_nn,
        return_sparse=True,
    )
    # df_results = []
    DE_results = (
        {
            "group1_group2": [],
            "niche1_group2": [],
        }
        if count_corruption is None
        else {
            "group1_group2": [],
            "group1_corrupted1": [],
        }
    )
    # if group2 is not None:
    #     DE_results["group1_niche2"] = []

    dc = DifferentialComputation(model_fn, representation_fn, adata_manager)
    for g1 in track(
        group1,
        description="DE...",
        disable=silent,
    ):
        cell_idx1 = (adata.obs[groupby] == g1).to_numpy().ravel()
        neighbors_idx1 = _get_nonzero_indices_from_rows(A, cell_idx1)

        if count_corruption is not None:
            x_uncorr = adata.layers["counts"][cell_idx1]
            x_niche1 = A[cell_idx1] @ adata.layers["counts"]

            x_uncorr_sum = x_uncorr.sum(axis=1)
            x_niche1_sum = x_niche1.sum(axis=1)

            corruption_weights = np.divide(
                count_corruption * x_uncorr_sum,
                x_niche1_sum,
                out=np.zeros_like(x_niche1_sum, dtype=float),
                where=x_niche1_sum != 0,
            )

            x_corr = x_niche1.multiply(corruption_weights)
            x_corr = x_uncorr + x_corr.ceil().astype(int)

            # Trick to avoid double counting. Only works if len(neighbors_idx1) > len(cell_idx1) which I assume is the case
            neighbors_idx1 = neighbors_idx1[: cell_idx1.sum()]

            # Save the original counts of this index
            x_original = adata.layers["counts"][neighbors_idx1]

            # Then replace with the corrupted counts
            dc.adata.layers["counts"][neighbors_idx1] = x_corr

        if group2 is None:
            cell_idx2 = ~cell_idx1
            # neighbors_idx2 = None
            DE_indices = (
                {
                    "group1_group2": [cell_idx1, cell_idx2],
                    "group1_corrupted1": [cell_idx1, neighbors_idx1],
                }
                if count_corruption
                else {
                    "group1_group2": [cell_idx1, cell_idx2],
                    "niche1_group2": [neighbors_idx1, cell_idx2],
                }
            )
            DE_group_names = (
                {
                    "group1_group2": [g1, "Rest"],
                    "group1_corrupted1": [g1, f"{g1}_corrupted"],
                }
                if count_corruption
                else {
                    "group1_group2": [g1, "Rest"],
                    "niche1_group2": [f"{g1}_neighbors", "Rest"],
                }
            )
        else:
            cell_idx2 = (adata.obs[groupby] == group2).to_numpy().ravel()
            # neighbors_idx2 = _get_nonzero_indices_from_rows(A, cell_idx2)
            DE_indices = (
                {
                    "group1_group2": [cell_idx1, cell_idx2],
                    "niche1_group2": [neighbors_idx1, cell_idx2],
                    # "group1_niche1": [cell_idx1, neighbors_idx1],
                    # "group2_niche2": [cell_idx2, neighbors_idx2],
                }
                if count_corruption is None
                else {
                    "group1_group2": [cell_idx1, cell_idx2],
                    "group1_corrupted1": [cell_idx1, neighbors_idx1],
                }
            )
            DE_group_names = (
                {
                    "group1_group2": [g1, group2],
                    "niche1_group2": [f"{g1}_neighbors", group2],
                    # "group1_niche1": [g1, f"{g1}_neighbors"],
                    # "group2_niche2": [group2, f"{group2}_neighbors"],
                }
                if count_corruption is None
                else {
                    "group1_group2": [g1, group2],
                    "group1_corrupted1": [g1, f"{g1}_corrupted"],
                }
            )

        for comparison, [cell_idx_a, cell_idx_b] in DE_indices.items():
            print(f"Running DE for {comparison}")

            all_info = dc.get_bayes_factors(
                cell_idx_a,
                cell_idx_b,
                mode=mode,
                delta=delta,
                batchid1=batchid1,
                batchid2=batchid2,
                use_observed_batches=not batch_correction,
                **kwargs,
            )

            if all_stats is True:
                genes_properties_dict = all_stats_fn(adata_manager, cell_idx_a, cell_idx_b)
                all_info = {**all_info, **genes_properties_dict}

            res = pd.DataFrame(all_info, index=col_names)
            sort_key = "proba_de" if mode == "change" else "bayes_factor"
            res = res.sort_values(by=sort_key, ascending=False)
            if mode == "change":
                res[f"is_de_fdr_{fdr}"] = _fdr_de_prediction(res["proba_de"], fdr=fdr)
            if idx1 is None:
                # g2 = "Rest" if group2 is None else group2
                g1_name = DE_group_names[comparison][0]
                g2_name = DE_group_names[comparison][1]
                res["comparison"] = f"{g1_name} vs {g2_name}"
                res["group1"] = g1_name
                res["group2"] = g2_name
            DE_results[comparison].append(res)

        if count_corruption is not None:
            # Restore the original counts
            dc.adata.layers["counts"][neighbors_idx1] = x_original

    if temp_key is not None:
        del adata.obs[temp_key]

    DE_results["group1_group2"] = pd.concat(DE_results["group1_group2"], axis=0)
    idx_g1_g2 = DE_results["group1_group2"].index

    for groups in list(DE_results.keys())[1:]:
        group_DE_result = DE_results[groups]
        DE_results[groups] = pd.concat(group_DE_result, axis=0).reindex(idx_g1_g2)

    return DE_results
