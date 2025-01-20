import numpy as np
import pandas as pd
from anndata import AnnData


def compute_jaccard(
    adata_CD4: AnnData,
    key1: str,
    key2: str,
    jaccard: bool = True,
) -> pd.DataFrame:
    """Compute the Jaccard index between two leiden clusterings"""
    if jaccard:
        dict_1 = {}

        OBS = key1

        for category in adata_CD4.obs[OBS].unique():
            dict_1[category] = adata_CD4.obs.index[adata_CD4.obs[OBS] == category].tolist()

        dict_2 = {}

        OBS = key2

        for category in adata_CD4.obs[OBS].unique():
            dict_2[category] = adata_CD4.obs.index[adata_CD4.obs[OBS] == category].tolist()

        jaccard_sim_matrix_hepa_all_mods = np.zeros((len(dict_1.keys()), len(dict_2.keys())))
        for i in range(len(dict_1.keys())):
            for j in range(len(dict_2.keys())):
                key_i = list(dict_1.keys())[i]
                key_j = list(dict_2.keys())[j]
                a = dict_1[key_i]
                b = dict_2[key_j]
                jaccard_sim_matrix_hepa_all_mods[i, j] = len(list(set(a) & set(b))) / len(
                    list(set(a) | set(b))
                )
        jaccard_sim_matrix_hepa_all_mods_df = pd.DataFrame(
            jaccard_sim_matrix_hepa_all_mods,
            index=dict_1.keys(),
            columns=dict_2.keys(),
        )

        proportions = jaccard_sim_matrix_hepa_all_mods_df.T

    else:
        # Calculate proportions
        proportions = (
            (
                adata_CD4.obs[[key1, key2]].groupby([key1, key2]).size()
                / adata_CD4.obs[[key1, key2]].groupby(key1).size()
            )
            .unstack()
            .T
        )

    return proportions


def get_gene_percentiles_list(
    adata: AnnData, gene_list: list[str], p: float, layer: str = None
) -> list:
    """
    Calculate the p-percentile of gene expression for a list of genes in an AnnData object.

    Parameters
    ----------
        adata (AnnData): The AnnData object containing expression data.
        gene_list (list): List of gene names for which to compute percentiles.
        p (float): Percentile to compute (between 0 and 100).
        layer (str or None): The layer from which to retrieve expression data.
                             If None, uses `adata.X`.

    Returns
    -------
        list: A list of p-percentile values for the genes, in the same order as gene_list.
              If a gene is not found, its value will be `None`.
    """
    percentiles = []

    for gene in gene_list:
        if gene in adata.var_names:
            if layer:
                data = adata[:, gene].layers[layer].flatten()
            else:
                data = adata[:, gene].X.flatten()

            # Compute the percentile
            percentiles.append(np.percentile(data, p))
        else:
            percentiles.append(None)  # Handle genes not in adata.var_names

    return percentiles
