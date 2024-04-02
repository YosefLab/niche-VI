from typing import List, Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from anndata import AnnData



def compute_jaccard(
    adata_CD4: AnnData,
    key1: str,
    key2: str,
    jaccard: bool = True,
) -> pd.DataFrame:
    """
    Compute the Jaccard index between two leiden clusterings
    """

    if jaccard:
        dict_1 = {}

        OBS = key1

        for category in adata_CD4.obs[OBS].unique():
            dict_1[category] = adata_CD4.obs.index[
                adata_CD4.obs[OBS] == category
            ].tolist()

        dict_2 = {}

        OBS = key2

        for category in adata_CD4.obs[OBS].unique():
            dict_2[category] = adata_CD4.obs.index[
                adata_CD4.obs[OBS] == category
            ].tolist()

        jaccard_sim_matrix_hepa_all_mods = np.zeros(
            (len(dict_1.keys()), len(dict_2.keys()))
        )
        for i in range(len(dict_1.keys())):
            for j in range(len(dict_2.keys())):
                key_i = list(dict_1.keys())[i]
                key_j = list(dict_2.keys())[j]
                a = dict_1[key_i]
                b = dict_2[key_j]
                jaccard_sim_matrix_hepa_all_mods[i, j] = len(
                    list(set(a) & set(b))
                ) / len(list(set(a) | set(b)))
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
