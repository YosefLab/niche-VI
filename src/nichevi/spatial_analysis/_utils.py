# def _lisi_per_cell_type(adatype, embedding_key, label_key, n_neighbors=90, perplexity=30):
#     from scib_metrics import lisi_knn
#     from scib_metrics.nearest_neighbors import NeighborsResults
#     from sklearn.neighbors import NearestNeighbors

#     X, labels = adatype.obsm[embedding_key], adatype.obs[label_key]
#     nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X)
#     dists, inds = nbrs.kneighbors(X)
#     neigh_results = NeighborsResults(indices=inds, distances=dists)
#     lisi_res = lisi_knn(neigh_results, labels, perplexity=perplexity)
#     return lisi_res

import numpy as np
import pandas as pd

# from rich import print
from scib_metrics import lisi_knn
from scib_metrics.nearest_neighbors import NeighborsResults
from sklearn.neighbors import NearestNeighbors


def _lisi_per_cell_type(adatype, embedding_key, label_key, n_neighbors=90, perplexity=30):
    # adapted from https://github.com/YosefLab/scib-metrics/blob/main/src/scib_metrics/metrics/_lisi.py
    X, labels = adatype.obsm[embedding_key], adatype.obs[label_key]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)
    neigh_results = NeighborsResults(indices=inds, distances=dists)
    lisi_res = lisi_knn(neigh_results, labels, perplexity=perplexity)

    nlabels = len(np.unique(labels))
    print("Number of labels:", nlabels)
    print("Min", np.min(lisi_res), "Max", np.max(lisi_res))
    clisi = (nlabels - lisi_res) / (nlabels - 1)

    return clisi


def _integration_lisi(adatype, embedding_key, label_key, n_neighbors=90, perplexity=30) -> float:
    # adapted from https://github.com/YosefLab/scib-metrics/blob/main/src/scib_metrics/metrics/_lisi.py
    X, labels = adatype.obsm[embedding_key], adatype.obs[label_key]
    print(f"Computing iLISI for {embedding_key} based on {label_key}")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)
    neigh_results = NeighborsResults(indices=inds, distances=dists)
    lisi_res = lisi_knn(neigh_results, labels, perplexity=perplexity)
    # ilisi = np.nanmedian(lisi)
    nbatches = len(np.unique(labels))
    print("Number of batches:", nbatches)
    print("Min", np.min(lisi_res), "Max", np.max(lisi_res))
    lisi_res = np.clip(lisi_res, 1.0, nbatches)
    ilisi = lisi_res
    ilisi = (lisi_res - 1) / (nbatches - 1)
    assert np.all((ilisi >= 0) & (ilisi <= 1)), "iLISI score should be between 0 and 1"
    print("Median iLISI:", np.nanmedian(ilisi))
    print("Mean iLISI:", np.nanmean(ilisi))
    return ilisi


def plot_history(models_history: dict, figures_folder: str):
    import matplotlib.pyplot as plt

    print(models_history.keys())

    for key in models_history.keys():
        list_of_registered_keys = models_history[key].keys()
        keys_to_plot = (
            [keys for keys in list_of_registered_keys if "validation" in keys] + ["kl_weight"]
            if "kl_weight" in list_of_registered_keys
            else [keys for keys in list_of_registered_keys if "validation" in keys]
        )
        series_to_plot = [models_history[key][keys] for keys in keys_to_plot]

        n_plots = len(series_to_plot)

        fig, axs = plt.subplots(nrows=n_plots, ncols=1, figsize=(6, 5 * n_plots))

        # Flatten the axs array to access each subplot individually
        axs = axs.flatten()

        # Plot each series in a subplot
        for i, ax in enumerate(axs[:n_plots]):
            if i < len(series_to_plot):
                series_to_plot[i].plot(ax=ax)
                ax.set_title(keys_to_plot[i])

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        # Save the figure to a file
        plt.savefig(f"{figures_folder}/{key}_history.png", dpi=80, bbox_inches="tight")
