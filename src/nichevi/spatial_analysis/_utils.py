def _lisi_per_cell_type(adatype, embedding_key, label_key, n_neighbors=90, perplexity=30):
    from scib_metrics import clisi_knn
    from scib_metrics.nearest_neighbors import NeighborsResults
    from sklearn.neighbors import NearestNeighbors

    X, labels = adatype.obsm[embedding_key], adatype.obs[label_key]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)
    neigh_results = NeighborsResults(indices=inds, distances=dists)
    lisi_res = clisi_knn(neigh_results, labels, perplexity=perplexity, return_median=False)
    return lisi_res
