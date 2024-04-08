import numpy as np
import pandas as pd

# import pytest
import scvi
import torch
from scvi.data import _constants, synthetic_iid
from scvi.data._compat import LEGACY_REGISTRY_KEY_MAP, registry_from_setup_dict
from scvi.model.utils import mde

from nichevi import nicheSCVI

LEGACY_REGISTRY_KEYS = set(LEGACY_REGISTRY_KEY_MAP.values())
LEGACY_SETUP_DICT = {
    "scvi_version": "0.0.0",
    "categorical_mappings": {
        "_scvi_batch": {
            "original_key": "testbatch",
            "mapping": np.array(["batch_0", "batch_1"], dtype=object),
        },
        "_scvi_labels": {
            "original_key": "testlabels",
            "mapping": np.array(["label_0", "label_1", "label_2"], dtype=object),
        },
    },
    "extra_categoricals": {
        "mappings": {
            "cat1": np.array([0, 1, 2, 3, 4]),
            "cat2": np.array([0, 1, 2, 3, 4]),
        },
        "keys": ["cat1", "cat2"],
        "n_cats_per_key": [5, 5],
    },
    "extra_continuous_keys": np.array(["cont1", "cont2"], dtype=object),
    "data_registry": {
        "X": {"attr_name": "X", "attr_key": None},
        "batch_indices": {"attr_name": "obs", "attr_key": "_scvi_batch"},
        "labels": {"attr_name": "obs", "attr_key": "_scvi_labels"},
        "cat_covs": {
            "attr_name": "obsm",
            "attr_key": "_scvi_extra_categoricals",
        },
        "cont_covs": {
            "attr_name": "obsm",
            "attr_key": "_scvi_extra_continuous",
        },
    },
    "summary_stats": {
        "n_batch": 2,
        "n_cells": 400,
        "n_vars": 100,
        "n_labels": 3,
        "n_proteins": 0,
        "n_continuous_covs": 2,
    },
}


N_LAYERS = 1
N_LATENT = 2
LIKELIHOOD = "nb"
K_NN = 5
N_HEADS = 1
ATTENTION_DECODER = False
N_EPOCHS_NICHEVI = 1


def test_nichevi():
    adata = synthetic_iid(
        batch_size=256,
        n_genes=100,
        n_proteins=0,
        n_regions=0,
        n_batches=2,
        n_labels=3,
        dropout_ratio=0.5,
        generate_coordinates=True,
        sparse_format=None,
        return_mudata=False,
    )

    adata.obsm["qz1_m"] = np.random.normal(size=(adata.shape[0], N_LATENT))
    adata.layers["counts"] = adata.X.copy()

    nicheSCVI.preprocessing_anndata(
        adata,
        niche_composition_key="neighborhood_composition",
        niche_indexes_key="niche_indexes",
        niche_distances_key="niche_distances",
        niche_type_key="niche_type",
        niche_treshold=None,
        cell_type_for_niches=None,
        label_key="labels",
        sample_key="batch",
        cell_coordinates_key="coordinates",
        k_nn=K_NN,
        latent_mean_key="qz1_m",
        latent_mean_niche_key="qz1_m_niche_ct",
    )

    nicheSCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="batch",
        labels_key="labels",
        niche_composition_key="neighborhood_composition",
        niche_indexes_key="niche_indexes",
        niche_distances_key="niche_distances",
        latent_mean_key="qz1_m",
        latent_mean_ct_key="qz1_m_niche_ct",
    )

    niche_setup = {
        "r1_kl1_c1_n1": {
            "cell_rec_weight": 1.0,
            "niche_rec_weight": 1.0,
            "niche_compo_weight": 1.0,
            "latent_kl_weight": 1.0,
        },
        "r0_kl0_c0_n0": {
            "cell_rec_weight": 0.0,
            "niche_rec_weight": 0.0,
            "niche_compo_weight": 0.0,
            "latent_kl_weight": 0.0,
        },
    }

    setup_dict = niche_setup["r0_kl0_c0_n0"]

    nichevae = nicheSCVI(
        adata,
        cell_rec_weight=setup_dict["cell_rec_weight"],
        latent_kl_weight=setup_dict["latent_kl_weight"],
        niche_rec_weight=setup_dict["niche_rec_weight"],
        compo_rec_weight=setup_dict["niche_compo_weight"],
        gene_likelihood=LIKELIHOOD,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_layers_niche=1,
        n_layers_compo=1,
        n_hidden_niche=48,
        n_hidden_compo=48,
        n_latent=N_LATENT,
        use_batch_norm="both",
        use_layer_norm="none",
        attention_decoder=ATTENTION_DECODER,
    )

    nichevae.train(
        max_epochs=N_EPOCHS_NICHEVI,
        train_size=0.8,
        validation_size=0.2,
        early_stopping=True,
        check_val_every_n_epoch=1,
        # plan_kwargs={
        #     "lr": setup.LR,
        #     "n_epochs_kl_warmup": setup.KL_WARMUP,
        #     "n_epochs_spatial_warmup": setup.SPATIAL_WARMUP,
        #     # n_epochs_kl_warmup=N_EPOCHS_NICHEVI,
        #     "min_spatial_weight": setup.MIN_SPATIAL_WEIGHT,
        #     "max_spatial_weight": setup.MAX_SPATIAL_WEIGHT,
        #     "optimizer": setup.OPTIMIZER,
        #     "weight_decay": setup.WEIGHT_DECAY,
        #     "reduce_lr_on_plateau": setup.REDUCE_LR_ON_PLATEAU,
        # },
    )

    print("Finished training")

    nichevae.get_elbo(indices=nichevae.validation_indices)
    nichevae.get_normalized_expression()
    nichevae.get_latent_representation()
    # nichevae.predict_neighborhood()  # specific to nicheSCVI
    # nichevae.predict_niche_activation()  # specific to nicheSCVI
    nichevae.differential_expression(
        groupby="labels",
        group1="label_1",
        batch_correction=False,
    )
    nichevae.differential_expression(
        groupby="labels",
        group1="label_1",
        group2="label_2",
        batch_correction=False,
    )


test_nichevi()

print("nicheSCVI test passed")
