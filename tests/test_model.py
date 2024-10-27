import numpy as np

# import pytest
from scvi.data import _constants, synthetic_iid

from nichevi import nicheSCVI

N_LAYERS = 1
N_LATENT = 15
LIKELIHOOD = "nb"
K_NN = 5
N_HEADS = None
N_EPOCHS_NICHEVI = 2
N_TOKENS = 10
USE_BATCH_NORM = False


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
    # adata.obsm["qz1_m"] = adata.X.copy()
    adata.layers["counts"] = adata.X.copy()

    setup_kwargs = {
        "sample_key": "batch",
        "labels_key": "labels",
        "cell_coordinates_key": "coordinates",
        "expression_embedding_key": "qz1_m",
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    nicheSCVI.preprocessing_anndata(
        adata,
        k_nn=K_NN,
        **setup_kwargs,
    )

    nicheSCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="batch",
        **setup_kwargs,
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
        n_tokens_decoder=N_TOKENS,
        n_layers_niche=2,
        n_layers_compo=1,
        n_hidden_niche=48,
        n_hidden_compo=48,
        n_latent=N_LATENT,
        use_batch_norm="both" if USE_BATCH_NORM else "none",
        use_layer_norm="none" if USE_BATCH_NORM else "both",
        ###
        prior_mixture=True,
        # prior_mixture_k = 20,
        semisupervised=True,
        linear_classifier=True,
    )

    nichevae.train(
        max_epochs=N_EPOCHS_NICHEVI,
        train_size=0.8,
        validation_size=0.2,
        early_stopping=True,
        check_val_every_n_epoch=1,
        accelerator="cpu",
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

    print(nichevae.history.keys())
    nichevae.get_elbo(indices=nichevae.validation_indices)
    nichevae.get_composition_error(return_mean=False, indices=nichevae.validation_indices)
    nichevae.get_niche_error(return_mean=False, indices=nichevae.validation_indices)
    nichevae.get_normalized_expression()
    nichevae.get_latent_representation()
    nichevae.predict_neighborhood()  # specific to nicheSCVI
    # nichevae.predict_niche_activation()  # specific to nicheSCVI

    nichevae.differential_expression(
        groupby="labels",
        group1="label_1",
        group2="label_2",
        batch_correction=False,
        # sample_key="batch",
        # cell_coordinates_key="coordinates",
        # label_key="labels",
        radius=None,
        k_nn=5,
        count_corruption=None,
    )
    nichevae.differential_expression(
        groupby="labels",
        group1="label_1",
        group2="label_2",
        batch_correction=False,
        # sample_key="batch",
        # cell_coordinates_key="coordinates",
        # label_key="labels",
        radius=50,
        k_nn=None,
        count_corruption=None,
    )


test_nichevi()

print("nicheSCVI test passed")
