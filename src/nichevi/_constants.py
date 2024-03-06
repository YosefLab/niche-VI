from typing import NamedTuple


class _NICHEVI_REGISTRY_KEYS_NT(NamedTuple):
    NICHE_COMPOSITION_KEY: str = "niche_composition"
    Z1_MEAN_KEY: str = "latent_mean"
    Z1_VAR_KEY: str = "latent_var"
    NICHE_INDEXES_KEY: str = "niche_indexes"
    Z1_MEAN_CT_KEY: str = "latent_mean_ct_key"
    Z1_VAR_CT_KEY: str = "latent_var_ct_key"
    Z1_MEAN_KNN_KEY: str = "latent_mean_knn_key"


NICHEVI_REGISTRY_KEYS = _NICHEVI_REGISTRY_KEYS_NT()


class _NICHEVI_MODULE_KEYS(NamedTuple):
    # generative model
    NICHE_MEAN: str = "niche_mean"
    NICHE_MEAN: str = "niche_variance"
    P_NICHE_COMPOSITION: str = "niche_composition"
    P_NICHE_EXPRESSION: str = "niche_expression"
    # loss
    NLL_NICHE_COMPOSITION_KEY: str = "niche_compo"
    NLL_NICHE_EXPRESSION_KEY: str = "niche_reconst"


NICHEVI_MODULE_KEYS = _NICHEVI_MODULE_KEYS()
