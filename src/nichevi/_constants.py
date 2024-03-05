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
