from collections.abc import Iterator
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi.data._utils import _validate_adata_dataloader_input
from scvi.model._utils import scrna_raw_counts_properties
from scvi.model.base import (
    RNASeqMixin,
    _de_core,
)

# from scvi.model.base._utils import _de_core
from scvi.utils import de_dsp

from .differential_expression import _niche_de_core


class NicheRNASeqMixin(RNASeqMixin):
    @de_dsp.dedent
    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: list[str] | None = None,
        group2: str | None = None,
        idx1: list[int] | list[bool] | str | None = None,
        idx2: list[int] | list[bool] | str | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: list[str] | None = None,
        batchid2: list[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        weights: Literal["uniform", "importance"] | None = "uniform",
        filter_outlier_cells: bool = False,
        importance_weighting_kwargs: dict | None = None,
        ###### NicheSCVI specific ######
        radius: int | None = 50,
        k_nn: int | None = None,
        count_corruption: float | None = None,
        niche_mode: bool = True,
        n_restarts_optimizer_gpc: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for differential expression analysis.

        Implements ``'vanilla'`` DE :cite:p:`Lopez18` and ``'change'`` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        weights
            Weights to use for sampling. If `None`, defaults to `"uniform"`.
        filter_outlier_cells
            Whether to filter outlier cells with
            :meth:`~scvi.model.base.DifferentialComputation.filter_outlier_cells`.
        importance_weighting_kwargs
            Keyword arguments passed into
            :meth:`~scvi.model.base.RNASeqMixin._get_importance_weights`.
        radius
            Radius for NicheSCVI DE.
        k_nn
            Number of nearest neighbors for NicheSCVI DE.
        count_corruption
            Whether to corrupt the counts for NicheSCVI DE, and if so, by how much in terms of percentage of added noise.
        niche_mode
            Whether to use NicheSCVI DE or SCVI DE.
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)
        col_names = adata.var_names
        importance_weighting_kwargs = importance_weighting_kwargs or {}
        model_fn = partial(
            self.get_normalized_expression,
            return_numpy=True,
            n_samples=1,
            batch_size=batch_size,
            weights=weights,
            **importance_weighting_kwargs,
        )
        representation_fn = self.get_latent_representation if filter_outlier_cells else None

        if niche_mode:
            result = _niche_de_core(
                self.get_anndata_manager(adata, required=True),
                model_fn,
                representation_fn,
                groupby,
                group1,
                group2,
                idx1,
                idx2,
                all_stats,
                scrna_raw_counts_properties,
                col_names,
                mode,
                batchid1,
                batchid2,
                delta,
                batch_correction,
                fdr_target,
                silent,
                radius=radius,
                k_nn=k_nn,
                count_corruption=count_corruption,
                n_restarts_optimizer_gpc=n_restarts_optimizer_gpc,
                **kwargs,
            )

        else:
            result = _de_core(
                self.get_anndata_manager(adata, required=True),
                model_fn,
                representation_fn,
                groupby,
                group1,
                group2,
                idx1,
                idx2,
                all_stats,
                scrna_raw_counts_properties,
                col_names,
                mode,
                batchid1,
                batchid2,
                delta,
                batch_correction,
                fdr_target,
                silent,
                **kwargs,
            )

        return result

    @torch.inference_mode()
    def get_likelihood_parameters(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        n_samples: int | None = 1,
        give_mean: bool | None = False,
        batch_size: int | None = None,
        dataloader: Iterator[dict[str, torch.Tensor | None]] | None = None,
        **data_loader_kwargs,
    ) -> dict[str, np.ndarray]:
        r"""Estimates for the parameters of the likelihood :math:`p(x \mid z)`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of posterior samples to use for estimation.
        give_mean
            Return expected value of parameters or a samples
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        dataloader
            An iterator over minibatches of data on which to compute the metric. The minibatches
            should be formatted as a dictionary of :class:`~torch.Tensor` with keys as expected by
            the model. If ``None``, a dataloader is created from ``adata``.
        **data_loader_kwargs
            Keyword args for data loader.

        """
        _validate_adata_dataloader_input(self, adata, dataloader)

        if dataloader is None:
            adata = self._validate_anndata(adata)
            scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, **data_loader_kwargs)
        else:
            scdl = dataloader
            for param in [indices, batch_size, n_samples]:
                if param is not None:
                    Warning(
                        f"Using {param} after custom Dataloader was initialize is redundant, "
                        f"please re-initialize with selected {param}",
                    )

        dropout_list = []
        mean_list = []
        dispersion_list = []
        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            _, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )
            px = generative_outputs["px"]
            if self.module.gene_likelihood != "poisson":
                px_r = px.theta
                px_rate = px.mu
            else:
                px_rate = px.rate
            if self.module.gene_likelihood == "zinb":
                px_dropout = px.zi_probs
                dropout_list += [px_dropout.cpu().numpy()]
                dropout = np.concatenate(dropout_list, axis=-2)

            n_batch = px_rate.size(0) if n_samples == 1 else px_rate.size(1)
            if self.module.gene_likelihood != "poisson":
                px_r = px_r.cpu().numpy()
                if len(px_r.shape) == 1:
                    dispersion_list += [np.repeat(px_r[np.newaxis, :], n_batch, axis=0)]
                else:
                    dispersion_list += [px_r]
            mean_list += [px_rate.cpu().numpy()]

        means = np.concatenate(mean_list, axis=-2)
        if self.module.gene_likelihood != "poisson":
            dispersions = np.concatenate(dispersion_list, axis=-2)

        if give_mean and n_samples > 1:
            if self.module.gene_likelihood == "zinb":
                dropout = dropout.mean(0)
            if self.module.gene_likelihood != "poisson":
                dispersions = dispersions.mean(0)
            means = means.mean(0)

        return_dict = {}
        return_dict["mean"] = means

        if self.module.gene_likelihood == "zinb":
            return_dict["dropout"] = dropout
        if self.module.gene_likelihood != "poisson":
            return_dict["dispersions"] = dispersions

        return return_dict
    