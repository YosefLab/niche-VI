import inspect
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, Optional, Union

import numpy as np
from scvi import REGISTRY_KEYS, settings
from scvi._types import Number
from scvi.model.base import DifferentialComputation
from scvi.model.base._differential import (
    densify,
    describe_continuous_distrib,
    estimate_delta,
    estimate_pseudocounts_offset,
    pairs_sampler,
)

logger = logging.getLogger(__name__)


class DifferentialComputation(DifferentialComputation):
    """Unified class for differential computation.

    This class takes a function from a model like `SCVI` or `TOTALVI` and takes outputs
    from this function with respect to the adata input and computed Bayes factors as
    described in :cite:p:`Lopez18`, :cite:p:`Xu21`, or :cite:p:`Boyeau19`.

    Parameters
    ----------
    model_fn
        Callable in model API to get values from.
    representation_fn
        Callable providing latent representations, e.g.,
        :meth:`~scvi.model.SCVI.get_latent_representation`, for scVI.
    adata_manager
        AnnDataManager created by :meth:`~scvi.model.SCVI.setup_anndata`.
    """

    def get_bayes_factors(
        self,
        idx1: list[bool] | np.ndarray,
        idx2: list[bool] | np.ndarray,
        mode: Literal["vanilla", "change"] = "vanilla",
        batchid1: Sequence[Number | str] | None = None,
        batchid2: Optional[Sequence[Union[Number, str]]] = None,
        use_observed_batches: Optional[bool] = False,
        n_samples: int = 5000,
        use_permutation: bool = False,
        m_permutation: int = 10000,
        change_fn: Optional[Union[str, Callable]] = None,
        m1_domain_fn: Optional[Callable] = None,
        delta: Optional[float] = 0.5,
        pseudocounts: Union[float, None] = 0.0,
        cred_interval_lvls: Optional[Union[list[float], np.ndarray]] = None,
    ) -> dict[str, np.ndarray]:
        r"""A unified method for differential expression inference.

        Two modes coexist:

        - The ``'vanilla'`` mode follows protocol described in :cite:p:`Lopez18` and
          :cite:p:`Xu21`.

            In this case, we perform hypothesis testing based on the hypotheses.

        .. math::
            M_1: h_1 > h_2 ~\text{and}~ M_2: h_1 \leq h_2.

        DE can then be based on the study of the Bayes factors

        .. math::
            \log p(M_1 | x_1, x_2) / p(M_2 | x_1, x_2).

        - The ``'change'`` mode (described in :cite:p:`Boyeau19`).

            This mode consists of estimating an effect size random variable (e.g., log fold-change)
            and performing Bayesian hypothesis testing on this variable. The `change_fn` function
            computes the effect size variable :math:`r` based on two inputs corresponding to the
            posterior quantities (e.g., normalized expression) in both populations.

        Hypotheses:

        .. math::
            M_1: r \in R_1 ~\text{(effect size r in region inducing differential expression)}

        .. math::
            M_2: r  \notin R_1 ~\text{(no differential expression)}

        To characterize the region :math:`R_1`, which induces DE, the user has two choices.

        1. A common case is when the region :math:`[-\delta, \delta]` does not induce differential
           expression. If the user specifies a threshold delta, we suppose that
           :math:`R_1 = \mathbb{R} \setminus [-\delta, \delta]`
        2. Specify an specific indicator function:

        .. math::
            f: \mathbb{R} \mapsto \{0, 1\} ~\text{s.t.}~ r \in R_1 ~\text{iff.}~ f(r) = 1.

        Decision-making can then be based on the estimates of

        .. math::
            p(M_1 \mid x_1, x_2).

        Both modes require to sample the posterior distributions.
        To that purpose, we sample the posterior in the following way:

        1. The posterior is sampled `n_samples` times for each subpopulation.
        2. For computational efficiency (posterior sampling is quite expensive), instead of
           comparing the obtained samples element-wise, we can permute posterior samples.
           Remember that computing the Bayes Factor requires sampling :math:`q(z_A \mid x_A)` and
           :math:`q(z_B \mid x_B)`.

        Currently, the code covers several batch handling configurations:

        1. If ``use_observed_batches=True``, then batch are considered as observations
           and cells' normalized means are conditioned on real batch observations.
        2. If case (cell group 1) and control (cell group 2) are conditioned on the same
           batch ids. This requires ``set(batchid1) == set(batchid2)`` or
           ``batchid1 == batchid2 === None``.
        3. If case and control are conditioned on different batch ids that do not intersect
           i.e., ``set(batchid1) != set(batchid2)`` and
           ``len(set(batchid1).intersection(set(batchid2))) == 0``.

        This function does not cover other cases yet and will warn users in such cases.

        Parameters
        ----------
        mode
            one of ["vanilla", "change"]
        idx1
            bool array masking subpopulation cells 1. Should be True where cell is
            from associated population
        idx2
            bool array masking subpopulation cells 2. Should be True where cell is
            from associated population
        batchid1
            List of batch ids for which you want to perform DE Analysis for
            subpopulation 1. By default, all ids are taken into account
        batchid2
            List of batch ids for which you want to perform DE Analysis for
            subpopulation 2. By default, all ids are taken into account
        use_observed_batches
            Whether posterior values are conditioned on observed
            batches
        n_samples
            Number of posterior samples
        use_permutation
            Activates step 2 described above.
            Simply formulated, pairs obtained from posterior sampling
            will be randomly permuted so that the number of pairs used
            to compute Bayes Factors becomes `m_permutation`.
        m_permutation
            Number of times we will "mix" posterior samples in step 2.
            Only makes sense when `use_permutation=True`
        change_fn
            function computing effect size based on both posterior values
        m1_domain_fn
            custom indicator function of effect size regions
            inducing differential expression
        delta
            specific case of region inducing differential expression.
            In this case, we suppose that :math:`R \setminus [-\delta, \delta]` does not induce
            differential expression (LFC case). If the provided value is `None`, then a proper
            threshold is determined from the distribution of LFCs accross genes.
        pseudocounts
            pseudocount offset used for the mode `change`.
            When None, observations from non-expressed genes are used to estimate its value.
        cred_interval_lvls
            List of credible interval levels to compute for the posterior
            LFC distribution

        Returns
        -------
        Differential expression properties

        """
        # if not np.array_equal(self.indices, np.arange(len(self.dataset))):
        #     warnings.warn(
        #         "Differential expression requires a Posterior object created with all indices."
        #     )
        eps = 1e-8
        # Normalized means sampling for both populations
        if self.representation_fn is not None:
            idx1 = self.filter_outlier_cells(idx1)
            idx2 = self.filter_outlier_cells(idx2)

        scales_batches_1 = self.scale_sampler(
            selection=idx1,
            batchid=batchid1,
            use_observed_batches=use_observed_batches,
            n_samples=n_samples,
        )
        scales_batches_2 = self.scale_sampler(
            selection=idx2,
            batchid=batchid2,
            use_observed_batches=use_observed_batches,
            n_samples=n_samples,
        )

        px_scale_mean1 = scales_batches_1["scale"].mean(axis=0)
        px_scale_mean2 = scales_batches_2["scale"].mean(axis=0)

        # Sampling pairs
        # The objective of code section below is to ensure than the samples of normalized
        # means we consider are conditioned on the same batch id
        batchid1_vals = np.unique(scales_batches_1["batch"])
        batchid2_vals = np.unique(scales_batches_2["batch"])

        create_pairs_from_same_batches = (set(batchid1_vals) == set(batchid2_vals)) and not use_observed_batches
        if create_pairs_from_same_batches:
            # First case: same batch normalization in two groups
            logger.debug("Same batches in both cell groups")
            n_batches = len(set(batchid1_vals))
            n_samples_per_batch = m_permutation // n_batches if m_permutation is not None else None
            logger.debug(f"Using {n_samples_per_batch} samples per batch for pair matching")
            scales_1 = []
            scales_2 = []
            for batch_val in set(batchid1_vals):
                # Select scale samples that originate from the same batch id
                scales_1_batch = scales_batches_1["scale"][scales_batches_1["batch"] == batch_val]
                scales_2_batch = scales_batches_2["scale"][scales_batches_2["batch"] == batch_val]

                # Create more pairs
                scales_1_local, scales_2_local = pairs_sampler(
                    scales_1_batch,
                    scales_2_batch,
                    use_permutation=use_permutation,
                    m_permutation=n_samples_per_batch,
                )
                scales_1.append(scales_1_local)
                scales_2.append(scales_2_local)
            scales_1 = np.concatenate(scales_1, axis=0)
            scales_2 = np.concatenate(scales_2, axis=0)
        else:
            logger.debug("Ignoring batch conditionings to compare means")
            if len(set(batchid1_vals).intersection(set(batchid2_vals))) >= 1:
                warnings.warn(
                    "Batchids of cells groups 1 and 2 are different but have an non-null "
                    "intersection. Specific handling of such situations is not "
                    "implemented yet and batch correction is not trustworthy.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            scales_1, scales_2 = pairs_sampler(
                scales_batches_1["scale"],
                scales_batches_2["scale"],
                use_permutation=use_permutation,
                m_permutation=m_permutation,
            )

        # Adding pseudocounts to the scales
        if pseudocounts is None:
            logger.debug("Estimating pseudocounts offet from the data")
            x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
            where_zero_a = densify(np.max(x[idx1], 0)) == 0
            where_zero_b = densify(np.max(x[idx2], 0)) == 0
            pseudocounts = estimate_pseudocounts_offset(
                scales_a=scales_1,
                scales_b=scales_2,
                where_zero_a=where_zero_a,
                where_zero_b=where_zero_b,
            )
        logger.debug(f"Using pseudocounts ~ {pseudocounts}")
        # Core of function: hypotheses testing based on the posterior samples we obtained above
        if mode == "vanilla":
            logger.debug("Differential expression using vanilla mode")
            proba_m1 = np.mean(scales_1 > scales_2, 0)
            proba_m2 = 1.0 - proba_m1
            res = {
                "proba_m1": proba_m1,
                "proba_m2": proba_m2,
                "bayes_factor": np.log(proba_m1 + eps) - np.log(proba_m2 + eps),
                "scale1": px_scale_mean1,
                "scale2": px_scale_mean2,
            }

        elif mode == "change":
            logger.debug("Differential expression using change mode")

            # step 1: Construct the change function
            def lfc(x, y):
                return np.log2(x + pseudocounts) - np.log2(y + pseudocounts)

            if change_fn == "log-fold" or change_fn is None:
                change_fn = lfc
            elif not callable(change_fn):
                raise ValueError("'change_fn' attribute not understood")

            # step2: Construct the DE area function
            if m1_domain_fn is None:

                def m1_domain_fn(samples):
                    delta_ = delta if delta is not None else estimate_delta(lfc_means=samples.mean(0))
                    logger.debug(f"Using delta ~ {delta_:.2f}")
                    # return np.abs(samples) >= delta_
                    samples_plus = samples >= delta_
                    samples_minus = samples < -delta_
                    return samples_plus, samples_minus

            change_fn_specs = inspect.getfullargspec(change_fn)
            domain_fn_specs = inspect.getfullargspec(m1_domain_fn)
            if (len(change_fn_specs.args) != 2) | (len(domain_fn_specs.args) != 1):
                raise ValueError(
                    "change_fn should take exactly two parameters as inputs; m1_domain_fn one " "parameter."
                )
            try:
                change_distribution = change_fn(scales_1, scales_2)
                # is_de = m1_domain_fn(change_distribution)
                is_de_plus, is_de_minus = m1_domain_fn(change_distribution)
                delta_ = estimate_delta(lfc_means=change_distribution.mean(0)) if delta is None else delta
            except TypeError as err:
                raise TypeError(
                    "change_fn or m1_domain_fn have has wrong properties."
                    "Please ensure that these functions have the right signatures and"
                    "outputs and that they can process numpy arrays"
                ) from err
            proba_m1 = np.mean(is_de_plus, 0)
            proba_m2 = np.mean(is_de_minus, 0)

            proba_de = np.maximum(proba_m1, proba_m2)
            change_distribution_props = describe_continuous_distrib(
                samples=change_distribution,
                credible_intervals_levels=cred_interval_lvls,
            )
            change_distribution_props = {"lfc_" + key: val for (key, val) in change_distribution_props.items()}

            res = dict(
                proba_de=proba_de,
                proba_not_de=1.0 - proba_de,
                # bayes_factor=np.log(proba_de + eps) - np.log(1.0 - proba_de + eps),
                bayes_factor=np.log(proba_m1 + eps) - np.log(proba_m2 + eps),
                scale1=px_scale_mean1,
                scale2=px_scale_mean2,
                pseudocounts=pseudocounts,
                delta=delta_,
                **change_distribution_props,
            )
        else:
            raise NotImplementedError(f"Mode {mode} not recognized")

        return res
