import collections
from typing import Callable, Iterable, List, Literal, Optional

import torch
from torch import nn
from torch.distributions import Normal, Dirichlet
from torch.nn import ModuleList

from scvi.nn import Decoder, FCLayers


class DirichletDecoder(Decoder):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        **kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_output=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, *cat_list: int, eps: float = 1e-6):
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)

        p_m = torch.nn.Softplus()(p_m) + eps

        dist = Dirichlet(p_m)

        return dist


class NicheDecoder(nn.Module):
    """Decodes data from latent space to LATENT NICHE space.

    ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    kwargs
        Keyword args for :class:`~scvi.module._base.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_niche_components: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        **kwargs,
    ):
        super().__init__()

        self.n_niche_components = n_niche_components
        self.n_output = n_output

        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            **kwargs,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output * n_niche_components)
        self.var_decoder = nn.Linear(n_hidden, n_output * n_niche_components)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        x
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple of :py:class:`torch.Tensor`
            Mean and variance tensors of shape ``(n_output,)``

        """
        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.nn.Softplus()(
            self.var_decoder(p)
        )  # changed exp to softplus todo add eps to p_v

        p_m = p_m.reshape(p_m.shape[0], self.n_niche_components, self.n_output)

        p_v = p_v.reshape(p_v.shape[0], self.n_niche_components, self.n_output)

        return p_m, p_v
