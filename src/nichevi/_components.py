from collections.abc import Iterable

from typing import Tuple

import torch
from scvi.nn import Decoder, FCLayers
from torch import nn
from torch.distributions import Dirichlet


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
            dropout_rate=0,  # why ?
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


class NicheDecoderAttention(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_niche_components: int,
        n_input_attention: int,  # Size of the attention layer, should be equal to n_input as we add the attention to the input
        n_heads: int = 1,
        n_cat_list: Iterable[int] = None,
        n_layers_proj: int = 1,
        n_hidden_proj: int = 64,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(NicheDecoderAttention, self).__init__()

        # standard scvi decoder > z_ = MLP(z | batch)
        # batch token?

        # Input z | batch
        self.z_proj = FCLayers(
            n_in=n_input,
            n_out=n_input_attention,
            n_cat_list=n_cat_list,
            n_layers=n_layers_proj,
            n_hidden=n_hidden_proj,
            use_activation=False,
            use_batch_norm=False,
            use_layer_norm=True,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.cell_type_embedding = nn.Embedding(
            num_embeddings=n_niche_components,
            embedding_dim=n_input_attention,
        )

        self.layer_norm_cell_type_embedding = nn.LayerNorm(n_input_attention)

        self.attention_module = nn.MultiheadAttention(
            embed_dim=n_input_attention,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.layer_norm_attention_module = nn.LayerNorm(n_input_attention)

        self.decoder = FCLayers(
            n_in=n_input_attention,
            n_out=n_input_attention,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            **kwargs,
        )

        self.layer_norm_decoder = nn.LayerNorm(n_input_attention)

        self.mean_decoder = nn.Linear(
            n_input_attention, n_output
        )  # input 2 x n_input_attention then chunk
        self.var_decoder = nn.Linear(n_input_attention, n_output)

    def forward(
        self, z: torch.Tensor, *cat_list: int, eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Project the input
        z_proj = self.z_proj(z)

        # Embed all the cell types
        cell_type_embedding = self.layer_norm_cell_type_embedding(
            self.cell_type_embedding.weight
        )  # TODO test without layer norm also

        # cell_type_embeddings = self.cell_type_embedding.weight
        cell_type_embedding = cell_type_embedding.unsqueeze(0).expand(
            z_proj.size(0), -1, -1
        )
        z_proj = z_proj.unsqueeze(1)

        # Build the sequence of [latent,cell type embeddings]
        qkv = torch.cat([z_proj, cell_type_embedding], dim=1)

        # Apply the attention mechanism
        attention_output, _ = self.attention_module(qkv, qkv, qkv)

        # Apply layer norm
        attention_output = self.layer_norm_attention_module(attention_output + qkv)

        # Decode the attention output
        decoded = self.decoder(attention_output, *cat_list)

        # Apply layer norm
        p = self.layer_norm_decoder(decoded + attention_output)

        # Decode the mean and variance
        p_m = self.mean_decoder(p[:, 1:, :])
        p_v = torch.nn.Softplus()(self.var_decoder(p[:, 1:, :])) + eps

        return p_m, p_v
