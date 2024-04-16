from collections.abc import Iterable
from typing import Callable, Optional, Tuple

import torch
from scvi.nn import Decoder, FCLayers
from torch import nn
from torch.distributions import Dirichlet, Normal


def _identity(x):
    return x


# Encoder
class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
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
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.dist_encoder = nn.Linear(n_hidden, 2 * n_output)

        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)

        q_m, q_v = self.dist_encoder(q).chunk(2, dim=-1)
        q_v = torch.nn.Softplus()(q_v) + self.var_eps

        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


# Decoders
class DirichletDecoder(Decoder):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        concentration_eps: float = 1e-6,
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

        self.concentration_eps = concentration_eps

    def forward(self, x: torch.Tensor, *cat_list: int):
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)

        p_m = torch.nn.Softplus()(p_m) + self.concentration_eps

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
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__()

        self.n_niche_components = n_niche_components
        self.n_output = n_output
        self.var_eps = var_eps

        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
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
        p_v = torch.nn.Softplus()(self.var_decoder(p)) + self.var_eps

        if p.ndim == 2:
            p_m = p_m.view(p_m.shape[0], self.n_niche_components, self.n_output)
            p_v = p_v.view(p_v.shape[0], self.n_niche_components, self.n_output)

        elif p.ndim == 3:
            p_m = p_m.view(-1, p_m.shape[1], self.n_niche_components, self.n_output)
            p_v = p_v.view(-1, p_v.shape[1], self.n_niche_components, self.n_output)

        return p_m, p_v


class NicheDecoderAttention(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_niche_components: int,
        n_input_attention: int,
        n_heads: int = 1,
        n_cat_list: Iterable[int] = None,
        n_layers_proj: int = 1,
        n_hidden_proj: int = 64,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        var_eps=1e-4,
        n_hidden_dist_decoder: int | None = None,
        **kwargs,
    ):
        super().__init__()

        # standard scvi decoder > z_ = MLP(z | batch)
        # batch token?
        self.var_eps = var_eps

        # Input z | batch
        self.z_proj = FCLayers(
            n_in=n_input,
            n_out=n_input_attention,
            n_cat_list=n_cat_list,
            n_layers=n_layers_proj,
            n_hidden=n_hidden_proj,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=True,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.z_proj_linear = nn.Sequential(
            nn.Linear(n_input_attention, n_input_attention),
            nn.LayerNorm(n_input_attention),
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
            n_cat_list=None,
            n_out=n_input_attention,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            **kwargs,
        )

        self.layer_norm_decoder = nn.LayerNorm(n_input_attention)

        self.dist_decoder = nn.Linear(n_input_attention, 2 * n_output)

    def forward(self, z: torch.Tensor, *cat_list: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project the input

        z_proj = self.z_proj(z, *cat_list)
        z_proj = self.z_proj_linear(z_proj)

        # Embed all the cell types
        cell_type_embedding = self.layer_norm_cell_type_embedding(
            self.cell_type_embedding.weight
        )  # TODO test without layer norm also

        # Build the sequence of [latent,cell type embeddings]
        if z.ndim == 2:
            z_proj = z_proj.unsqueeze(1)
            cell_type_embedding = cell_type_embedding.expand(z_proj.size(0), -1, -1)
            qkv = torch.cat([z_proj, cell_type_embedding], dim=1)
        elif z.ndim == 3:
            z_proj = z_proj.unsqueeze(2)
            cell_type_embedding = cell_type_embedding.expand(z_proj.size(0), z_proj.size(1), -1, -1)
            qkv = torch.cat([z_proj, cell_type_embedding], dim=2)
            qkv = qkv.view(-1, qkv.size(2), qkv.size(3))

        # Apply the attention mechanism
        attention_output, attention_weights = self.attention_module(qkv, qkv, qkv)

        # Apply layer norm
        attention_output = self.layer_norm_attention_module(attention_output + qkv)

        if z.ndim == 3:
            attention_output = attention_output.view(z_proj.size(0), z_proj.size(1), -1, z_proj.size(3))

        # Decode the attention output
        decoded = self.decoder(attention_output)

        # Apply layer norm
        p = self.layer_norm_decoder(decoded + attention_output)

        p_ct = p[:, 1:, :] if z.ndim == 2 else p[:, :, 1:, :]

        p_m, p_v = self.dist_decoder(p_ct).chunk(2, dim=-1)
        p_v = torch.nn.Softplus()(p_v) + self.var_eps

        return p_m, p_v, attention_weights
