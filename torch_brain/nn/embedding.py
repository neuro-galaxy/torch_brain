import torch
import torch.nn as nn


class Embedding(nn.Embedding):
    r"""A simple extension of :class:`torch.nn.Embedding` to allow more control over
    the weights initializer. The learnable weights of the module of shape
    `(num_embeddings, embedding_dim)` are initialized from
    :math:`\mathcal{N}(0, \text{init_scale})`.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        init_scale (float, optional): standard deviation of the normal distribution used
            for the initialization. Defaults to 0.02, which is the default value used in
            most transformer models
        **kwargs: Additional arguments. Refer to the documentation of
            :class:`torch.nn.Embedding` for details
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_scale: float = 0.02,
        **kwargs,
    ):
        self.init_scale = init_scale
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.normal_(self.weight, mean=0, std=self.init_scale)
        self._fill_padding_idx_with_zero()
