"""Utilities for collating samples of variable length into a batch.

The core entry point is :func:`collate`, a drop-in extension of PyTorch's
:obj:`~torch.utils.data.default_collate` that handles tensors of differing
lengths. Instead of configuring it directly, you wrap objects with a *collation
recipe*:

- :func:`pad` (and its variants) to pad a dimension to the batch maximum.
- :func:`chain` to concatenate along the first dimension.
- Optionally, :func:`track_mask` or :func:`track_batch` to emit a padding mask
  or per-element batch index.

When used with PyTorch's default :obj:`~torch.utils.data.DataLoader`,
:func:`collate` applies the wrapped recipes when batching. Below is
an example of how these are used in practice.

.. code:: python

   import torch
   from torch_brain.datasets import Dataset
   from torch_brain.batching import collate, chain, pad

   class MyDataset(Dataset):
       ...

       def __getitem__(self, index):
           # logic to gather data
           ...

           return {
               "spike_times": chain(spike_times),
               "spike_seqlen": len(spike_times),
               "query_times": pad(query_times),
               "query_mask": track_mask(query_times),
           }

   ds = MyDataset(...)

   loader = torch.utils.data.DataLoader(
       dataset=ds,
       sampler=...,
       collate_fn=collate,  # <- specify collate from torch_brain.batching
       num_workers=4,
       ...
   )

"""

from .collate import (
    chain,
    collate,
    pad,
    pad2d,
    pad2d8,
    pad8,
    track_batch,
    track_mask,
    track_mask2d,
    track_mask2d8,
    track_mask8,
)

__all__ = [
    "chain",
    "collate",
    "pad",
    "pad8",
    "pad2d",
    "pad2d8",
    "track_batch",
    "track_mask",
    "track_mask8",
    "track_mask2d",
    "track_mask2d8",
]

# see docs/source/api_reference.py
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Collate Function",
            "autosummary": [
                "collate",
            ]
        },
        {
            "title": "Collation Recipes",
            "description": (
                "Wrap objects with these functions "
                "to collate them in different ways."
            ),
            "autosummary": [
                "chain",
                "pad",
                "pad8",
                "pad2d",
                "pad2d8",
            ],
        },
        {
            "title": "Mask & Index Tracking",
            "description": (
                "Use these alongside a collation wrapper to track padding masks "
                "or batch indices for a particular array or tensor."
            ),
            "autosummary": [
                "track_batch",
                "track_mask",
                "track_mask8",
                "track_mask2d",
                "track_mask2d8",
            ],
        },
    ],
}
