# NuCLR Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuro-galaxy/torch_brain/blob/main/examples/nuclr/pretrained_embeddings.ipynb)

The [`pretrained_embeddings.ipynb`](pretrained_embeddings.ipynb) notebook embeds
every neuron of a recording with a pretrained
[NuCLR](https://arxiv.org/abs/2512.01199) model, and visualizes the embeddings
with UMAP.

The pretrained checkpoints ([`nerdslab/ibl-bwb-nuclr`](https://huggingface.co/nerdslab/ibl-bwb-nuclr))
were trained on the [IBL Brain-Wide Bench](https://github.com/brainbench-org/ibl-bwb)
pretraining recordings. The notebook embeds a held-out IBL recording whose probe
goes through somatosensory cortex, hippocampus and thalamus. The model only sees
spikes, yet units from different brain regions, and even neighboring thalamic
nuclei, end up in different parts of the embedding space.

To run it locally, install `torch_brain` and:
```bash
pip install huggingface_hub omegaconf umap-learn matplotlib scikit-learn jupyter
```
