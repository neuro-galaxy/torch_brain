# NuCLR Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuro-galaxy/torch_brain/blob/main/examples/nuclr/pretrained_embeddings.ipynb)

This example shows how to embed the neurons of a recording with a pretrained
[NuCLR](https://arxiv.org/abs/2512.01199) model.

We use the [`nerdslab/ibl-bwb-nuclr`](https://huggingface.co/nerdslab/ibl-bwb-nuclr)
checkpoints, pretrained on the [IBL Brain-Wide Bench](https://github.com/brainbench-org/ibl-bwb).

## Running

Open [`pretrained_embeddings.ipynb`](pretrained_embeddings.ipynb) in Colab with the
badge above, or run it locally. With **torch_brain** installed:

```bash
pip install huggingface_hub omegaconf umap-learn matplotlib scikit-learn jupyter
jupyter notebook pretrained_embeddings.ipynb
```

The notebook downloads a ~130MB checkpoint and a ~540MB recording. Once they are
downloaded, it runs in ~20 seconds on an RTX5090, and ~2.5 minutes on a 32-core CPU.
