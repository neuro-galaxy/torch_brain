# NuCLR Example

Embeds every neuron of a recording with a pretrained
[NuCLR](https://arxiv.org/abs/2512.01199) model, and visualizes the embeddings
with UMAP.

The pretrained checkpoints ([`nerdslab/ibl-bwb-nuclr`](https://huggingface.co/nerdslab/ibl-bwb-nuclr))
were trained on the [IBL Brain-Wide Bench](https://github.com/brainbench-org/ibl-bwb)
pretraining recordings. By default, this example embeds a held-out IBL recording
whose probe goes through somatosensory cortex, hippocampus and thalamus. The
model only sees spikes, yet units from different brain regions, and even
neighboring thalamic nuclei, end up in different parts of the embedding space.

### Setup

In addition to installing `torch_brain`, you need to:
```bash
pip install huggingface_hub omegaconf umap-learn matplotlib
```

### Running

```bash
python embed_units.py
```

This downloads the checkpoint (~130 MB) from Hugging Face and the recording
(~540 MB) into `data/`, then saves the UMAP to `nuclr_umap.png`. On a GPU, embedding the
recording takes a few seconds. On a CPU it is much slower, so embed only the
start of the recording:

```bash
python embed_units.py --max-minutes 10
```

Other held-out recordings can be chosen with `--recording-id`. Run
`python embed_units.py --help` for all options.
