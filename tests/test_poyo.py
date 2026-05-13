import pytest
import torch
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from torch_brain.models.poyo import POYO

READOUT_DIM = 2


@pytest.fixture(scope="session")
def pretrained_checkpoint():
    """Download the pretrained POYO model checkpoint from S3."""
    checkpoint_filename = "poyo_mp.ckpt"
    checkpoint_path = Path(checkpoint_filename)

    # Only download if the file doesn't exist
    if not checkpoint_path.exists():
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        # Get file size for progress bar
        file_info = s3.head_object(
            Bucket="torch-brain", Key=f"model-zoo/{checkpoint_filename}"
        )
        file_size = file_info["ContentLength"]

        # Download with progress bar
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {checkpoint_filename}",
        ) as pbar:
            s3.download_file(
                "torch-brain",
                f"model-zoo/{checkpoint_filename}",
                checkpoint_filename,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )

    yield str(checkpoint_path)

    # Cleanup: remove the downloaded checkpoint after tests
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def test_load_pretrained_basic(pretrained_checkpoint):
    """Test loading a pretrained POYO model with default settings"""

    # Load the pretrained model
    model = POYO.load_pretrained(checkpoint_path=pretrained_checkpoint)

    assert isinstance(model, POYO)

    assert hasattr(model, "sequence_length")
    assert hasattr(model, "latent_step")
    assert hasattr(model, "num_latents_per_step")

    assert hasattr(model, "readout")
    assert isinstance(model.readout, torch.nn.Linear)
    assert model.readout.out_features == READOUT_DIM


def test_load_pretrained_forward_pass(pretrained_checkpoint):
    """Test that the loaded model can perform a forward pass"""

    model = POYO.load_pretrained(checkpoint_path=pretrained_checkpoint)

    # Reinitialize the vocabs for the new units and sessions
    model.unit_emb.extend_vocab([f"unit_{i}" for i in range(100)])
    model.unit_emb.subset_vocab([f"unit_{i}" for i in range(100)])
    model.session_emb.extend_vocab([f"session_{i}" for i in range(10)])
    model.session_emb.subset_vocab([f"session_{i}" for i in range(10)])

    # Create dummy input data
    batch_size = 2
    n_in = 10
    n_latent = (
        int(model.sequence_length / model.latent_step) * model.num_latents_per_step
    )
    n_out = 4

    inputs = {
        "input_unit_index": torch.randint(0, 100, (batch_size, n_in)),
        "input_timestamps": torch.rand(batch_size, n_in) * model.sequence_length,
        "input_token_type": torch.randint(0, 4, (batch_size, n_in)),
        "input_mask": torch.ones(batch_size, n_in, dtype=torch.bool),
        "latent_index": torch.arange(n_latent).repeat(batch_size, 1)
        % model.num_latents_per_step,
        "latent_timestamps": torch.linspace(0, model.sequence_length, n_latent).repeat(
            batch_size, 1
        ),
        "session_index": torch.zeros(batch_size, dtype=torch.long),
        "output_timestamps": torch.rand(batch_size, n_out) * model.sequence_length,
    }

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Verify output shape
    assert outputs.shape == (batch_size, n_out, READOUT_DIM)
