import hashlib
import os
from pathlib import Path

import boto3
import numpy as np
import pytest
import torch
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from torch.testing import assert_close
from tqdm import tqdm

from torch_brain.batching import collate
from torch_brain.data import ArrayDict, Data, Interval, IrregularTimeSeries
from torch_brain.models.poyo import POYO, POYOInputFn

# The checkpoints all have the same readout dimension
# (since they were trained for hand-velocity decoding)
READOUT_DIM = 2

# Updatable SHA256 checksum of the pretrained POYO model checkpoint
MANIFEST = Path(__file__).parent / "poyo_checkpoint.sha256"
EXPECTED_SHA, NAME = MANIFEST.read_text().split()

BUCKET = "torch-brain"
KEY = "model-zoo/poyo_mp.ckpt"
CACHE_DIR = Path(
    os.environ.get("TORCH_BRAIN_CACHE", Path.home() / ".cache" / "torch_brain")
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="session")
def pretrained_checkpoint():
    """Download the pretrained POYO model checkpoint from S3."""
    checkpoint_path = CACHE_DIR / Path(KEY).name

    if checkpoint_path.exists() and _sha256(checkpoint_path) != EXPECTED_SHA:
        checkpoint_path.unlink()

    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint_path.with_name(f"{checkpoint_path.name}.{os.getpid()}.tmp")
        try:
            s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

            # Get file size for progress bar
            file_info = s3.head_object(Bucket=BUCKET, Key=KEY)
            file_size = file_info["ContentLength"]

            # Download with progress bar
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {checkpoint_path.name}",
            ) as pbar:
                s3.download_file(
                    BUCKET,
                    KEY,
                    str(tmp),
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                )

            if _sha256(tmp) != EXPECTED_SHA:
                pytest.fail(
                    f"Downloaded POYO checkpoint has incorrect SHA256 checksum: {_sha256(tmp)}"
                )

            tmp.replace(checkpoint_path)

        except (BotoCoreError, ClientError) as e:
            pytest.fail(f"POYO checkpoint not cached and download failed: {e}")

        finally:
            tmp.unlink(missing_ok=True)

    return checkpoint_path


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


UNIT_IDS = ["a/u0", "a/u1", "a/u2", "b/u0"]
SESSION_IDS = ["a", "b"]


def make_data(timestamps, unit_index, unit_ids, session_id="a", duration=1.0):
    domain = Interval(0.0, duration)
    return Data(
        spikes=IrregularTimeSeries(
            timestamps=np.asarray(timestamps, dtype=np.float64),
            unit_index=np.asarray(unit_index, dtype=np.int64),
            domain=domain,
        ),
        units=ArrayDict(id=np.array(unit_ids)),
        session=Data(id=session_id),
        domain=domain,
    )


def random_data(num_spikes, seed=0):
    rng = np.random.default_rng(seed)
    return make_data(
        np.sort(rng.uniform(0, 1.0, num_spikes)),
        rng.integers(0, 3, num_spikes),
        UNIT_IDS[:3],
    )


def unpad(batch, key, i=0):
    """Valid (unpadded) input tokens of the ``i``-th sample of a collated batch."""
    return batch[key][i][batch["input_mask"][i]]


@pytest.fixture
def model():
    torch.manual_seed(0)
    model = POYO(
        sequence_length=1.0,
        latent_step=0.25,
        num_latents_per_step=3,
        dim_out=2,
        dim=32,
        depth=1,
        dim_head=16,
        self_heads=2,
    ).eval()
    model.unit_emb.initialize_vocab(UNIT_IDS)
    model.session_emb.initialize_vocab(SESSION_IDS)
    return model


class TestPOYOInputFn:
    @staticmethod
    def make_input_fn():
        return POYOInputFn(
            sequence_length=1.0,
            latent_step=0.25,
            num_latents_per_step=3,
            unit_vocab={"NA": 0} | {uid: i + 1 for i, uid in enumerate(UNIT_IDS)},
            session_vocab={"NA": 0, "a": 1, "b": 2},
        )

    def test_start_end_tokens(self):
        """Each unit gets a start and an end token, followed by one token per spike."""
        data = make_data([0.2, 0.5, 0.7], [1, 0, 1], ["a/u0", "a/u1"])
        batch = collate([self.make_input_fn()(data)])

        assert unpad(batch, "input_token_type").tolist() == [1, 2, 1, 2, 0, 0, 0]
        assert_close(
            unpad(batch, "input_timestamps"),
            torch.tensor([0.0, 1.0, 0.0, 1.0, 0.2, 0.5, 0.7], dtype=torch.float64),
        )

    def test_unit_index_maps_to_vocab(self):
        """Local unit indices of a sample are mapped to the global vocab indices."""
        # Units of the sample are a shuffled subset of the vocab
        data = make_data([0.2, 0.5, 0.7], [0, 1, 0], ["a/u2", "a/u0"])
        batch = collate([self.make_input_fn()(data)])

        assert unpad(batch, "input_unit_index").tolist() == [3, 3, 1, 1, 3, 1, 3]

    def test_latent_tokens(self):
        """``num_latents_per_step`` latents are placed at the center of every step."""
        batch = collate([self.make_input_fn()(random_data(10))])

        assert batch["latent_index"][0].tolist() == [0, 1, 2] * 4
        assert_close(
            batch["latent_timestamps"][0],
            torch.tensor(
                [0.125, 0.375, 0.625, 0.875], dtype=torch.float64
            ).repeat_interleave(3),
        )


class TestGetInputFn:
    def test_uninitialized_vocab_raises(self):
        model = POYO(sequence_length=1.0, latent_step=0.25, dim_out=2, dim=32)
        with pytest.raises(ValueError, match="must be initialized"):
            model.get_input_fn()

    @torch.no_grad()
    def test_batch_independent(self, model):
        """A sample's outputs don't depend on the other samples in its batch."""
        input_fn = model.get_input_fn()
        a = input_fn(random_data(20, seed=0))
        b = input_fn(random_data(45, seed=1))
        output_timestamps = torch.rand(2, 5)

        batch_a, batch_ab = collate([a]), collate([a, b])
        out_a = model(**batch_a, output_timestamps=output_timestamps[:1])
        out_ab = model(**batch_ab, output_timestamps=output_timestamps)
        assert_close(out_ab[:1], out_a)


# TODO: remove when fully deprecating `tokenize` method
class TestTokenize:
    def test_tracks_vocab_changes(self, model):
        """``tokenize`` uses the model's current vocabs, both when they are extended
        in place and when they are replaced."""

        def assert_matches_input_fn(data):
            out = collate([model.tokenize(data)])
            expected = collate([model.get_input_fn()(data)])
            for key in expected:
                assert_close(out[key], expected[key])

        with pytest.warns(UserWarning, match="soft deprecated"):
            assert_matches_input_fn(random_data(10))

        model.unit_emb.extend_vocab(["c/u0"])
        model.session_emb.extend_vocab(["c"])
        data = make_data([0.2, 0.5], [0, 1], ["c/u0", "a/u1"], session_id="c")
        assert_matches_input_fn(data)

        model.unit_emb.subset_vocab(["a/u1", "c/u0"])
        assert_matches_input_fn(data)
