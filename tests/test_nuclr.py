import numpy as np
import pytest
import torch
from torch.testing import assert_close

from torch_brain.batching import collate
from torch_brain.data import ArrayDict, Data, Interval, IrregularTimeSeries
from torch_brain.models import NuCLR
from torch_brain.models.nuclr import NuCLRInputFn, _attention

CTX_DURATION = 1.0
BIN_SIZE = 0.02
NUM_BINS = 50


def make_data(timestamps, unit_index, intervals=((0.0, 1.0),), probe_id="abab"):
    domain = Interval(
        np.array([s for s, _ in intervals]), np.array([e for _, e in intervals])
    )
    return Data(
        spikes=IrregularTimeSeries(
            timestamps=np.asarray(timestamps, dtype=np.float64),
            unit_index=np.asarray(unit_index, dtype=np.int64),
            domain=domain,
        ),
        units=ArrayDict(
            id=np.array([f"unit_{i}" for i in range(len(probe_id))]),
            probe_id=np.array(list(probe_id)),
        ),
        domain=domain,
    )


def random_data(duration, num_units=4, rate=30.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(rate * duration * num_units)
    timestamps = np.sort(rng.uniform(0, duration, n))
    unit_index = rng.integers(0, num_units, n)
    return make_data(timestamps, unit_index, ((0.0, duration),))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NuCLR(
        ctx_duration=CTX_DURATION,
        latent_step=0.2,
        bin_size=BIN_SIZE,
        dim=32,
        self_heads=2,
        t_layers=1,
        st_layers=1,
    ).eval()


class TestNuCLRInputFn:
    def test_binning(self):
        data = make_data([0.05, 0.07, 0.45, 0.95], [0, 0, 2, 0])
        batch = collate([NuCLRInputFn(1.0, 0.1, group_by=None)(data)])

        expected = torch.zeros(2, 10, dtype=batch["bins"].dtype)
        expected[0, 0], expected[0, 9], expected[1, 4] = 2, 1, 1
        assert_close(batch["bins"], expected)

    def test_spikes_outside_window_ignored(self):
        data = make_data([-0.05, 0.5, 1.0, 1.05], [1, 0, 1, 1])
        batch = collate([NuCLRInputFn(1.0, 0.1, group_by=None)(data)])

        assert batch["unit_ids"].tolist() == ["unit_0"]
        assert batch["bins"].sum() == 1

    def test_silent_units_dropped(self):
        data = make_data([0.1, 0.2, 0.3], [1, 3, 3])
        batch = collate([NuCLRInputFn(1.0, 0.1, group_by=None)(data)])

        assert batch["unit_ids"].tolist() == ["unit_1", "unit_3"]
        assert batch["num_units"].tolist() == [2]

    def test_grouping(self):
        """Units are sorted by group, and all outputs follow the same unit order."""
        # Unit i spikes i + 1 times, so each row of bins is identifiable.
        unit_index = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        data = make_data(np.linspace(0.01, 0.09, 10), unit_index)
        batch = collate([NuCLRInputFn(1.0, 0.1, group_by="probe_id")(data)])

        assert batch["unit_ids"].tolist() == ["unit_0", "unit_2", "unit_1", "unit_3"]
        assert batch["group_ids"].tolist() == ["a", "a", "b", "b"]
        assert batch["group_lengths"].tolist() == [2, 2]
        assert batch["bins"][:, 0].tolist() == [1, 3, 2, 4]

    def test_group_by_none(self):
        data = make_data([0.1, 0.2, 0.3, 0.4], [0, 1, 2, 3])
        batch = collate([NuCLRInputFn(1.0, 0.1, group_by=None)(data)])

        assert batch["unit_ids"].tolist() == ["unit_0", "unit_1", "unit_2", "unit_3"]
        assert batch["group_lengths"].tolist() == [4]

    def test_longer_than_ctx_raises(self):
        data = make_data([0.1], [0], ((0.0, 1.5),))
        with pytest.raises(ValueError, match="longer than ctx_duration"):
            NuCLRInputFn(1.0, 0.1)(data)

    def test_shorter_than_ctx_zero_padded(self):
        data = make_data([0.15, 0.45], [0, 0], ((0.0, 0.5),))
        batch = collate([NuCLRInputFn(1.0, 0.1)(data)])

        assert batch["bins"].shape == (1, 10)
        assert batch["bins"][0].tolist() == [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]


class TestNuCLRForward:
    def test_output_shape(self, model):
        """Output is ``(U, dim)``; ``dim_head`` defaults to ``dim // self_heads``."""
        bins = torch.randint(0, 3, (5, NUM_BINS))
        out = model(bins, torch.tensor([2, 3]))

        assert out.shape == (5, 32)
        assert model.t_blocks[0][0].to_qkv.weight.shape == (3 * 32, 32)

    def test_wrong_num_bins_raises(self, model):
        with pytest.raises(ValueError, match="bins per unit"):
            model(torch.zeros(3, NUM_BINS - 1), torch.tensor([3]))

    @torch.no_grad()
    def test_groups_independent(self, model):
        """Spatial attention stays within a group: changing the second group's bins
        doesn't change the first group's outputs."""
        bins = torch.randint(0, 3, (6, NUM_BINS))
        group_lengths = torch.tensor([3, 3])
        other = bins.clone()
        other[3:] = torch.randint(0, 3, (3, NUM_BINS))

        out, out_other = model(bins, group_lengths), model(other, group_lengths)
        assert_close(out[:3], out_other[:3])
        assert not torch.allclose(out[3:], out_other[3:])

    @torch.no_grad()
    def test_batch_independent(self, model):
        """A sample's outputs don't depend on the other samples in its batch."""
        data = random_data(2.0)
        input_fn = model.get_input_fn()
        a = input_fn(data.slice(0.0, 1.0))
        b = input_fn(data.slice(1.0, 2.0))

        batch_a, batch_ab = collate([a]), collate([a, b])
        out_a = model(batch_a["bins"], batch_a["group_lengths"])
        out_ab = model(batch_ab["bins"], batch_ab["group_lengths"])
        assert_close(out_ab[: len(out_a)], out_a)


class TestVarlenAttention:
    @staticmethod
    def _per_sequence(q, k, v, seqlen):
        bounds = torch.cumsum(seqlen, 0).tolist()
        return torch.cat(
            [
                _attention._attention(q[s:e], k[s:e], v[s:e])
                for s, e in zip([0, *bounds[:-1]], bounds, strict=True)
            ]
        )

    def test_fallback_matches_per_sequence(self):
        """The padded SDPA fallback equals attending over each sequence separately."""
        seqlen = torch.tensor([3, 5, 1])
        q, k, v = torch.randn(3, 9, 2, 8).unbind(0)

        out = _attention._varlen_attention(q, k, v, seqlen)
        assert_close(out, self._per_sequence(q, k, v, seqlen))

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or _attention.varlen_attn is None
        or torch.cuda.get_device_capability() < (8, 0),
        reason="Requires torch>=2.10 and an Ampere or newer GPU.",
    )
    def test_varlen_attn_matches_fallback(self, monkeypatch):
        """torch's ``varlen_attn`` gives the same outputs as the padded fallback."""
        seqlen = torch.tensor([3, 5, 1], device="cuda")
        q, k, v = torch.randn(3, 9, 2, 64, device="cuda", dtype=torch.float16).unbind(0)

        out = _attention._varlen_attention(q, k, v, seqlen)
        monkeypatch.setattr(_attention, "varlen_attn", None)
        expected = _attention._varlen_attention(q, k, v, seqlen)
        assert_close(out, expected, atol=1e-3, rtol=1e-3)


class TestEmbed:
    """Tests for ``NuCLR.embed``."""

    def test_matches_manual_average(self, model):
        """Each unit's embedding is the mean of its ``forward`` outputs over the
        windows in which it spikes."""
        data = random_data(1.5)  # Exactly 2 windows: [0, 1] and [0.5, 1.5].
        input_fn = model.get_input_fn()

        total = torch.zeros(4, 32, dtype=torch.float64)
        count = torch.zeros(4)
        with torch.no_grad():
            for start in (0.0, 0.5):
                batch = collate([input_fn(data.slice(start, start + CTX_DURATION))])
                z = model(batch["bins"], batch["group_lengths"])
                idx = torch.tensor([int(u.split("_")[1]) for u in batch["unit_ids"]])
                total.index_add_(0, idx, z.double())
                count.index_add_(0, idx, torch.ones(len(idx)))

        emb = model.embed(data)
        assert emb.dtype == torch.float64
        assert_close(emb, total / count[:, None], atol=1e-6, rtol=1e-5)

    def test_silent_units_nan(self, model):
        data = make_data([0.1, 0.2, 0.3], [0, 1, 2])
        emb = model.embed(data)

        assert emb[:3].isfinite().all()
        assert emb[3].isnan().all()

    @pytest.mark.parametrize("training", [True, False])
    def test_restores_training_mode(self, model, training):
        model.train(training)
        model.embed(random_data(1.0))
        assert model.training == training

    def test_batch_size_invariance(self, model):
        data = random_data(5.0)
        assert_close(
            model.embed(data, batch_size=1),
            model.embed(data, batch_size=32),
            atol=1e-6,
            rtol=1e-5,
        )

    def test_silent_batches(self, model):
        """Batches of windows where no unit spikes are skipped instead of crashing."""
        # Only the first batch of windows has spikes.
        data = make_data([0.1, 0.2, 0.3, 0.4], [0, 1, 2, 3], ((0.0, 5.0),))
        emb = model.embed(data, batch_size=2)
        assert emb.isfinite().all()

    def test_short_intervals_skipped(self, model):
        # unit_3 only spikes in the 0.5s-long interval.
        data = make_data([0.1, 0.2, 0.3, 3.2], [0, 1, 2, 3], ((0.0, 2.0), (3.0, 3.5)))
        emb = model.embed(data)

        assert emb[:3].isfinite().all()
        assert emb[3].isnan().all()

    def test_all_short_raises(self, model):
        """Raises if every interval of the domain is shorter than ``ctx_duration``."""
        data = make_data([0.1], [0], ((0.0, 0.5),))
        with pytest.raises(ValueError):
            model.embed(data)


class TestCheckpoint:
    def test_state_dict_matches_pretrained(self):
        """The pretrained config's keys and shapes match the checkpoints in
        nerdslab/ibl-bwb-nuclr, so they still load."""
        attn = {
            "norm.weight": (256,),
            "norm.bias": (256,),
            "to_qkv.weight": (768, 256),
            "to_qkv.bias": (768,),
            "to_out.weight": (256, 256),
            "to_out.bias": (256,),
        }
        ffn = {
            "norm.weight": (256,),
            "norm.bias": (256,),
            "in_proj.weight": (2048, 256),
            "in_proj.bias": (2048,),
            "out_proj.weight": (256, 1024),
            "out_proj.bias": (256,),
        }
        expected = {
            "rotary_emb.omega": (32,),
            "read_in.weight": (256, 50),
            "read_in.bias": (256,),
        }
        for i in range(2):
            for j, block in enumerate([attn, ffn]):
                expected |= {f"t_blocks.{i}.{j}.{k}": s for k, s in block.items()}
        for i in range(4):
            for j, block in enumerate([attn, ffn, attn, ffn]):
                expected |= {f"st_blocks.{i}.{j}.{k}": s for k, s in block.items()}

        model = NuCLR(
            ctx_duration=20.0,
            latent_step=1.0,
            bin_size=0.02,
            dim=256,
            self_heads=4,
            lin_dropout=0.3,
            t_layers=2,
            st_layers=4,
            rot_ratio=0.5,
            rotate_value=True,
            attn_biases=True,
        )
        shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        assert shapes == expected
