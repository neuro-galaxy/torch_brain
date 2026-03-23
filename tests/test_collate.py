import numpy as np
import pytest
import torch

from torch_brain.data import (
    chain,
    collate,
    pad,
    pad8,
    pad2d,
    pad2d8,
    track_batch,
    track_mask,
    track_mask8,
    track_mask2d,
    track_mask2d8,
)


def test_pad():
    # padding applied to np.ndarrays
    x = pad(np.array([1, 2, 3]))
    y = pad(np.array([4, 5]))

    batch = collate([x, y])
    assert torch.allclose(batch, torch.tensor([[1, 2, 3], [4, 5, 0]]))

    # padding applied to torch.Tensors
    x = pad(torch.tensor([[1], [2], [3]]))
    y = pad(torch.tensor([[4], [5]]))

    batch = collate([x, y])
    assert torch.allclose(batch, torch.tensor([[[1], [2], [3]], [[4], [5], [0]]]))

    # paddding applied to other objects (lists, maps, etc.)
    x = [pad({"a": np.array([1, 2, 3]), "b": np.array([11, 12, 13])}), np.array([0, 1])]
    y = [pad({"a": np.array([4, 5]), "b": np.array([14, 15])}), np.array([2, 3])]

    batch = collate([x, y])
    assert torch.allclose(batch[0]["a"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert torch.allclose(batch[0]["b"], torch.tensor([[11, 12, 13], [14, 15, 0]]))
    assert torch.allclose(batch[1], torch.tensor([[0, 1], [2, 3]]))


def test_track_mask():
    # padding applied to np.ndarrays
    x_mask = track_mask(np.array([1, 2, 3]))
    y_mask = track_mask(np.array([4, 5]))

    batch = collate([x_mask, y_mask])
    assert batch.ndim == 2
    assert batch.dtype == torch.bool
    assert torch.allclose(batch, torch.BoolTensor([[1, 1, 1], [1, 1, 0]]))

    # padding applied to torch.Tensors
    x_mask = track_mask(torch.Tensor([[1], [2], [3]]))
    y_mask = track_mask(torch.Tensor([[4], [5]]))

    batch = collate([x_mask, y_mask])
    assert batch.ndim == 2
    assert batch.dtype == torch.bool
    assert torch.allclose(batch, torch.BoolTensor([[1, 1, 1], [1, 1, 0]]))

    # paddding applied to other objects (lists, maps, etc.)
    x = [
        pad({"a": np.array([1, 2, 3]), "b": np.array([11, 12, 13])}),
        track_mask(np.array([1, 2, 3])),
        np.array([0, 1]),
    ]
    y = [
        pad({"a": np.array([4, 5]), "b": np.array([14, 15])}),
        track_mask(np.array([4, 5])),
        np.array([2, 3]),
    ]

    batch = collate([x, y])
    assert batch[1].ndim == 2
    assert batch[1].dtype == torch.bool

    assert torch.allclose(batch[0]["a"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert torch.allclose(batch[0]["b"], torch.tensor([[11, 12, 13], [14, 15, 0]]))
    assert torch.allclose(batch[1], torch.BoolTensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.allclose(batch[2], torch.tensor([[0, 1], [2, 3]]))


def test_pad8():
    # padding applied to np.ndarrays
    x = pad8(np.array([1, 2, 3]))
    y = pad8(np.array([4, 5]))

    batch = collate([x, y])
    assert torch.allclose(
        batch, torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0, 0]])
    )


def test_track_mask8():
    # mask with pad8 rounding
    x_mask = track_mask8(np.array([1, 2, 3]))
    y_mask = track_mask8(np.array([4, 5]))

    batch = collate([x_mask, y_mask])
    # max len is 3, rounded up to 8
    assert batch.shape == (2, 8)
    assert batch.dtype == torch.bool
    assert torch.allclose(
        batch,
        torch.BoolTensor(
            [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]
        ),
    )

    # torch.Tensors
    x_mask = track_mask8(torch.Tensor([[1], [2], [3]]))
    y_mask = track_mask8(torch.Tensor([[4], [5]]))

    batch = collate([x_mask, y_mask])
    assert batch.shape == (2, 8)
    assert batch.dtype == torch.bool
    assert torch.allclose(
        batch,
        torch.BoolTensor(
            [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]
        ),
    )

    # length already a multiple of 8
    x_mask = track_mask8(np.zeros(8))
    y_mask = track_mask8(np.zeros(8))
    batch = collate([x_mask, y_mask])
    assert batch.shape == (2, 8)
    assert batch.all()

    # length 9 -> rounds to 16
    x_mask = track_mask8(np.zeros(9))
    y_mask = track_mask8(np.zeros(7))
    batch = collate([x_mask, y_mask])
    assert batch.shape == (2, 16)


def test_pad2d():
    # padding applied to np.ndarrays
    x = pad2d(np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]]))
    y = pad2d(np.array([[14, 15], [24, 25]]))

    batch = collate([x, y])
    batch_expected = torch.tensor(
        [
            [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            [[14, 15, 0], [24, 25, 0], [0, 0, 0]],
        ]
    )
    assert torch.allclose(batch, batch_expected)

    # padding applied to torch.Tensors
    x = pad2d(
        torch.tensor([[[11], [12], [13]], [[21], [22], [23]], [[31], [32], [33]]])
    )
    y = pad2d(torch.tensor([[[14], [15]], [[24], [25]]]))

    batch = collate([x, y])
    batch_expected = torch.tensor(
        [
            [[[11], [12], [13]], [[21], [22], [23]], [[31], [32], [33]]],
            [[[14], [15], [0]], [[24], [25], [0]], [[0], [0], [0]]],
        ]
    )
    assert torch.allclose(batch, batch_expected)

    # padding applied to bool
    x = pad2d(np.ones((3, 3), dtype=bool))
    y = pad2d(np.ones((2, 2), dtype=bool))
    batch = collate([x, y])
    batch_expected = torch.BoolTensor(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        ]
    )
    assert torch.allclose(batch, batch_expected)


def test_track_mask2d():
    # basic 2d mask tracking
    x_mask = track_mask2d(torch.ones(3, 4))
    y_mask = track_mask2d(torch.ones(2, 3))

    batch = collate([x_mask, y_mask])
    assert batch.shape == (2, 3, 4)
    assert batch.dtype == torch.bool
    # first sample fully True
    assert batch[0].all()
    # second sample: 2x3 True, rest False
    assert batch[1, :2, :3].all()
    assert not batch[1, :2, 3:].any()
    assert not batch[1, 2, :].any()

    # rejects non-2d input
    with pytest.raises(ValueError, match="2 dimensions"):
        track_mask2d(torch.ones(5))

    with pytest.raises(ValueError, match="2 dimensions"):
        track_mask2d(torch.ones(2, 3, 4))


def test_pad2d8():
    # padding applied to np.ndarrays, inner dim rounded up to multiple of 8
    x = pad2d8(np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]]))
    y = pad2d8(np.array([[14, 15], [24, 25]]))

    batch = collate([x, y])
    # inner dim max is 3, rounded up to 8
    assert batch.shape == (2, 3, 8)
    assert torch.allclose(
        batch,
        torch.tensor(
            [
                [
                    [11, 12, 13, 0, 0, 0, 0, 0],
                    [21, 22, 23, 0, 0, 0, 0, 0],
                    [31, 32, 33, 0, 0, 0, 0, 0],
                ],
                [
                    [14, 15, 0, 0, 0, 0, 0, 0],
                    [24, 25, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        ),
    )

    # padding applied to torch.Tensors with extra trailing dimensions
    x = pad2d8(
        torch.tensor([[[11], [12], [13]], [[21], [22], [23]], [[31], [32], [33]]])
    )
    y = pad2d8(torch.tensor([[[14], [15]], [[24], [25]]]))

    batch = collate([x, y])
    assert batch.shape == (2, 3, 8, 1)
    # first 3 cols populated, rest zero-padded to 8
    assert torch.allclose(
        batch[:, :, :3, :],
        torch.tensor(
            [
                [[[11], [12], [13]], [[21], [22], [23]], [[31], [32], [33]]],
                [[[14], [15], [0]], [[24], [25], [0]], [[0], [0], [0]]],
            ]
        ),
    )
    assert torch.allclose(
        batch[:, :, 3:, :], torch.zeros(2, 3, 5, 1, dtype=torch.long)
    )

    # inner dim already a multiple of 8 — no extra padding
    x = pad2d8(np.zeros((2, 8)))
    y = pad2d8(np.zeros((3, 8)))
    batch = collate([x, y])
    assert batch.shape == (2, 3, 8)

    # inner dim 9 -> rounds to 16
    x = pad2d8(np.zeros((2, 9)))
    y = pad2d8(np.zeros((3, 7)))
    batch = collate([x, y])
    assert batch.shape == (2, 3, 16)

    # padding applied to bool
    x = pad2d8(np.ones((3, 3), dtype=bool))
    y = pad2d8(np.ones((2, 2), dtype=bool))
    batch = collate([x, y])
    assert batch.shape == (2, 3, 8)
    assert batch.dtype == torch.bool
    # first sample: 3x3 ones, padded inner to 8
    assert batch[0, :, :3].all()
    assert not batch[0, :, 3:].any()
    # second sample: 2x2 ones, rows 0-1 cols 0-1
    assert batch[1, :2, :2].all()
    assert not batch[1, :2, 2:].any()
    assert not batch[1, 2, :].any()


def test_track_mask2d8():
    # basic 2d8 mask tracking — inner dim rounded to multiple of 8
    x_mask = track_mask2d8(torch.ones(3, 5))
    y_mask = track_mask2d8(torch.ones(2, 3))

    batch = collate([x_mask, y_mask])
    # inner dim max is 5, rounded up to 8
    assert batch.shape == (2, 3, 8)
    assert batch.dtype == torch.bool
    # first sample: 3x5 ones, cols 5-7 False
    assert batch[0, :, :5].all()
    assert not batch[0, :, 5:].any()
    # second sample: 2x3 ones
    assert batch[1, :2, :3].all()
    assert not batch[1, :2, 3:].any()
    assert not batch[1, 2, :].any()

    # inner dim already multiple of 8
    x_mask = track_mask2d8(torch.ones(2, 8))
    y_mask = track_mask2d8(torch.ones(3, 8))
    batch = collate([x_mask, y_mask])
    assert batch.shape == (2, 3, 8)

    # inner dim 9 -> rounds to 16
    x_mask = track_mask2d8(torch.ones(2, 9))
    y_mask = track_mask2d8(torch.ones(3, 7))
    batch = collate([x_mask, y_mask])
    assert batch.shape == (2, 3, 16)

    # rejects non-2d input
    with pytest.raises(ValueError, match="2 dimensions"):
        track_mask2d8(torch.ones(5))

    with pytest.raises(ValueError, match="2 dimensions"):
        track_mask2d8(torch.ones(2, 3, 4))


def test_chain():
    # chaining applied to np.ndarrays
    x = chain(np.array([1, 2, 3]))
    y = chain(np.array([4, 5]))

    batch = collate([x, y])
    assert torch.allclose(batch, torch.tensor([1, 2, 3, 4, 5]))

    # chaining applied to torch.Tensors
    x = chain(torch.tensor([[1], [2], [3]]))
    y = chain(torch.tensor([[4], [5]]))

    batch = collate([x, y])
    assert torch.allclose(batch, torch.tensor([[1], [2], [3], [4], [5]]))

    # chaining applied to other objects (lists, maps, etc.)
    x = [
        chain({"a": np.array([1, 2, 3]), "b": np.array([11, 12, 13])}),
        np.array([0, 1]),
    ]
    y = [chain({"a": np.array([4, 5]), "b": np.array([14, 15])}), np.array([2, 3])]
    batch = collate([x, y])

    assert torch.allclose(batch[0]["a"], torch.tensor([1, 2, 3, 4, 5]))
    assert torch.allclose(batch[0]["b"], torch.tensor([11, 12, 13, 14, 15]))
    assert torch.allclose(batch[1], torch.tensor([[0, 1], [2, 3]]))


def test_track_batch():
    # chaining applied to np.ndarrays
    x = track_batch(np.array([1, 2, 3]))
    y = track_batch(np.array([4, 5]))

    batch = collate([x, y])
    assert batch.ndim == 1
    assert batch.dtype == torch.int64
    assert torch.allclose(batch, torch.tensor([0, 0, 0, 1, 1]))

    # chaining applied to torch.Tensors
    x = track_batch(torch.tensor([[1], [2], [3]]))
    y = track_batch(torch.tensor([[4], [5]]))

    batch = collate([x, y])
    assert batch.ndim == 1
    assert batch.dtype == torch.int64
    assert torch.allclose(batch, torch.tensor([0, 0, 0, 1, 1]))

    # chaining applied to other objects (lists, maps, etc.)
    x = [
        chain({"a": np.array([1, 2, 3]), "b": np.array([11, 12, 13])}),
        track_batch(np.array([1, 2, 3])),
        np.array([0, 1]),
    ]
    y = [
        chain({"a": np.array([4, 5]), "b": np.array([14, 15])}),
        track_batch(np.array([4, 5])),
        np.array([2, 3]),
    ]
    batch = collate([x, y])

    assert torch.allclose(batch[0]["a"], torch.tensor([1, 2, 3, 4, 5]))
    assert torch.allclose(batch[0]["b"], torch.tensor([11, 12, 13, 14, 15]))
    assert torch.allclose(batch[1], torch.tensor([0, 0, 0, 1, 1]))
    assert torch.allclose(batch[2], torch.tensor([[0, 1], [2, 3]]))


def test_collate():
    # first sample
    a1 = np.array([1, 2, 3])
    b1 = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
    c1 = np.array(
        [[[101, 102], [103, 104]], [[105, 106], [107, 108]], [[109, 110], [111, 112]]]
    )
    d1 = torch.tensor(1.0)
    e1 = torch.tensor([[1001, 1002], [1003, 1004]])
    data1 = dict(
        a=pad(a1),
        b=pad(b1),
        c=chain(c1),
        d=d1,
        e=e1,
        mask=track_mask(a1),
        batch=track_batch(c1),
    )

    # second sample
    a2 = np.array([4, 5])
    b2 = np.array([[20, 21, 22], [23, 24, 25]])
    c2 = np.array([[[113, 114], [115, 116]], [[117, 118], [119, 120]]])
    d2 = torch.tensor(2.0)
    e2 = torch.tensor([[1005, 1006], [1007, 1008]])
    data2 = dict(
        a=pad(a2),
        b=pad(b2),
        c=chain(c2),
        d=d2,
        e=e2,
        mask=track_mask(a2),
        batch=track_batch(c2),
    )

    # collate
    batch = collate([data1, data2])

    # check
    assert torch.allclose(batch["a"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert torch.allclose(
        batch["b"],
        torch.tensor(
            [
                [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                [[20, 21, 22], [23, 24, 25], [0, 0, 0]],
            ]
        ),
    )
    assert torch.allclose(
        batch["c"],
        torch.tensor(
            [
                [[101, 102], [103, 104]],
                [[105, 106], [107, 108]],
                [[109, 110], [111, 112]],
                [[113, 114], [115, 116]],
                [[117, 118], [119, 120]],
            ]
        ),
    )
    assert torch.allclose(batch["d"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(
        batch["e"],
        torch.tensor([[[1001, 1002], [1003, 1004]], [[1005, 1006], [1007, 1008]]]),
    )
    assert batch["mask"].ndim == 2
    assert batch["mask"].dtype == torch.bool
    assert torch.allclose(batch["mask"], torch.BoolTensor([[1, 1, 1], [1, 1, 0]]))
    assert batch["batch"].ndim == 1
    assert batch["batch"].dtype == torch.int64
    assert torch.allclose(batch["batch"], torch.tensor([0, 0, 0, 1, 1]))


def test_chain_with_missing_keys():
    # chaining applied to np.ndarrays
    x = chain({"a": np.array([1, 2, 3])}, allow_missing_keys=True)
    y = chain({"a": np.array([4, 5]), "b": np.array([14, 15])}, allow_missing_keys=True)

    batch = collate([x, y])
    assert torch.allclose(batch["a"], torch.tensor([1, 2, 3, 4, 5]))
    assert torch.allclose(batch["b"], torch.tensor([14, 15]))

    # chaining should fail if keys are missing
    y = chain({"a": np.array([1, 2, 3])})
    x = chain({"a": np.array([4, 5]), "b": np.array([14, 15])})

    with pytest.raises(KeyError):
        collate([x, y])

    # chaining with allow_missing_keys=True should only work on dicts
    with pytest.raises(TypeError):
        x = chain(np.array([1, 2, 3]), allow_missing_keys=True)
