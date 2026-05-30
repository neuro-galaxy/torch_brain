import pytest
import numpy as np
import torch

from torch_brain.data import collate, pad, pad8, chain, track_mask, track_batch


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


def test_pad8():
    # padding applied to np.ndarrays
    x = pad8(np.array([1, 2, 3]))
    y = pad8(np.array([4, 5]))

    batch = collate([x, y])
    assert torch.allclose(
        batch, torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0, 0]])
    )


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
