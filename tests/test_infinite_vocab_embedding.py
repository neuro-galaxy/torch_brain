import copy
import pytest
import torch

from torch_brain.nn import InfiniteVocabEmbedding


def test_embedding():
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    assert emb.is_lazy(), "Embedding should be lazy, no vocabulary set yet."

    # initialize vocabulary
    emb.initialize_vocab(["word1", "word2", "word3"])
    assert not emb.is_lazy(), "Embedding should not be lazy, vocabulary set."
    assert emb.weight.shape == (4, 128), "Weight matrix should be initialized."

    assert emb.vocab == {
        "NA": 0,
        "word1": 1,
        "word2": 2,
        "word3": 3,
    }, "Vocabulary should be set."

    # tokenization
    assert emb.tokenizer("word1") == 1
    assert emb.tokenizer(["word2", "word2", "word1"]) == [2, 2, 1]

    # reverse tokenization
    assert emb.detokenizer(1) == "word1"

    # subset vocabulary
    subset_emb = emb.subset_vocab(["word1", "word3"], inplace=False)

    print(emb.vocab)
    print(subset_emb.vocab)

    assert subset_emb.weight.shape == (3, 128)
    assert subset_emb.vocab == {
        "NA": 0,
        "word1": 1,
        "word3": 2,
    }, "Vocabulary should be subsetted."
    assert torch.allclose(subset_emb.weight, emb.weight[[0, 1, 3]])

    # extend vocabulary
    extended_emb = copy.deepcopy(emb)
    extended_emb.extend_vocab(["word4", "word5"])

    assert extended_emb.weight.shape == (6, 128)
    assert extended_emb.vocab == {
        "NA": 0,
        "word1": 1,
        "word2": 2,
        "word3": 3,
        "word4": 4,
        "word5": 5,
    }, "Vocabulary should be extended."
    assert torch.allclose(extended_emb.weight[:4], emb.weight)


def test_duplicate_vocab_entries():
    emb = InfiniteVocabEmbedding(embedding_dim=128)

    # Test duplicate entries in initialize_vocab
    with pytest.raises(ValueError, match="Vocabulary contains duplicate words"):
        emb.initialize_vocab(["word1", "word2", "word1", "word3"])

    # Initialize with valid vocab
    emb.initialize_vocab(["word1", "word2", "word3"])

    # Test duplicate entries in extend_vocab
    with pytest.raises(ValueError, match="Vocabulary already contains"):
        emb.extend_vocab(["word4", "word2", "word5"])

    # Test duplicate entries in subset_vocab
    with pytest.raises(ValueError, match="Vocabulary contains duplicate words"):
        emb.subset_vocab(["word1", "word2", "word1"])


def test_checkpointing():
    # checkpointing a lazy embedding
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    torch.save(emb.state_dict(), "checkpoint.pth")
    del emb
    # load checkpoint
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.load_state_dict(torch.load("checkpoint.pth", weights_only=False))
    assert emb.is_lazy(), "Embedding should be lazy, no vocabulary set yet."

    # checkpointing a non-lazy embedding
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.initialize_vocab(["word1", "word2", "word3"])

    # checkpoint
    torch.save(emb.state_dict(), "checkpoint.pth")
    del emb
    # load checkpoint
    state_dict = torch.load("checkpoint.pth", weights_only=False)
    assert "weight" in state_dict, "Checkpoint should contain weight matrix."
    assert "vocab" in state_dict, "Checkpoint should contain vocabulary."

    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.load_state_dict(torch.load("checkpoint.pth", weights_only=False))

    assert emb.vocab == {
        "NA": 0,
        "word1": 1,
        "word2": 2,
        "word3": 3,
    }
    del emb

    # load checkpoint after vocab is initialized
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.initialize_vocab(["word1", "word2", "word3"])
    emb.load_state_dict(torch.load("checkpoint.pth", weights_only=False))

    assert emb.vocab == {
        "NA": 0,
        "word1": 1,
        "word2": 2,
        "word3": 3,
    }

    # Test extend_vocab with exist_ok=True
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.initialize_vocab(["word1", "word2", "word3"])
    original_weights = emb.weight.clone()

    # Try extending with mix of new and existing words
    emb.extend_vocab(["word2", "word4", "word1", "word5"], exist_ok=True)

    # Check vocab was extended correctly
    assert emb.vocab == {
        "NA": 0,
        "word1": 1,
        "word2": 2,
        "word3": 3,
        "word4": 4,
        "word5": 5,
    }

    # Check original embeddings were preserved
    assert torch.allclose(emb.weight[:4], original_weights)

    # Test extend_vocab with exist_ok=False raises error
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.initialize_vocab(["word1", "word2", "word3"])

    with pytest.raises(ValueError):
        emb.extend_vocab(["word2", "word4"])

    # Test subset_vocab with inplace=True
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.initialize_vocab(["word1", "word2", "word3", "word4"])
    original_weights = emb.weight.clone()

    emb.subset_vocab(["word2", "word4"], inplace=True)

    assert emb.vocab == {
        "NA": 0,
        "word2": 1,
        "word4": 2,
    }
    # Check embeddings were preserved for kept words
    assert torch.allclose(emb.weight[1], original_weights[2])  # word2
    assert torch.allclose(emb.weight[2], original_weights[4])  # word4

    # Test subset_vocab with inplace=False
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.initialize_vocab(["word1", "word2", "word3", "word4"])
    original_weights = emb.weight.clone()

    new_emb = emb.subset_vocab(["word2", "word4"], inplace=False)

    # Original embedding should be unchanged
    assert emb.vocab == {
        "NA": 0,
        "word1": 1,
        "word2": 2,
        "word3": 3,
        "word4": 4,
    }
    assert torch.allclose(emb.weight, original_weights)

    # New embedding should have subset
    assert new_emb.vocab == {
        "NA": 0,
        "word2": 1,
        "word4": 2,
    }
    assert torch.allclose(new_emb.weight[1], original_weights[2])  # word2
    assert torch.allclose(new_emb.weight[2], original_weights[4])  # word4

    # Test subset_vocab with invalid words
    with pytest.raises(ValueError):
        emb.subset_vocab(["word2", "nonexistent"])

    # Test subset_vocab with duplicate words
    with pytest.raises(ValueError):
        emb.subset_vocab(["word2", "word2"])

    # Test subset_vocab with empty list
    with pytest.raises(AssertionError):
        emb.subset_vocab([])


def test_vocab_ordering():
    """Test that vocabulary ordering behavior works consistently across all operations"""

    # Test initial vocab creation maintains order
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.initialize_vocab(["word3", "word1", "word2"])
    assert list(emb.vocab.keys()) == ["NA", "word3", "word1", "word2"]

    # Test extend_vocab maintains existing order and appends new words
    emb.extend_vocab(["word5", "word4"])
    assert list(emb.vocab.keys()) == ["NA", "word3", "word1", "word2", "word5", "word4"]

    # Test subset_vocab maintains relative order of selected words
    subset_emb = emb.subset_vocab(["word2", "word5", "word1"], inplace=False)
    assert list(subset_emb.vocab.keys()) == ["NA", "word2", "word5", "word1"]


# TODO: fix InfiniteVocabEmbedding.load_state_dict()
# The below test is currently failing.
# def test_state_dict_loading():
#     # Test state dict loading preserves embeddings while allowing different order
#     emb1 = InfiniteVocabEmbedding(embedding_dim=128)
#     emb1.initialize_vocab(["word1", "word2", "word3"])
#     original_weights = emb1.weight.clone()

#     emb2 = InfiniteVocabEmbedding(embedding_dim=128)
#     emb2.initialize_vocab(["word3", "word1", "word2"])

#     # Load state dict and verify embeddings are correctly remapped
#     emb2.load_state_dict(emb1.state_dict())
#     # Need to use tokenizer() since vocab dict order may be different
#     assert torch.allclose(emb2.weight[emb2.tokenizer("word1")],
#                          original_weights[emb1.tokenizer("word1")])
#     assert torch.allclose(emb2.weight[emb2.tokenizer("word2")],
#                          original_weights[emb1.tokenizer("word2")])
#     assert torch.allclose(emb2.weight[emb2.tokenizer("word3")],
#                          original_weights[emb1.tokenizer("word3")])
