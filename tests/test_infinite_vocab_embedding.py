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
    emb.load_state_dict(torch.load("checkpoint.pth"))
    assert emb.is_lazy(), "Embedding should be lazy, no vocabulary set yet."

    # checkpointing a non-lazy embedding
    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.initialize_vocab(["word1", "word2", "word3"])

    # checkpoint
    torch.save(emb.state_dict(), "checkpoint.pth")
    del emb
    # load checkpoint
    state_dict = torch.load("checkpoint.pth")
    assert "weight" in state_dict, "Checkpoint should contain weight matrix."
    assert "vocab" in state_dict, "Checkpoint should contain vocabulary."

    emb = InfiniteVocabEmbedding(embedding_dim=128)
    emb.load_state_dict(torch.load("checkpoint.pth"))

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
    emb.load_state_dict(torch.load("checkpoint.pth"))

    assert emb.vocab == {
        "NA": 0,
        "word1": 1,
        "word2": 2,
        "word3": 3,
    }

    ### UPDATE: The below test is no longer valid, since words are always sorted.
    # load checkpoint after vocab is initialized but the order of the words is different
    # emb = InfiniteVocabEmbedding(embedding_dim=128)
    # emb.initialize_vocab(["word3", "word1", "word2"])
    # state_dict = torch.load("checkpoint.pth")
    # emb.load_state_dict(state_dict)

    # assert emb.vocab == {
    #     "NA": 0,
    #     "word3": 1,
    #     "word1": 2,
    #     "word2": 3,
    # }

    # assert torch.allclose(emb.weight, state_dict["weight"][[0, 3, 1, 2]])
    ###
