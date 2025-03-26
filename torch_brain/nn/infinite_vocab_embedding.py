import warnings
from collections.abc import Iterable
from typing import List, Union
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter


class InfiniteVocabEmbedding(nn.Module):
    """Embedding layer with a vocabulary that can be extended. Vocabulary is saved along
    with the model, and is reloaded when the state_dict is loaded. This is useful when
    the vocabulary is dynamically generated, e.g. from a dataset. For this reason this
    class also plays the role of the tokenizer.

    This layer is initially lazy, i.e. it does not have a weight matrix. The weight
    matrix is initialized when:

    - The vocabulary is initialized via :meth:`initialize_vocab()`.
    - or The model is loaded from a checkpoint that contains the vocabulary.

    If the vocabulary is initialized before :meth:`load_state_dict` is called,
    an error will be raised if the vocabulary in the checkpoint does not match the
    vocabulary in the model. The order of the words in the vocabulary does not matter,
    as long as the words are the same.

    If you would like to create a new variant of an existing :obj:`InfiniteVocabEmbedding`
    (that you loaded from a checkpoint), you can use:

    - :meth:`extend_vocab()` to add new words to the vocabulary. The embeddings for the new
    words will be initialized randomly.

    - :meth:`subset_vocab()` to select a subset of the vocabulary. The embeddings for the
    selected words will be copied from the original embeddings, and the ids for the
    selected words will change and :meth:`tokenizer` will be updated accordingly.

    This module also plays the role of the tokenizer, which is accessible via
    :meth:`tokenizer`, and is a Callable.

    .. warning:: If you are only interested in loading a subset of words from a checkpoint, do not call :meth:`initialize_vocab()`, first load the checkpoint then use :meth:`subset_vocab`.

    Args:
        embedding_dim (int): Embedding dimension.
        init_scale (float): The standard deviation of the normal distribution used to
            initialize the embedding matrix. Default is 0.02.
    """

    def __init__(self, embedding_dim, init_scale=0.02):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.init_scale = init_scale
        self.padding_idx = 0

        self.weight = UninitializedParameter()
        self.vocab = None

        # Unfortunately, this hook is private, though there has been a PR to make it
        # public: https://github.com/pytorch/pytorch/issues/75287
        self._register_load_state_dict_pre_hook(
            self._hook_vocab_on_load_state_dict, with_module=False
        )

    def initialize_vocab(self, vocab: List[str]):
        r"""Initialize the vocabulary with a list of words. This method should be called
        only once, and before the model is trained. If you would like to add new words
        to the vocabulary, use :obj:`extend_vocab()` instead.

        .. note:: A special word "NA" will always be in the vocabulary, and is assigned the index 0. 0 is used for padding.

        Args:
            vocab (List[str]): A list of words to initialize the vocabulary.

        Example ::

            >>> from torch_brain.nn import InfiniteVocabEmbedding

            >>> embedding = InfiniteVocabEmbedding(64)

            >>> vocab = ["apple", "banana", "cherry"]
            >>> embedding.initialize_vocab(vocab)

            >>> embedding.vocab
            OrderedDict([('NA', 0), ('apple', 1), ('banana', 2), ('cherry', 3)])

            >>> embedding.weight.shape
            torch.Size([4, 64])
        """
        assert (
            self.vocab is None
        ), f"Vocabulary already initialized, and has {len(self.vocab)} words. "
        "If you want to add new words to the vocabulary, use extend_vocab() instead."

        # Create a mapping from words to indices
        if isinstance(vocab, str):
            raise ValueError("vocab cannot be a single string")
        elif isinstance(vocab, Iterable):
            # check that all words are unique
            if len(vocab) != len(set(vocab)):
                raise ValueError("Vocabulary contains duplicate words")
            self.vocab = OrderedDict(zip(vocab, range(1, len(vocab) + 1)))
            assert "NA" not in self.vocab, "NA is a reserved word"
            self.vocab["NA"] = 0
            self.vocab.move_to_end("NA", last=False)
        else:
            raise ValueError("vocab must be a list or dict")

        self.initialize_parameters(len(self.vocab))

    def extend_vocab(self, vocab: List[str], exist_ok=False):
        r"""Extend the vocabulary with a list of words. If a word already exists in the
        vocabulary, an error will be raised. The embeddings for the new words will be
        initialized randomly, and new ids will be assigned to the new words.

        Args:
            vocab (List[str]): A list of words to add to the vocabulary.
            exist_ok (bool): If True, the method will not raise an error if the new words
                already exist in the vocabulary and will skip them. Default is False.

        Example ::

            >>> from torch_brain.nn import InfiniteVocabEmbedding

            >>> embedding = InfiniteVocabEmbedding(64)

            >>> vocab = ["apple", "banana", "cherry"]
            >>> embedding.initialize_vocab(vocab)
            >>> embedding
            InfiniteVocabEmbedding(embedding_dim=64, num_embeddings=4)

            >>> new_words = ["date", "elderberry", "fig"]
            >>> embedding.extend_vocab(new_words)
            InfiniteVocabEmbedding(embedding_dim=64, num_embeddings=7)

            >>> embedding.vocab
            OrderedDict([('NA', 0), ('apple', 1), ('banana', 2), ('cherry', 3), ('date', 4), ('elderberry', 5), ('fig', 6)])

            >>> embedding.weight.shape
            torch.Size([7, 64])
        """
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")

        # find intersection and difference between key sets
        new_words, existing_words = [], []
        for word in vocab:
            if word not in self.vocab:
                new_words.append(word)
            else:
                existing_words.append(word)

        if not exist_ok and len(existing_words) > 0:
            raise ValueError(
                f"Vocabulary already contains {len(existing_words)} out of {len(vocab)}"
                f" words that are being added. You can skip this error by setting "
                f"exist_ok=True, but be aware that the embeddings for these existing "
                f"words won't be re-initialized."
            )

        # update tokenizer
        self.vocab.update(
            OrderedDict(
                zip(new_words, range(len(self.vocab), len(self.vocab) + len(new_words)))
            )
        )

        # check that the dictionary update was done correctly
        assert len(self.vocab) == len(self.weight) + len(new_words), (
            f"Expected vocabulary length to be {len(self.weight) + len(new_words)}, "
            f"but got {len(self.vocab)}"
        )

        # check that the largest value in the vocab matches its size
        assert max(self.vocab.values()) == len(self.vocab) - 1, (
            f"Expected maximum vocabulary index to be {len(self.vocab) - 1}, "
            f"but got {max(self.vocab.values())}"
        )

        # make a copy of existing embeddings
        embeddings_for_existing_words = self.weight.clone().detach()

        # reinitalize weight matrix after extending it
        self.weight = UninitializedParameter()
        self.initialize_parameters(len(self.vocab))

        # copy existing embeddings into new weight matrix
        self.weight.data[: len(embeddings_for_existing_words)] = (
            embeddings_for_existing_words
        )
        return self

    def subset_vocab(self, vocab: List[str], inplace=True):
        r"""Select a subset of the vocabulary. The embeddings for the selected words
        will be copied from the original embeddings, and the ids for the selected words
        will be updated accordingly.

        An error will be raised if one of the words does not exist in the vocabulary.

        Args:
            vocab (List[str]): A list of words to select from the vocabulary.
            inplace (bool): If True, the method will modify the vocabulary and the weight
                matrix in place. If False, a new InfiniteVocabEmbedding will be returned
                with the selected words. Default is True.

        Example ::

            >>> from torch_brain.nn import InfiniteVocabEmbedding

            >>> embedding = InfiniteVocabEmbedding(64)

            >>> vocab = ["apple", "banana", "cherry"]
            >>> embedding.initialize_vocab(vocab)
            >>> embedding
            InfiniteVocabEmbedding(embedding_dim=64, num_embeddings=4)

            >>> selected_words = ["banana", "cherry"]
            >>> embedding.subset_vocab(selected_words)
            InfiniteVocabEmbedding(embedding_dim=64, num_embeddings=3)

            >>> embedding.vocab
            OrderedDict([('NA', 0), ('banana', 1), ('cherry', 2)])

            >>> embedding.weight.shape
            torch.Size([3, 64])
        """
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")

        assert len(vocab) > 0, "Vocabulary must contain at least one word."

        # check that all words are unique
        if len(vocab) != len(set(vocab)):
            raise ValueError("Vocabulary contains duplicate words")

        # find intersection and difference between key sets
        word_indices = [0]  # NA will be added
        for word in vocab:
            if word not in self.vocab:
                raise ValueError(f"Vocabulary does not contain word {word}")
            else:
                word_indices.append(self.tokenizer(word))

        # make a copy of selected embeddings
        word_indices = torch.tensor(word_indices, dtype=torch.long)
        with torch.no_grad():
            embeddings_for_selected_words = self.weight[word_indices].clone().detach()

        if inplace:
            self.vocab = None
            self.weight = UninitializedParameter()

            self.initialize_vocab(vocab)

            self.weight.data = embeddings_for_selected_words
            return self
        else:
            new_embedding = self.__class__(self.embedding_dim, self.init_scale)
            new_embedding.initialize_vocab(vocab)
            new_embedding.weight.data = embeddings_for_selected_words
            return new_embedding

    def tokenizer(self, words: Union[str, List[str]]):
        r"""Convert a word or a list of words to their token indices.

        Args:
            words (Union[str, List[str]]): A word or a list of words.

        Returns:
            Union[int, List[int]]: A token index or a list of token indices.

        Example ::

            >>> from torch_brain.nn import InfiniteVocabEmbedding

            >>> embedding = InfiniteVocabEmbedding(64)

            >>> vocab = ["apple", "banana", "cherry"]
            >>> embedding.initialize_vocab(vocab)

            >>> embedding.tokenizer("banana")
            2

            >>> embedding.tokenizer(["apple", "cherry", "apple"])
            [1, 3, 1]
        """
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")

        if isinstance(words, str):
            return self.vocab[words]
        return [self.vocab[w] for w in words]

    def detokenizer(self, index: int):
        r"""Convert a token index to a word.

        Args:
            index (int): A token index.

        Returns:
            str: A word.

        Example ::

            >>> from torch_brain.nn import InfiniteVocabEmbedding

            >>> embedding = InfiniteVocabEmbedding(64)

            >>> vocab = ["apple", "banana", "cherry"]
            >>> embedding.initialize_vocab(vocab)

            >>> embedding.detokenizer(2)
            'banana'
        """
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")

        return list(self.vocab.keys())[index]

    def is_lazy(self):
        r"""Returns True if the module is not initialized.

        Example ::

            >>> from torch_brain.nn import InfiniteVocabEmbedding

            >>> embedding = InfiniteVocabEmbedding(64)

            >>> embedding.is_lazy()
            True

            >>> vocab = ["apple", "banana", "cherry"]
            >>> embedding.initialize_vocab(vocab)

            >>> embedding.is_lazy()
            False
        """
        return isinstance(self.weight, UninitializedParameter)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module, but will not reset the
        vocabulary."""
        if not self.is_lazy():
            torch.nn.init.normal_(self.weight, mean=0, std=self.init_scale)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)

    def initialize_parameters(self, num_embeddings) -> None:  # type: ignore[override]
        if self.is_lazy():
            with torch.no_grad():
                self.weight.materialize((num_embeddings, self.embedding_dim))
                self.reset_parameters()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.is_lazy():
            destination[prefix + "weight"] = self.weight
            destination[prefix + "vocab"] = self.vocab
        else:
            super()._save_to_state_dict(destination, prefix, keep_vars)
            destination[prefix + "vocab"] = self.vocab

    def _hook_vocab_on_load_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not self.is_lazy():
            # pop the vocabulary from the state_dict
            # popping is done because by default pytorch does not know how to load
            # vocab which is an OrderedDict, so we cannot keep it inside of the state_dict.
            # however, if we pop it, it will also be removed in the original state_dict
            # which can have unintended side effects, noteably this will alter the actual
            # state_dict in the checkpoint or the other module being used.
            # TODO: find a way to keep the vocab in the state_dict
            incoming_vocab = state_dict.pop(prefix + "vocab")

            # incoming_vocab and self.vocab might have the same keys but in a different order
            # reorder incoming_vocab to match self.vocab, and get the remapping indices
            remap_indices = []
            for word, index in self.vocab.items():
                if not word in incoming_vocab:
                    raise ValueError(
                        f"Vocabulary mismatch: word {word} is missing. If "
                        "you would like to add new words, or a new "
                        "vocabulary, do not initialize the vocab, load the"
                        " checkpoint, and then call extend_vocab() to add"
                        "new words, and/or subset_vocab() to remove words."
                    )
                remap_indices.append(incoming_vocab.pop(word))
            if len(incoming_vocab) > 0:
                raise ValueError(
                    f"Vocabulary mismatch: {len(incoming_vocab)} words are"
                    "remaining and cannot be loaded. If you would like to "
                    "load a subset of the vocabulary, do not initialize "
                    "the vocab, load the checkpoint, and then call "
                    "subset_vocab()"
                )
            remap_indices = torch.tensor(remap_indices, dtype=torch.long)

            state_dict[prefix + "weight"] = state_dict[prefix + "weight"][remap_indices]

        else:
            if not isinstance(state_dict[prefix + "weight"], UninitializedParameter):
                # The module is not initialized, but the one being loaded is
                with torch.no_grad():
                    self.weight.materialize(state_dict[prefix + "weight"].shape)
                self.vocab = state_dict.pop(prefix + "vocab")
            else:
                # Both the module and the one being loaded are not initialized
                assert state_dict.pop(prefix + "vocab") is None

    def forward(self, input):
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")
        return F.embedding(input, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        return "embedding_dim={}, num_embeddings={}".format(
            self.embedding_dim, len(self.vocab) if self.vocab is not None else 0
        )
