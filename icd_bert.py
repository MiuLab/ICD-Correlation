# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""


import collections
import logging
import os

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers import BertModel, BertForMaskedLM
from transformers.modeling_bert import (
    BertOnlyMLMHead,
    BertLayerNorm,
    BertPooler,
    BertEncoder
)    


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class ICDBertTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a ICDBertTokenizer.

    Args:
        vocab_file: Path to a one-icdcode-per-line vocabulary file
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        **kwargs
    ):
        """Constructs a ICDBertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
        """
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
        
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        if unk_token not in self.vocab:
            self.vocab[unk_token] = len(self.vocab)
            self.vocab[pad_token] = len(self.vocab)
            self.vocab[mask_token] = len(self.vocab)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        split_tokens = whitespace_tokenize(text)
        return split_tokens

    def tokenize(self, text):
        return self._tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids):
        """
        Build model inputs from a sequence
        by concatenating and adding special tokens.
        """
        return token_ids

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        return list(map(lambda x: 1 if x == self.pad_token_id else 0, token_ids_0))

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        return len(token_ids_0) * [0]

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class ICDBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ICDBertModel(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = ICDBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()


class ICDBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = ICDBertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path="train", block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        _, model_name = os.path.split(args.model_name_or_path)
        cached_features_file = os.path.join(
            directory, model_name + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                texts = f.readlines()

            for text in texts:
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                pad_size = args.block_size - len(tokenized_text)
                attention_mask = [1] * len(tokenized_text) + [0] * pad_size
                tokenized_text = tokenized_text + [tokenizer.pad_token_id] * pad_size
                self.examples.append((
                    tokenizer.build_inputs_with_special_tokens(tokenized_text),
                    attention_mask
                ))

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        input_ids, attention_mask = self.examples[item]
        return torch.tensor(input_ids), torch.tensor(attention_mask, dtype=torch.float)
