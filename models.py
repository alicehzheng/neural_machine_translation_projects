import os
import sys
import csv
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence
# from dropout import WeightDrop, LockedDrop
import numpy as np

import math
import pickle
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
#from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size, dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.lstm4 = nn.LSTM(hidden_size, out_size, bidirectional=True)

        self.embed_drop = nn.Dropout(dropout_rate)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(out_size, out_size)
        self.fc2 = nn.Linear(out_size, out_size)
        self.act = nn.SELU(True)

    def forward(self, x):
        seq_len, batch_size = x.shape[0:2]
        embed = self.embedding(x.long())
        embed = self.embed_drop(embed)

        output_lstm, (hidden1, cell1) = self.lstm1(embed)  # L x N x H
        output_lstm = self.dropout1(output_lstm)
        output_lstm, (hidden2, cell2) = self.lstm2(output_lstm)  # L x N x H
        output_lstm = self.dropout2(output_lstm)
        output_lstm, (hidden3, cell3) = self.lstm3(output_lstm)  # L x N x H
        output_lstm = self.dropout3(output_lstm)
        output_lstm, (hidden4, cell4) = self.lstm4(output_lstm)  # L x N x H

        key = self.act(self.fc1(output_lstm)).transpose(0, 1)
        value = self.act(self.fc2(output_lstm)).transpose(0, 1)
        hidden = torch.cat([hidden4[0, :, :], hidden4[1, :, :]], dim=1)
        cell = torch.cat([cell4[0, :, :], cell4[1, :, :]], dim=1)
        print(key.shape)
        print(value.shape)
        print(hidden.shape)
        return output_lstm, key, value, hidden, cell


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(512, 512)
        self.E_softmax = nn.Softmax(dim=2)

    def forward(self, hidden2, key, value, seq_lens):
        # key: seq//8, batch, base*2 --> batch base*2, seq//8
        # hidden2: batch, base*2     --> batch 1 base*2

        # key: seq//8, batch, base*2 --> batch base*2, seq//8
        # hidden2: batch, base*2     --> batch 1 base*2
        # batch 1 seq//8
        # batch seq//8 1
        # value: seq//8, batch, base*2 --> batch base*2, seq//8
        # batch base*2, 1


        # key: batch * seq * 512
        # value: batch * seq * 512
        query = self.query_layer(hidden2)

        batch, slen, dim = key.shape[:]

        query = query.reshape(batch, dim, 1)

        E = torch.bmm(key, query).reshape(batch, 1, slen)  # transpose(1,2)

        for i in range(0, len(seq_lens)):
            E[i, 0, seq_lens[i]:] = float("-infinity")

        # print(E)
        E = self.E_softmax(E)

        context = torch.bmm(E, value).reshape(batch, dim)

        return context, E.cpu().squeeze(2).data.numpy()


class Decoder(nn.Module):
    def __init__(self, out_dim, lstm_dim):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(out_dim, lstm_dim)
        self.lstm1 = nn.LSTMCell(lstm_dim * 2, lstm_dim)
        self.lstm2 = nn.LSTMCell(lstm_dim, lstm_dim)
        self.drop = nn.Dropout(0.05)
        self.fc = nn.Linear(lstm_dim, out_dim)
        self.fc.weight = self.embed.weight

    def forward(self, x, context, hidden1, cell1, hidden2, cell2):
        x = self.embed(x)
        ##print(context.shape)
        x = torch.cat([x, context], dim=1)

        hidden1, cell1 = self.lstm1(x, (hidden1, cell1))
        hidden2, cell2 = self.lstm2(hidden1, (hidden2, cell2))
        x = self.drop(hidden2)
        x = self.fc(x)
        return x, hidden1, cell1, hidden2, cell2


class NMT(object):
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.encoder = Encoder()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab



        # initialize neural network layers...

    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        src_encodings, decoder_init_state = self.encode(src_sents, tgt_sents)
        scores = self.decode(src_encodings, decoder_init_state, tgt_sents)

        return scores

    def encode(self, src_sents: List[List[str]]) -> Tuple[Tensor, Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        return src_encodings, decoder_init_state

    def decode(self, src_encodings: Tensor, decoder_init_state: Any, tgt_sents: List[List[str]]) -> Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """

        return scores

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        return hypotheses

    def evaluate_ppl(self, dev_data: List[Any], batch_size: int = 32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

