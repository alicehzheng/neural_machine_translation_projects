
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence

from collections import namedtuple

from typing import List, Tuple, Dict, Set, Union

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size, dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.embed_drop = nn.Dropout(dropout_rate)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_size * 2, out_size)
        self.fc2 = nn.Linear(hidden_size * 2, out_size)
        self.act = nn.SELU(True)

    def forward(self, x):
        x, seq_lens = pad_packed_sequence(x, batch_first=True)

        embed = self.embedding(x.long())
        embed = self.embed_drop(embed)
        embed = pack_padded_sequence(embed, seq_lens, batch_first = True)
        output_lstm, (hidden1, cell1) = self.lstm1(embed)  # N x L x H
        output_lstm, seq_lens = pad_packed_sequence(output_lstm, batch_first=True)
        output_lstm = self.dropout1(output_lstm)
        output_lstm = pack_padded_sequence(output_lstm, seq_lens, batch_first=True)
        output_lstm, (hidden2, cell2) = self.lstm2(output_lstm)  # N x L x H
        output_lstm, seq_lens = pad_packed_sequence(output_lstm, batch_first=True)
        output_lstm = self.dropout2(output_lstm)
        output_lstm = pack_padded_sequence(output_lstm, seq_lens, batch_first=True)
        output_lstm, seq_lens = pad_packed_sequence(output_lstm, batch_first=True)
        output_lstm, (hidden3, cell3) = self.lstm3(output_lstm)  # N x L x H
        output_lstm = self.dropout3(output_lstm)
        output_lstm = pack_padded_sequence(output_lstm, seq_lens, batch_first=True)
        output_lstm, (hidden4, cell4) = self.lstm4(output_lstm)  # N x L x H
        output, seq_lens = pad_packed_sequence(output_lstm, batch_first=True)
        #print(seq_lens)
        #print(output.shape) # N * L * 512
        #print(hidden4.shape) # 2 * N * 256
        #print(cell4.shape) # 2 * N * 256
        key = self.act(self.fc1(output))
        value = self.act(self.fc2(output))

        hidden = torch.cat([hidden4[0, :, :], hidden4[1, :, :]], dim=1) # concatenate hidden states of both directions
        cell = torch.cat([cell4[0, :, :], cell4[1, :, :]], dim=1)
        #print(key.shape) # N * L * out_dim
        #print(value.shape)# N * L * out_dim
        #print(hidden.shape) # N * 512
        #print(cell.shape) # N * 512
        return output, key, value, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, out_dim)
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
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.05)
        self.fc = nn.Linear(hidden_size, vocab_size)
        #self.fc.weight = self.embed.weight

    def forward(self, x, context, hidden1, cell1, hidden2, cell2):
        x = self.embed(x)
        #print(x.shape) # N * 256
        #print(context.shape) # N * 256
        x = torch.cat([x, context], dim=1)
        #print(x.shape) # N * 512
        hidden1, cell1 = self.lstm1(x, (hidden1, cell1))
        hidden2, cell2 = self.lstm2(hidden1, (hidden2, cell2))
        x = self.drop(hidden2)
        x = self.fc(x)
        return x, hidden1, cell1, hidden2, cell2


class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size_src, vocab_size_tgt, out_size, device, dropout_rate=0.2):
        super(NMT, self).__init__()
        self.encoder = Encoder(vocab_size_src, embed_size, hidden_size, out_size, dropout_rate)
        self.decoder = Decoder(vocab_size_tgt, embed_size, hidden_size * 2)
        self.attention = Attention(hidden_size * 2, out_size)
        self.device = device
        self.vocab_size_tgt = vocab_size_tgt

    def forward(self, src_sents, tgt_sents, tgt_lens, teacher_forcing_ratio):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: tensor of source sentence tokens
            tgt_sents: tensor of target sentence tokens, wrapped by `<s>` and `</s>`
            teacher_forcing_ratio: teacher forcing ratio

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """

        batch_size, max_len = tgt_sents.shape[0], tgt_sents.shape[1]
        prediction = torch.zeros(max_len, batch_size, self.vocab_size_tgt).to(self.device)

        output_lstm, key, value, hidden2, cell2 = self.encoder(src_sents)

        word, hidden1, cell1 = tgt_sents[:, 0], hidden2, cell2
        tgt_lens = tgt_lens.long()
        mask = torch.arange(tgt_lens.max()).unsqueeze(0) < tgt_lens.unsqueeze(1)
        
        mask = mask.to(self.device)

        for t in range(max_len):
            context, attention = self.attention(hidden2, key, value, tgt_lens)
            # print("word")
            # print(word.shape)
            word_vec, hidden1, cell1, hidden2, cell2 = self.decoder(word.long(), context, hidden1, cell1, hidden2,
                                                                    cell2)
            prediction[t] = word_vec
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            if teacher_force:
                word = tgt_sents[:, t]
            else:
                word = word_vec.max(1)[1]

        return prediction

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

        #return hypotheses
        return None



