
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence

from collections import namedtuple

from typing import List, Tuple, Dict, Set

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def init_uniform(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        nn.init.uniform_(m.weight.data, -0.1, 0.1)
    elif type(m) == nn.LSTM or type(m) == nn.LSTMCell:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.uniform_(param, -0.1, 0.1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size, dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        #self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        #self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        #self.lstm4 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.embed_drop = nn.Dropout(dropout_rate)

        #self.dropout1 = nn.Dropout(dropout_rate)
        #self.dropout2 = nn.Dropout(dropout_rate)
        #self.dropout3 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_size * 2, out_size)
        self.fc2 = nn.Linear(hidden_size * 2, out_size)
        self.act = nn.SELU(True)

    def forward_for_one(self, x):
        embed = self.embedding(x.long())
        #embed = self.embed_drop(embed)
        output, (hidden1, cell1) = self.lstm1(embed)
        key = self.act(self.fc1(output))
        value = self.act(self.fc2(output))
        #key = output
        #value = output
        #hidden = torch.cat([hidden1[0, :, :], hidden1[1, :, :]], dim=1)  # concatenate hidden states of both directions
        hidden = hidden1.sum(dim=0)
        #cell = torch.cat([cell1[0, :, :], cell1[1, :, :]], dim=1)
        cell = cell1.sum(dim=0)
        return None, key, value, hidden, cell

    def forward(self, x):
        #print("-----encoder forward function-----")
        x, seq_lens = pad_packed_sequence(x, batch_first=True)
        #print("x")
        #print(x.shape)
        #print(x)
        embed = self.embedding(x.long())
        #embed = self.embed_drop(embed)
        #print("embed")
        #print(embed.shape)
        #print(embed)
        embed = pack_padded_sequence(embed, seq_lens, batch_first = True)
        output_lstm, (hidden1, cell1) = self.lstm1(embed)  # N x L x H
        #output_lstm, seq_lens = pad_packed_sequence(output_lstm, batch_first=True)
        #output_lstm = self.dropout1(output_lstm)
        #output_lstm = pack_padded_sequence(output_lstm, seq_lens, batch_first=True)
        #output_lstm, (hidden2, cell2) = self.lstm2(output_lstm)  # N x L x H
        #output_lstm, seq_lens = pad_packed_sequence(output_lstm, batch_first=True)
        #output_lstm = self.dropout2(output_lstm)
        #output_lstm = pack_padded_sequence(output_lstm, seq_lens, batch_first=True)
        #output_lstm, seq_lens = pad_packed_sequence(output_lstm, batch_first=True)
        #output_lstm, (hidden3, cell3) = self.lstm3(output_lstm)  # N x L x H
        #output_lstm = self.dropout3(output_lstm)
        #output_lstm = pack_padded_sequence(output_lstm, seq_lens, batch_first=True)
        #output_lstm, (hidden4, cell4) = self.lstm4(output_lstm)  # N x L x H
        output, seq_lens = pad_packed_sequence(output_lstm, batch_first=True)
        #print("output")
        #print(output.shape)
        #print(output)
        #print(seq_lens)
        #print(output.shape) # N * L * 512
        #print(hidden4.shape) # 2 * N * 256
        #print(cell4.shape) # 2 * N * 256
        #######################key = self.act(self.fc1(output))
        #######################value = self.act(self.fc2(output))
        key = self.act(self.fc1(output))
        value = self.act(self.fc2(output))
        #key = output
        #value = output

        #print("key")
        #print(key.shape)
        #print(key)
        #print("value")
        #print(value.shape)
        #print(value)

        #print("hidden1")
        #print(hidden1.shape)
        #print(hidden1)
        #print("cell1")
        #print(cell1.shape)
        #print(cell1)
        ####################hidden = torch.cat([hidden1[0, :, :], hidden1[1, :, :]], dim=1) # concatenate hidden states of both directions
        hidden = hidden1.sum(dim=0)
        #print("hidden")
        #print(hidden.shape)
        #print(hidden)
        ####################cell = torch.cat([cell1[0, :, :], cell1[1, :, :]], dim=1)
        cell = cell1.sum(dim=0)
        #print("cell")
        #print(cell.shape)
        #print(cell)
        #print(key.shape) # N * L * out_dim
        #print(value.shape)# N * L * out_dim
        #print(hidden.shape) # N * 512
        #print(cell.shape) # N * 512
        return seq_lens, key, value, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(Attention, self).__init__()
        self.query_mlp = nn.Linear(hidden_dim, out_dim)
        self.attention_mlp1 = nn.Linear(out_dim, out_dim // 2)
        self.attention_mlp2 = nn.Linear(out_dim // 2, 1)
        self.attention_drop = nn.Dropout(0.2)
        self.E_softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, key, value, seq_lens):
        #print("-----attention forward function-----")
        #print("hidden")
        #print(hidden.shape)
        #print(hidden)
        query = self.query_mlp(hidden)
        #print("query")
        #print(query.shape)
        #print(query)

        #print("key")
        #print(key.shape) # key: N * L * out_dim
        #print(key)
        #print("value")
        #print(value.shape) # value: N * L * out_dim
        #print(value)
        #print(query.shape) # query: N * out_dim

        attention_score_hidden = torch.tanh(key + query.unsqueeze(1)) # query: N * 1 * out_dim
        #print(key + query.unsqueeze(1))
        #print("score_hidden")
        #print(attention_score_hidden.shape) # attention_score_hidden: N * L * out_dim
        #print(attention_score_hidden)
        attention_score_hidden = self.attention_drop(self.attention_mlp1(attention_score_hidden))
        attention_score_weight = self.attention_mlp2(attention_score_hidden).squeeze(2)
        #print("score_weight")
        #print(attention_score_weight.shape) # attention_score_weight: N * L
        #print(attention_score_weight)

        #print(attention_score_weight)
        #print(seq_lens)
        if seq_lens is not None:
            for i in range(0, len(seq_lens)):
                attention_score_weight[i, seq_lens[i]:] = float("-infinity")
        #print(attention_score_weight)
        attention_score = self.E_softmax(attention_score_weight)
        #attention_score = F.normalize(attention_score_weight, dim=-1)
        #print("score")
        #print(attention_score.shape)
        #print(attention_score)

        context = torch.bmm(attention_score.unsqueeze(1), value).squeeze(1) # bmm: (N * 1 * L), (N * L * dim) -> (N * 1 * dim)
        #print("context")
        #print(context.shape) # N * dim
        #print(context)
        return context, attention_score

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        #self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        #self.drop = nn.Dropout(0.05)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.act = nn.SELU(True)
        #self.fc.weight = self.embed.weight  # weight tying
        #self.E_softmax = nn.Softmax(dim=-1)

    def forward(self, word, context, hidden1, cell1):
        #print("----decoder forward function----")
        #print("context")
        #print(context)
        #print("word")
        #print(word)
        x = self.embed(word)
        #print("x")
        #print(x)
        #print(x.shape) # N * 256
        #print(context.shape) # N * 256
        x = torch.cat([x, context], dim=1)
        #print(x)
        #print(x.shape) # N * 512
        hidden, cell = self.lstm1(x, (hidden1, cell1))
        #hidden2, cell2 = self.lstm2(hidden1, (hidden2, cell2))

        #print("hidden")
        #print(hidden.shape)
        #print(hidden)
        #print("cell")
        #print(cell.shape)
        #print(cell)
        ##x = self.drop(hidden)
        x = self.act(self.fc(hidden))
        #x = F.softmax(x, dim=1)
        #print("x")
        #print(x.shape)
        #print(x)
        #return x, hidden1, cell1, hidden2, cell2
        return x, hidden, cell


class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size_src, vocab_size_tgt, out_size, device, dropout_rate=0.2):
        super(NMT, self).__init__()
        self.encoder = Encoder(vocab_size_src, embed_size, hidden_size, out_size, dropout_rate)
        self.decoder = Decoder(vocab_size_tgt, embed_size, hidden_size)
        self.attention = Attention(hidden_size, hidden_size )
        # Uniform Initialization
        self.encoder.apply(init_uniform)
        self.decoder.apply(init_uniform)
        self.attention.apply(init_uniform)

        self.device = device
        self.vocab_size_tgt = vocab_size_tgt

    def forward(self, src_sents, tgt_sents, teacher_forcing_ratio):
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

        batch_size, max_len = tgt_sents.shape[0:2]
        prediction = torch.zeros(max_len, batch_size, self.vocab_size_tgt).to(self.device)
        #print("prediction")
        #print(prediction.shape)

        seq_lens, key, value, hidden, cell = self.encoder(src_sents)

        #print("return results from encoder")
        #print(seq_lens)
        #print(key)
        #print(value)
        #print(hidden)
        #print(cell)

        word = tgt_sents[:, 0]
        word_vec_0 = torch.zeros(batch_size, self.vocab_size_tgt).to(self.device)
        word_vec_0[:, 1] = 1.0
        print("word_vec_0")
        #print(word_vec_0.shape)
        print(word_vec_0)
        prediction[0] = word_vec_0

        for t in range(1, max_len):
            print("word feed in at step: " + str(t))
            print(word)
            #print(word.shape)
            context, attention = self.attention(hidden, key, value, seq_lens)
            #print("return results form attention")
            #print("context")
            #print(context)
            # print("word")
            #print(word.shape)
            word_vec, hidden, cell = self.decoder(word.long(), context, hidden, cell)
            #print("return results from decoder")
            #print("word_vec")
            #print(word_vec.shape) # N * vocab_size
            #print(word_vec)
            prediction[t] = word_vec
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            if teacher_force:
                word = tgt_sents[:, t]
            else:
                word = word_vec.max(1)[1]
            print("predicted word at step: " + str(t))
            print(word_vec.max(1)[1])

        return prediction # L * N * vocab_size



    def beam_search(self, src_sents, beam_size=10, max_length=100):

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


        print("beam search with beam size and length " + str(beam_size) + " " + str(max_length))

        _, key, value, hidden, cell = self.encoder.forward_for_one(src_sents.to(self.device))

        sequences = []

        context_map = {}
        for idx in range(1, max_length):
            # print("-------" + str(idx) + "--------")
            if idx > 10  and sequences[0][0][-1] == 2: #end of sentence </s>
                break
            if idx == 1:
                word = torch.LongTensor([1]).to(self.device) # start of sentence <s>
                if hidden in context_map.keys():
                    context = context_map[hidden]
                else:
                    context, attention = self.attention(hidden, key, value, None)
                    context_map[hidden] = context
                word_vec, hidden, cell = self.decoder(word, context, hidden, cell)
                word_vec = F.softmax(word_vec, dim=1) # 1 * vocab_size
                for i in range(1, self.vocab_size_tgt):
                    sequences.append([[1, i], word_vec[0, i], hidden, cell])
                # print(sequences)
                ordered = sorted(sequences, key=lambda tup: tup[1], reverse=True)
                sequences = ordered[:min(beam_size, len(ordered))]
                # print(ordered)
            else:
                new_seqs = []
                for seq, prob, hidden, cell in sequences:
                    # print(seq)
                    # print(prob)
                    if seq[-1] == 2: #end of sentence </s>
                        new_seqs.append([seq, prob, hidden, cell])
                        # print("flag")
                        # candidates_list.append([seq, prob, hidden1, cell1, hidden2, cell2])
                    else:
                        word = torch.LongTensor([seq[-1]]).to(self.device)
                        # print("word:")
                        # print(word)
                        if hidden in context_map.keys():
                            context = context_map[hidden]
                        else:
                            context, attention = self.attention(hidden, key, value, None)
                            context_map[hidden] = context
                        word_vec, hidden, cell = self.decoder(word, context, hidden, cell)
                        word_vec = F.softmax(word_vec, dim=1)
                        for i in range(1, self.vocab_size_tgt):
                            new_seqs.append([seq + [i], prob * word_vec[0, i], hidden, cell])
                            # print(word_vec)
                            # print(word_vec.shape)
                            # print(word_vec.sum())
                # print(new_seqs)
                ordered = sorted(new_seqs, key=lambda tup: tup[1], reverse=True)
                sequences = ordered[:min(beam_size, len(ordered))]
                # print(ordered)
                # print(sequences)
        # candidates_list += sequences
        ordered = sorted(sequences, key=lambda tup: tup[1], reverse=True)

        for seq, prob, hidden, cell in ordered:
            # print(seq)
            # print(prob)
            if seq[-1] == 2 and len(seq) > 10:
                return Hypothesis(seq, prob)
        return Hypothesis(ordered[0][0], ordered[0][1])



