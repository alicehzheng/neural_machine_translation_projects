from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence
import numpy as np


def collate(seqs):
    # print(seqs)
    batch_size = len(seqs)
    # print(batch_size)
    seq_lengths = torch.zeros(batch_size, dtype=torch.int)
    max_seq_len = 0
    for i in range(0, batch_size):
        seq_lengths[i] = seqs[i][0].shape[0]
        max_seq_len = max(max_seq_len, seqs[i][0].shape[0])
    # print(seq_lengths)
    seq_lengths, perm_idx_1 = seq_lengths.sort(0, descending=True)
    # print(seq_lengths)
    # print(perm_idx)
    data = torch.zeros([batch_size, max_seq_len, 40])
    for i in range(0, batch_size):
        x_idx = perm_idx_1[i]
        # print(seqs[x_idx][0].shape)
        data[i, 0:seq_lengths[i], :] = torch.LongTensor(seqs[x_idx][0])

    packed_data = pack_padded_sequence(data, seq_lengths.numpy(), batch_first=True)

    lens = torch.zeros(batch_size, dtype=torch.int)
    max_target_len = 0
    for i in range(0, batch_size):
        lens[i] = seqs[i][1].shape[0] - 1 # omitting leading `<s>`
        max_target_len = max(max_target_len, lens[i])

    target = torch.zeros([batch_size, max_target_len])

    target_lens = torch.zeros(batch_size, dtype=torch.int)
    for i in range(0, batch_size):
        t_idx = perm_idx_1[i]
        target_lens[i] = lens[t_idx] - 1 # omitting leading `<s>`
        target[i, 1:target_lens[i] + 1] = torch.LongTensor(seqs[t_idx][1])

    return [packed_data, target, target_lens, max_target_len]


class Trainset(Dataset):
    def __init__(self, src_sents, tgt_sents):
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.train_size = src_sents.shape[0]

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        return self.src_sents[idx], self.tgt_sents[idx]


class Testset(Dataset):
    def __init__(self, src_sents):
        self.src_sents = src_sents
        self.test_size = src_sents.shape[0]

    def __len__(self):
        return self.test_size

    def __getitem__(self, idx):
        return self.src_sents[idx]
