from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence
import numpy as np


def collate(seqs):
    print("collate function")
    #print(seqs)
    batch_size = len(seqs)
    #print(batch_size)
    seq_lengths = torch.zeros(batch_size, dtype=torch.int)
    max_seq_len = 0
    for i in range(0, batch_size):
        seq_lengths[i] = seqs[i][0].shape[0]
        max_seq_len = max(max_seq_len, seqs[i][0].shape[0])
    
    #print("seq_lengths")
    #print(seq_lengths)
    seq_lengths, perm_idx_1 = seq_lengths.sort(0, descending=True)
    data = torch.zeros([batch_size, max_seq_len])
    for i in range(0, batch_size):
        x_idx = perm_idx_1[i]
        # print(seqs[x_idx][0].shape)
        data[i, 0:seq_lengths[i]] = torch.LongTensor(seqs[x_idx][0])
        #print(data[i])
    print("data")
    print(data)
    print("seq_lengths")
    print(seq_lengths)
    packed_data = pack_padded_sequence(data, seq_lengths.numpy(), batch_first=True)

    lens = torch.zeros(batch_size, dtype=torch.int)
    max_target_len = 0
    for i in range(0, batch_size):
        #print(seqs[i][1])
        #lens[i] = seqs[i][1].shape[0] - 1  # omitting leading `<s>`
        lens[i] = seqs[i][1].shape[0]
        max_target_len = max(max_target_len, lens[i])

    target = torch.zeros([batch_size, max_target_len])

    target_lens = torch.zeros(batch_size, dtype=torch.int)
    for i in range(0, batch_size):
        t_idx = perm_idx_1[i]
        target_lens[i] = lens[t_idx]
        #target[i, 0:target_lens[i]] = torch.LongTensor(seqs[t_idx][1][1:])
        target[i, 0:target_lens[i]] = torch.LongTensor(seqs[t_idx][1][:])
        #print(target[i])
    print("target")
    print(target)
    print("target_lens")
    print(target_lens)
    return [packed_data, target, target_lens]


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
