# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import sys
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import math
import pickle
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry
from models import NMT
from dataloader import collate, Trainset, Testset


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train_epoch(model, train_loader, criterion, optimizer, teacher_forcing_ratio, clip_grad, device):
    model.train()
    model.to(device)

    running_loss = 0
    running_len = 0
    running_sample = 0
    for batch_idx, (source, target, target_lens) in enumerate(train_loader):
        print(batch_idx)
        optimizer.zero_grad()

        source = source.to(device)

        batch_size = target.shape[0]
        target = target.to(device)

        prediction = model(source, target, teacher_forcing_ratio) # prediction: L * N * vocab_size

        #print(prediction.shape)

        prediction = prediction.transpose(0, 1) # N * L * vocab_size
        #print(prediction.shape)
        #print(target.shape) # target: N * L

        output_list = []
        target_list = []
        total_len = 0
        for i in range(0, batch_size):
            t_len = target_lens[i]
            total_len += t_len
            output_list.append(prediction[i, 0:t_len])
            target_list.append(target[i, 0:t_len])

        outputs = torch.cat(output_list, 0)
        targets = torch.cat(target_list, 0).long()

        loss = criterion(outputs, targets)
        running_loss += loss.item()
        running_len += total_len
        running_sample += batch_size

        loss /= total_len
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()

        # release memory
        torch.cuda.empty_cache()

    # Return Average Loss and Average PPL
    return running_loss / running_sample, math.exp(running_loss / running_len)

def evaluate(model, dev_loader, criterion, device):

    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0
        running_len = 0
        running_sample = 0
        for batch_idx, (source, target, target_lens) in enumerate(dev_loader):
            source = source.to(device)
            batch_size = target.shape[0]
            target = target.to(device)

            prediction = model(source, target, teacher_forcing_ratio = 0)

            prediction = prediction.transpose(0, 1)

            output_list = []
            target_list = []
            total_len = 0
            for i in range(0, batch_size):
                t_len = target_lens[i]
                total_len += t_len
                output_list.append(prediction[i, 0:t_len])
                target_list.append(target[i, 0:t_len])

            outputs = torch.cat(output_list, 0)
            targets = torch.cat(target_list, 0).long()

            loss = criterion(outputs, targets)
            running_loss += loss.item()
            running_len += total_len
            running_sample += batch_size

        # Return Average Loss and Average PPL
        return running_loss / running_sample, math.exp(running_loss / running_len)


def train(args: Dict[str, str]):

    vocab = pickle.load(open(args['--vocab'], 'rb'))
    srcEntry = vocab.src # VocabEntry for src
    tgtEntry = vocab.tgt # VocabEntry for tgt
    vocab_size_src = len(srcEntry)
    vocab_size_tgt = len(tgtEntry)
    batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_dir = args['--save-to']
    max_epoch = int(args['--max-epoch'])
    max_patience = int(args['--patience'])
    lr_decay = float(args['--lr-decay'])

    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')




    # each sent is represented by indices in the corresponding VocabEntry
    train_data = Trainset(srcEntry.words2indices(train_data_src), tgtEntry.words2indices(train_data_tgt))
    dev_data = Trainset(srcEntry.words2indices(dev_data_src), tgtEntry.words2indices(dev_data_tgt))
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    dev_loader = DataLoader(dev_data, batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NMT(
        embed_size=int(args['--embed-size']), hidden_size=int(args['--hidden-size']), vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt, out_size=int(args['--hidden-size']), device=device, dropout_rate=float(args['--dropout']))

    num_trial = 0
    patience = 0
    epoch = valid_num = 0
    hist_valid_scores = []
    begin_time = time.time()



    criterion = nn.CrossEntropyLoss(reduction='sum',ignore_index=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    teacher_forcing_ratio = 1

    model_save_path = None
    saved_optimizer = None

    while epoch <= max_epoch:
        epoch += 1
        print("Training for Epoch " + str(epoch) + "\n")

        avg_loss, avg_ppl = train_epoch(model, train_loader, criterion, optimizer, teacher_forcing_ratio, clip_grad, device)
        print('epoch %d:  avg. loss %.2f, avg. ppl %.2f, time elapsed %.2f sec' % (epoch, avg_loss, avg_ppl, time.time() - begin_time),
              file=sys.stderr)

        #torch.save(model, model_save_dir + "/model_" + str(epoch) + ".pt")

        # decrease teacher_forcing_ration after certain epochs
        teacher_forcing_ratio = teacher_forcing_ratio - 0.05 if epoch > 10 else teacher_forcing_ratio

        # the following code performs validation on dev set, and controls the learning schedule
        # if the dev score is better than the last check point, then the current model is saved.
        # otherwise, we allow for that performance degeneration for up to `--patience` times;
        # if the dev score does not increase after `--patience` iterations, we reload the previously
        # saved best model (and the state of the optimizer), halve the learning rate and continue
        # training. This repeats for up to `--max-num-trial` times.


        cum_loss = cumulative_examples = cumulative_tgt_words = 0.
        valid_num += 1

        print('begin validation ...', file=sys.stderr)

        # compute dev. ppl and bleu
        dev_loss, dev_ppl = evaluate(model, dev_loader, criterion, device)


        print('validation: avg. loss %.2f, dev. ppl %f' % (dev_loss, dev_ppl), file=sys.stderr)

        is_better = len(hist_valid_scores) == 0 or dev_ppl < min(hist_valid_scores)
        hist_valid_scores.append(dev_ppl)

        torch.save(model, model_save_dir + '/model_' + str(epoch) + '.pt')
        if is_better:
            patience = 0
            model_save_path = model_save_dir + '/best_model.pt'
            print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
            torch.save(model, model_save_path)
            # You may also save the optimizer's state
            saved_optimizer = optimizer
        elif patience < max_patience:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

            if patience == max_patience:
                num_trial += 1
                print('hit #%d trial' % num_trial, file=sys.stderr)
                if num_trial == int(args['--max-num-trial']):
                    print('early stop!', file=sys.stderr)
                    exit(0)

                # load model
                if model_save_path:
                    model = torch.load(model_save_path)
                # You may also need to load the state of the optimizer saved before
                if saved_optimizer:
                    optimizer = saved_optimizer
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_decay

                # reset patience
                patience = 0




def beam_search(model, test_loader, beam_size, max_decoding_time_step, tgtEntry, device):
    hypotheses = []
    model.eval()
    model.to(device)
    for batch_idx, (src_sent) in enumerate(test_loader):
        src_sent = src_sent.to(device)
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_length=max_decoding_time_step)
        value = example_hyps.value
        translated_sent = []
        for wid in value:
            translated_sent.append(tgtEntry.index2word(wid))
        print(translated_sent)
        hypotheses.append(Hypothesis(translated_sent[1:-1], example_hyps.score))
    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    vocab = pickle.load(open(args['--vocab'], 'rb'))
    srcEntry = vocab.src  # VocabEntry for src
    tgtEntry = vocab.tgt  # VocabEntry for tgt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data_src = read_corpus(args['--test-src'], source='src')

    test_data = Testset(srcEntry.words2indices(test_data_src))
    test_loader = DataLoader(test_data, 1, shuffle=False)

    if args['--test-tgt']:
        test_data_tgt = read_corpus(args['--test-tgt'], source='tgt')

    print(f"load model from {args['--model-path']}", file=sys.stderr)


    model = NMT.load(args['--model-path'])

    hypotheses = beam_search(model, test_loader,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']), tgtEntry=tgtEntry, device=device)

    if args['--test-tgt']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['--output-path'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        print("train mode")
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
