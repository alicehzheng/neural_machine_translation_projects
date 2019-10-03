#!/bin/sh

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de.wmixerprep"
dev_tgt="data/valid.de-en.en.wmixerprep"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir"

mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir} \
    --valid-niter 2400 \
    --batch-size 64 \
    --hidden-size 512 \
    --embed-size 512 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --max-epoch 25 \
    --lr-decay 0.5

python nmt.py \
    decode \
    --cuda \
    --vocab ${vocab} \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    --model-path ${work_dir}/best_model.pt \
    --test-src ${train_src} \
    --test-tgt ${train_tgt} \
    --output-path ${work_dir}/decode.txt

perl multi-bleu.perl ${train_tgt} < ${work_dir}/decode.txt