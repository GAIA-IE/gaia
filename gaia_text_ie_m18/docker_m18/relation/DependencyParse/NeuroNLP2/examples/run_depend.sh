#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/depend.py --cuda --mode FastLSTM --num_epochs 200 --batch_size 32 --hidden_size 512 --num_layers 3 \
 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 \
 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --pos \
 --objective cross_entropy --decode mst \
 --word_embedding sskip --word_path "/data/m1/whites5/AIDA/pretrained_embeddings/enwiki.cbow.50d.txt.gz" \
 --punctuation '.' '``' "''" ':' ',' \
 --train "/data/m1/whites5/AIDA/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllx" \
 --dev "/data/m1/whites5/AIDA/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-dev.conllx" \
 --test "/data/m1/whites5/AIDA/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllx" \
 --model_path "/data/m1/whites5/AIDA/DependencyParse/models/" \
 --model_name 'biaffine.pt'