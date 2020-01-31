#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python aida_depend.py \
--model_path "/data/m1/whites5/AIDA/DependencyParse/models/" \
--model_name "biaffine.pt" \
--gpu \
--punctuation '.' '``' "''" ':' ',' \
--decode mst \
test_phase \
--parser biaffine \
--input_data "/home/whites5/AIDA/RelationExtraction/utils/tmp/en.doc_segments.txt" \
--output_path "/home/whites5/AIDA/RelationExtraction/utils/tmp/en.dep_parse.txt"
