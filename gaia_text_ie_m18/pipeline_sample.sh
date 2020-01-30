#!/usr/bin/env bash

######################################################
# Arguments
######################################################
# input root path
data_root=$1

# data folder that specified with language
data_root_en=${data_root}/en

######################################################
# Knowledge Extraction for each language
######################################################
python aida_utilities/preprocess_detect_languages.py ${data_root} ${data_root}

if [ -d "${data_root_en}/ltf" ]
then
    sh pipeline_sample_en.sh ${data_root_en}
else
    echo "No English documents in the corpus. Please double check. "
fi