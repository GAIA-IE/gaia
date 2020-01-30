print("begin smoke test")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import cv2
import os
import json
import sys
import lmdb
from collections import defaultdict
import random
from utils import *
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # GPU_ID
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True,allow_soft_placement=True)

#############
#ToDo: add capabality of finding jpgs in parent even if ID.ltf.xml or ID.psm.xml is given
#############

working_path = '/root/data/'
#working_path = '/home/bobby/aida_copy/AIDA/M18_copy/data/'

# look at README to run with volume attached /nfs/isicvlnas01/projects/AIDA/2019-05-dryrun/dryrun

#/data/src/bobby/data/data_bobby
#models
#│   ├── model_ELMo_PNASNET_VOA_norm.data-00000-of-00001
#│   ├── model_ELMo_PNASNET_VOA_norm.index
#│   ├── model_ELMo_PNASNET_VOA_norm.meta
#│   ├── model_universal_no_recons_ins_only.data-00000-of-00001
#│   ├── model_universal_no_recons_ins_only.index
#│   └── model_universal_no_recons_ins_only.meta
#├── objdet_results
#│   └── E
#│       ├── det_results_merged_34a_jpg.pkl
#│       └── det_results_merged_34b_kf.pkl
#├── raw_files
#│   └── RPI_TA1_E_asr
#│       ├── en_asr_mapping.zip
#│       ├── en_asr.zip
#│       ├── ltf_asr
#│       │   ├── HC0000FUP.ltf.xml

corpus_path = '/root/dryrun/'
#corpus_path = '/nfs/isicvlnas01/projects/AIDA/2019-05-dryrun/dryrun/'
#corpus_path = '/dvmm-filer2/projects/AIDA/data/ldc_eval_m18/LDC2019E42_AIDA_Phase_1_Evaluation_Source_Data_V1.0/'
##corpus_path = '/dvmm-filer2/projects/AIDA/data/ldc_isi_dryrun3/dryrun-updated_tmp/dryrun/'
##corpus_path = '/dvmm-filer2/projects/AIDA/data/ldc_isi_dryrun3/dryrun/' # D3
##corpus_path = '/dvmm-filer2/projects/AIDA/data/ldc_isi_dryrun/dryrun/' # D2
#img_path = corpus_path
kfrm_path = corpus_path + 'data/video_shot_boundaries/representative_frames'
# DPN added path without 'sorted'
parent_child_tab = corpus_path + 'docs/parent_children.tab'
#parent_child_tab = corpus_path + 'docs/parent_children.sorted.tab'
#kfrm_msb = corpus_path + 'docs/masterShotBoundary.msb'
#print('Check Point: Raw Data corpus_path change',corpus_path)
##rpi mention results in AIF
#p_f = 'E'
#video_asr_path = working_path + 'raw_files/RPI_TA1_'+p_f+'_asr/ltf_asr'
#video_map_path = working_path + 'raw_files/RPI_TA1_'+p_f+'_asr/map_asr'
#print('Check Point: RPI path change',video_asr_path)
#det_results_path_img = working_path + 'objdet_results/'+p_f+'/det_results_merged_34a_jpg.pkl'
## det_results_path_img = working_path + 'objdet_results/det_results_merged_32_jpg.pkl'
#det_results_path_kfrm = working_path + 'objdet_results/'+p_f+'/det_results_merged_34b_kf.pkl'
## det_results_path_kfrm = working_path + 'objdet_results/det_results_merged_33_kf.pkl'
#print('Check Point: Alireza path change:','\n',det_results_path_img,'\n', det_results_path_kfrm,'\n')
#loading grounding pretrained model
print('Loading grounding pretrained model...')
model_path = working_path+ 'models/model_ELMo_PNASNET_VOA_norm'
sess, graph = load_model(model_path,config)
input_img = graph.get_tensor_by_name("input_img:0")
mode = graph.get_tensor_by_name("mode:0")
v = graph.get_tensor_by_name("image_local_features:0")
v_bar = graph.get_tensor_by_name("image_global_features:0")
print('Loading done.')


#preparing dicts
parent_dict, child_dict = create_dict(parent_child_tab)
id2dir_dict_kfrm = create_dict_kfrm(kfrm_path, kfrm_msb, video_asr_path, video_map_path)

print("smoke test complete")

