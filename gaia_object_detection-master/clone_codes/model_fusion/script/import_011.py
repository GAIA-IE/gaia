
# coding: utf-8

# In[1]:


import os
import sys
import pickle
import csv
import numpy as np


# In[4]:


with open('/home/alireza/aida/rootdir/Ram/M18/main/m18_eval_representative_frames_voc_detn_fin_results.pkl', 'rb') as fin:
    det_results_voc = pickle.load(fin)


# In[5]:


with open('../../../../data/eval_m18/kf_id2path.pkl', 'rb') as fin:
    kf_id_to_img_path = pickle.load(fin)


# In[6]:


kf_fname_to_id = {val.split('/')[-1].split('.')[0]: key for key, val in kf_id_to_img_path.items()}


# In[8]:


det_results_new = {}
for key, val in det_results_voc.items():
    if key in kf_fname_to_id:
        det_results_new[kf_fname_to_id[key]] = val


# In[9]:


len(det_results_new)


# In[12]:


with open('/home/alireza/aida/rootdir/Ram/M18/main/m18_eval_representative_frames_voc_detn_fin_results_renamed.pkl', 'wb') as fout:
    pickle.dump(det_results_new, fout)

