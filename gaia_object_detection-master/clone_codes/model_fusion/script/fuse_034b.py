
# coding: utf-8

# In[1]:


import os
import sys
import pickle
import csv
import numpy as np


# ## Concatenating the results 

# In[2]:


with open('../../../tfobjdetect/results/det_results_m18_kf_oi_1_filtered.pkl', 'rb') as fin:
    det_results_oi = pickle.load(fin)

with open('../../../tfobjdetect/results/det_results_m18_kf_coco_1_filtered.pkl', 'rb') as fin:
    det_results_coco = pickle.load(fin)

with open('../../../wsod/results/det_results_dpl_034b.pkl', 'rb') as fin:
    det_results_ws = pickle.load(fin)
    
with open('/home/alireza/aida/rootdir/Ram/M18/main/m18_eval_representative_frames_voc_detn_fin_results_renamed.pkl', 'rb') as fin:
    det_results_voc = pickle.load(fin)


# In[3]:


with open('../../../wsod/metadata/ont_m18/mapping.pkl', 'rb') as fin:
    mid2ont, syn2mid, single_mids, mid2syn, class2ont, ont2name, class_names = pickle.load(fin)  


# In[4]:


label_set = set()
for imgid, det_list in det_results_oi.items():
    for det in det_list:
        if det['label'] in mid2syn:
            label_set.add(det['label'])
            det['label'] = mid2syn[det['label']]


# In[5]:


label_set


# In[6]:


label_set = set()
for imgid, det_list in det_results_coco.items():
    for det in det_list:
        if det['label'] in mid2syn:
            label_set.add(det['label'])
            det['label'] = mid2syn[det['label']]


# In[7]:


label_set


# In[8]:


len(set(det_results_oi)), len(set(det_results_coco)), len(set(det_results_ws)), len(set(det_results_voc))


# In[9]:


det_results_concat = {}
for imgid in det_results_oi:
    if imgid not in det_results_concat:
        det_results_concat[imgid] = []
    for det in det_results_oi[imgid]:
        det_results_concat[imgid].append({
            'label': det['label'],
            'score': det['score'],
            'bbox': det['bbox'],
            'bbox_normalized': det['bbox_normalized'],
            'model': 'oi',
        })

for imgid in det_results_coco:
    if imgid not in det_results_concat:
        det_results_concat[imgid] = []
        print("WARNING: image in coco not in oi")
    for det in det_results_coco[imgid]:
        det_results_concat[imgid].append({
            'label': det['label'],
            'score': det['score'],
            'bbox': det['bbox'],
            'bbox_normalized': det['bbox_normalized'],
            'model': 'coco',
        })
                
for imgid in det_results_ws:
    if imgid not in det_results_concat:
        det_results_concat[imgid] = []
        print("WARNING: image in ws not in oi")
    for det in det_results_ws[imgid]:
        det_results_concat[imgid].append({
            'label': det['label'],
            'score': det['score'],
            'bbox': det['bbox'],
            'bbox_normalized': det['bbox_normalized'],
            'model': 'ws',
        })

for imgid in det_results_voc:
    if imgid not in det_results_concat:
        det_results_concat[imgid] = []
        print("WARNING: image in ws not in oi")
    for det in det_results_voc[imgid]:
        det_results_concat[imgid].append({
            'label': det['label'],
            'score': det['score'],
            'bbox': det['bbox'],
            'bbox_normalized': det['bbox_normalized'],
            'model': 'voc',
        })
        


# In[11]:


len(set(det_results_concat))


# In[10]:


with open('../../results/det_results_concat_34b.pkl', 'wb') as fout:
    pickle.dump(det_results_concat, fout)


# ## Merging duplicate results

# In[12]:


def iou(det_bbox, gt_bbox):
    x_d_len = det_bbox[2] - det_bbox[0]
    y_d_len = det_bbox[3] - det_bbox[1]
    x_t_len = gt_bbox[2] - gt_bbox[0]
    y_t_len = gt_bbox[3] - gt_bbox[1]
    x_int_len = max(0, min(gt_bbox[2], det_bbox[2]) - max(gt_bbox[0], det_bbox[0]))
    y_int_len = max(0, min(gt_bbox[3], det_bbox[3]) - max(gt_bbox[1], det_bbox[1]))
    iou = (x_int_len*y_int_len) / (x_d_len*y_d_len + x_t_len*y_t_len - x_int_len*y_int_len)
    return iou
'''
def ioa(det_bbox, gt_bbox):
    x_d_len = det_bbox[2] - det_bbox[0]
    y_d_len = det_bbox[3] - det_bbox[1]
    x_t_len = gt_bbox[2] - gt_bbox[0]
    y_t_len = gt_bbox[3] - gt_bbox[1]
    x_int_len = max(0, min(gt_bbox[2], det_bbox[2]) - max(gt_bbox[0], det_bbox[0]))
    y_int_len = max(0, min(gt_bbox[3], det_bbox[3]) - max(gt_bbox[1], det_bbox[1]))
    iou = (x_int_len*y_int_len) / (x_d_len*y_d_len)
    return iou
'''


# In[13]:


_STAT_num_same_merged = 0
_STAT_num_diff_merged = 0

thresh_same = 0.5
thresh_diff = 0.7

all_groups = {}
for imgid, det in det_results_concat.items():
    groups = []
    for ii in range(len(det)):
        if det[ii]['label'] not in mid2ont:
            continue
        matching_gr = None
        for gr in groups:
            for item in gr:
                if det[ii]['label'] == det[item]['label'] and iou(det[ii]['bbox'], det[item]['bbox']) > thresh_same:
                    if matching_gr == None:
                        gr.append(ii)
                        matching_gr = gr
                    else:
                        matching_gr += gr
                        gr.clear()
                    _STAT_num_same_merged += 1
                    break
                if det[ii]['label'] != det[item]['label'] and iou(det[ii]['bbox'], det[item]['bbox']) > thresh_diff:
                    if matching_gr == None:
                        gr.append(ii)
                        matching_gr = gr
                    else:
                        matching_gr += gr
                        gr.clear()
                    _STAT_num_diff_merged += 1
                    break
                
        if matching_gr == None:
            groups.append([ii])
    all_groups[imgid] = groups
            


# In[14]:


_STAT_num_same_merged, _STAT_num_diff_merged


# In[15]:


with open('../../../wsod/metadata/ont_m18/class_names_all.pkl', 'rb') as fin:
    mid2name_all = pickle.load(fin)


# In[16]:


mid2level = {mid: len(name.split(' ')[0].split('.')) for mid, name in mid2name_all.items()}


# In[17]:


box_preference = {'voc': 2.0, 'coco': 2.0, 'oi': 2.0, 'ws': 1.0}
class_preference = {'voc': 1.0, 'coco': 2.0, 'oi': 3.0, 'ws': 4.0}

det_results_merged = {}
for imgid, groups in all_groups.items():
    det_results_merged[imgid] = []
    det = det_results_concat[imgid]
    for g in groups:
        if len(g) == 0:
            continue
        suff = '/J' if len(g) > 1 else ''

        mod_scores = [det[i]['score'] + (10.0 * class_preference[det[i]['model']]) + (100.0 * mid2level[det[i]['label']]) for i in g]
        imax = np.argmax(mod_scores)
        label = det[g[imax]]['label']
        model = det[g[imax]]['model'] + suff

        mod_scores = [det[i]['score'] + (10.0 * box_preference[det[i]['model']]) for i in g]
        imax = np.argmax(mod_scores)
        box = det[g[imax]]['bbox']
        box_norm = det[g[imax]]['bbox_normalized']
            
        scores = [det[ii]['score'] for ii in g]
        score = np.max(scores)
        
        if score < 0.01:
            continue
        
        det_results_merged[imgid].append({
            'label': label,
            'score': score,
            'bbox': box,
            'bbox_normalized': box_norm,
            'model': model,            
        })


# In[18]:


with open('../../results/det_results_merged_34b.pkl', 'wb') as fout:
    pickle.dump(det_results_merged, fout)

