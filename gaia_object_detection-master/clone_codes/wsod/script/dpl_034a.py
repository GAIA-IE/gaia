
# coding: utf-8

# In[1]:


exp_name = 'dpl_034a'


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[3]:


import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# In[4]:


import pickle
import csv
import os
import time
import datetime
import lmdb

from skimage.feature import peak_local_max
import scipy
from scipy import ndimage as ndi
from multiprocessing import Pool


# In[5]:


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
logfile = open(f'../../log/{exp_name}.txt', 'a')
logfile.write(f'\n\n\n\n ----------------- {now} ----------------- \n\n')
logfile.flush()
def log(string, stdout=True):
    if stdout:
        print(string)
    logfile.write(string + '\n')
    logfile.flush()


# In[6]:


with open('../../metadata/ont_m18/180_classes.csv', 'r') as fin:
    all_labels = [item.strip() for item in fin]
all_labels.append('background')
    
label2idx = {}
for i, l in enumerate(all_labels):
    label2idx[l] = i
    


# In[7]:


with open('../../../../data/eval_m18/jpg.txt', 'r') as fin:
    test_img_path = ['../../../../data/eval_m18/' + line.strip() for line in fin]


# In[8]:


test_img_path[0]


# In[9]:


resnet = torchvision.models.resnet152() # resnet.eval() is very important, do not forget this line during testing!
print(resnet.fc)
resnet.fc = nn.Linear(2048, len(all_labels)-1)

resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#saved_state_dict = torch.load(os.path.join('data/snapshots/','d2_rel_open_1600.pth'))
saved_state_dict = torch.load(os.path.join('../../snapshots', 'train_022', 'ckpt_5000'))
resnet.load_state_dict(saved_state_dict)
resnet.cuda(0)
resnet.eval()
resnet.float()


# In[10]:


finalconv_name = 'layer4'

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

resnet._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(resnet.parameters())
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())


# In[11]:


def preproc(im):
    target_size = 256
    max_size = 1024
    im_size_min = np.min(im.shape[0:2])
    im_size_max = np.max(im.shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)    
    return im


# In[12]:


batch_size = 1
model_output = {}
batch = []
batch_keys = []
img_shape = {}
for i, filename in enumerate(test_img_path):
    try:
        imgid = filename.split('/')[-1][:-9]
        with open(filename, 'rb') as fin:
            _ = fin.read(1024)
            imgbin = fin.read()
        imgbgr = cv2.imdecode(np.fromstring(imgbin, dtype='uint8'), cv2.IMREAD_COLOR)
        
        img_shape[imgid] = (imgbgr.shape[1], imgbgr.shape[0])
        
        imgbgr = preproc(imgbgr)
        imgrgb = imgbgr[:,:,[2,1,0]]        
        #assert(imgrgb.shape[0] == 224 and imgrgb.shape[1] == 224 and imgrgb.shape[2] == 3)
    except Exception as ex:
        log(str(ex))
        continue
        
    imgrgb = imgrgb.astype(float)/255.0
    
    imgrgb[:,:,0] = (imgrgb[:,:,0] - 0.485)/0.229
    imgrgb[:,:,1] = (imgrgb[:,:,1] - 0.456)/0.224
    imgrgb[:,:,2] = (imgrgb[:,:,2] - 0.406)/0.225

    imgrgb = imgrgb.transpose((2,0,1))
    batch.append(imgrgb)
    
    batch_keys.append(imgid)
    
    if len(batch) == batch_size or (i + 1) == len(test_img_path):
        batch = np.stack(batch)
    
        features_blobs = []
        
        with torch.no_grad():
            inp = torch.from_numpy(batch).float().cuda(0)
            outputs = resnet(inp)
            outputs = torch.cat([outputs, torch.ones(batch.shape[0], 1).cuda(0)], dim=1)
            h_x = F.softmax(outputs)#.data.squeeze()
            class_probs, class_idx = h_x.sort(1, True)
            class_idx = np.asarray(class_idx)
            class_probs = np.asarray(class_probs)
        
        for ii in range(len(batch_keys)):
            model_output[batch_keys[ii]] = {
                'features_blobs': features_blobs[0][ii],
                'sorted_labels': class_idx[ii],
                'sorted_probs': class_probs[ii],                
            }
        
        batch = []
        batch_keys = []
        
    if (i+1) % 100 == 0:
        log(f'Processed {i + 1} out of {len(test_img_path)} images.')


# In[13]:


ioa_thr = 0.9
topk_boxes = 300
rel_peak_thr = 0.7
rel_rel_thr = 0.7


# In[14]:


def postprocess(imgid):
    features_blob = model_output[imgid]['features_blobs']
    class_idx = model_output[imgid]['sorted_labels']
    class_probs = model_output[imgid]['sorted_probs']
    
    nc, h, w = features_blob.shape
    
    detections = []

    for ii in range(class_idx.shape[0]):
        if all_labels[class_idx[ii]] == 'background':
            break
        cam = weight_softmax[class_idx[ii]].dot(features_blob.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        bboxes = []
        box_scores = []

        peak_coords = peak_local_max(cam, exclude_border=False, threshold_rel=rel_peak_thr)

        heat_resized = cv2.resize(cam, img_shape[imgid])
        
        peak_coords_resized = ((peak_coords + 0.5) * 
                               np.asarray([list(reversed(img_shape[imgid]))]) / 
                               np.asarray([[h, w]])
                              ).astype('int32')

        for pk_coord in peak_coords_resized:
            pk_value = heat_resized[tuple(pk_coord)]
            mask = heat_resized > pk_value * rel_rel_thr
            labeled, n = ndi.label(mask) 
            l = labeled[tuple(pk_coord)]
            yy, xx = np.where(labeled == l)
            min_x = np.min(xx)
            min_y = np.min(yy)
            max_x = np.max(xx)
            max_y = np.max(yy)
            bboxes.append((min_x, min_y, max_x, max_y))
            box_scores.append(pk_value * class_probs[ii])

        box_idx = np.argsort(-np.asarray(box_scores))
        box_idx = box_idx[:min(topk_boxes, len(box_scores))]
        bboxes = [bboxes[i] for i in box_idx]
        box_scores = [box_scores[i] for i in box_idx]

        to_remove = []
        for iii in range(len(bboxes)):
            for iiii in range(iii):
                if iiii in to_remove:
                    continue
                b1 = bboxes[iii]
                b2 = bboxes[iiii]
                isec = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
                ioa1 = isec / ((b1[2] - b1[0]) * (b1[3] - b1[1]))
                ioa2 = isec / ((b2[2] - b2[0]) * (b2[3] - b2[1]))
                if ioa1 > ioa_thr and ioa1 == ioa2:
                    to_remove.append(iii)
                elif ioa1 > ioa_thr and ioa1 >= ioa2:
                    to_remove.append(iii)
                elif ioa2 > ioa_thr and ioa2 >= ioa1:
                    to_remove.append(iiii)

        for i in range(len(bboxes)): 
            if i not in to_remove:
                detections.append({
                    'label': all_labels[class_idx[ii]],
                    'score': box_scores[i],
                    'bbox': bboxes[i],
                    'bbox_normalized': np.asarray([
                        bboxes[i][0] / heat_resized.shape[1],
                        bboxes[i][1] / heat_resized.shape[0],
                        bboxes[i][2] / heat_resized.shape[1],
                        bboxes[i][3] / heat_resized.shape[0],
                    ]),
                    'model': 'WS_019',
                    'heatmap': cam,
                })
    
    
    return imgid, detections


# In[15]:


det_results = {}
cnt = 0
with Pool(20) as p:
    for i, res in enumerate(p.imap_unordered(postprocess, model_output.keys())):
        key, val = res
        det_results[key] = val
        cnt += 1
        if cnt % 100 == 0:
            print(f'Postprocessed {cnt} out of {len(model_output)} images.')


# In[16]:


save_obj(det_results, f'../../results/det_results_{exp_name}')

