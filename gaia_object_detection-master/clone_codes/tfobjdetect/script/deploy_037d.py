
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../../lib")
from object_detection.utils import ops as utils_ops


# In[2]:


import pickle
import cv2
import csv


# In[3]:


os.environ["CUDA_VISIBLE_DEVICES"]="1"


# ## Env setup

# In[4]:


# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[5]:


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[6]:


# What model to download.
MODEL_NAME = '../../checkpoints/faster_rcnn_nas_coco'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../../lib/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Load a (frozen) Tensorflow model into memory.

# In[7]:


od_graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[8]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[9]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Data Preparation

# In[10]:


with open('../../../../data/eval_m18/kf_id2path.pkl', 'rb') as fin:
    kf_id_to_img_path = pickle.load(fin)


# # Detection

# In[11]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[12]:


def run_inference_for_single_image(image):
  # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  # Run inference
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# In[ ]:


det_results = {}
image_shape = {}
num_imgs = len(kf_id_to_img_path)
for i, (imgid, imgpath) in enumerate(kf_id_to_img_path.items()):
    try:
        with open(imgpath, 'rb') as fin:
            if imgpath.endswith('.ldcc'):
                _ = fin.read(1024)
            imgbin = fin.read()
        imgbgr = cv2.imdecode(np.fromstring(imgbin, dtype='uint8'), cv2.IMREAD_COLOR)
        image_np = imgbgr[:,:,[2,1,0]]
        image_shape[imgid] = (image_np.shape[1], image_np.shape[0])
    except Exception as ex:
        print(imgid)
        print(ex)
        continue
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np)
    
    det_results[imgid] = output_dict
    
    if i % 100 == 0:
        print(f'{i} images processed out of {num_imgs}.')
    


# In[ ]:


idx2mid = {}
for item in label_map.item:
    idx2mid[item.id] = item.name


# In[ ]:


for imgid in det_results:
    det_results[imgid]['detection_boxes'] = det_results[imgid]['detection_boxes'][:,[1,0,3,2]]
    det_results[imgid]['detection_boxes_normalized'] = np.copy(det_results[imgid]['detection_boxes'])
    det_results[imgid]['detection_boxes'] = (det_results[imgid]['detection_boxes'] * np.asarray(image_shape[imgid]*2)).astype('int32')


# In[ ]:


det_results_2 = {}
for imgid in det_results:
    det_results_2[imgid] = []
    for ii in range(det_results[imgid]['num_detections']):
        if det_results[imgid]['detection_classes'][ii] == 0 or det_results[imgid]['detection_scores'][ii] == 0:
            continue
        label = idx2mid[det_results[imgid]['detection_classes'][ii]]
        if label == '/m/05czz6l':     ################################### IMPORTANT
            label = '/m/0cmf2'
        det_results_2[imgid].append({
            'label': label,
            'score': det_results[imgid]['detection_scores'][ii],
            'bbox': det_results[imgid]['detection_boxes'][ii],
            'bbox_normalized': det_results[imgid]['detection_boxes_normalized'][ii],
            'model': 'nasnet-faster-rcnn-coco'
        })


# In[ ]:


with open('../../results/det_results_m18_kf_coco_1.pkl', 'wb') as fout:
    pickle.dump(det_results_2, fout)


# In[ ]:


with open('../../../wsod/metadata/ont_m18/oi600_to_m18.pkl', 'rb') as fin:
    label_map = pickle.load(fin)


# In[ ]:


det_results_filtered = {}
for key, val in det_results_2.items():
    det_results_filtered[key] = []
    for det in val:
        label = label_map.get(det['label'])
        if label == None:
            continue
        det_results_filtered[key].append({
            'label': label,
            'score': det['score'],
            'bbox': det['bbox'],
            'bbox_normalized': det['bbox_normalized'],
            'model': det['model'],            
        })


# In[ ]:


with open('../../results/det_results_m18_kf_coco_1_filtered.pkl', 'wb') as fout:
    pickle.dump(det_results_filtered, fout)

