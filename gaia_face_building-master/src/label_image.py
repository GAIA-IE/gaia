import os, sys

import tensorflow as tf
import pickle
from collections import defaultdict
flag_dict = defaultdict(list)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from glob import glob
# change this as you see fit
import sys
input_folder = sys.argv[1] #'/home/brian/tensorflow/models/research/object_detection/m18/m18/'
output_pickle = sys.argv[2] #'flag_m18_2'
for name in glob(input_folder+'/*'):
    image_path = name#sys.argv[1]
    print (name)
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("retrained_labels2.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph2.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            #print('%s (score = %.5f)' % (human_string, score))
            flag_dict[name].append((human_string, score))
with open(output_pickle+'.pickle', 'wb') as f:
    pickle.dump(flag_dict,f,protocol=pickle.HIGHEST_PROTOCOL)