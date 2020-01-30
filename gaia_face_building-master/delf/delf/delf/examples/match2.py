# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf

from tensorflow.python.platform import app
from delf import feature_io

cmd_args = None

_DISTANCE_THRESHOLD = 0.8
import pickle
from glob import glob
import sys

feature_dir = sys.argv[1]
output_pickle = sys.argv[2]

import multiprocessing

pickle_in = open("num2name.p","rb")
num2name = pickle.load(pickle_in)

def count_i(land_name):
	data = land_name.split('/')[-1].split('.')[0]
	#file1 = open('txt_file_m18_m/'+data+'.txt','w')
	maxNum = 0
	maxName = ''
	for name in glob(feature_dir+'/*'):
		#print (name)
		name2 = name.split('/')
		number = name2[-1].split('.')[0]			
		# Read features.
		locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(land_name)
				#cmd_args.features_1_path)
		num_features_1 = locations_1.shape[0]
		#tf.logging.info("Loaded image 1's %d features" % num_features_1)
		locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(name)
				#cmd_args.features_2_path)
		num_features_2 = locations_2.shape[0]
		#tf.logging.info("Loaded image 2's %d features" % num_features_2)

		# Find nearest-neighbor matches using a KD tree.
		d1_tree = cKDTree(descriptors_1)
		_, indices = d1_tree.query(
				descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

		# Select feature locations for putative matches.
		locations_2_to_use = np.array([
				locations_2[i,]
				for i in range(num_features_2)
				if indices[i] != num_features_1
		])
		locations_1_to_use = np.array([
				locations_1[indices[i],]
				for i in range(num_features_2)
				if indices[i] != num_features_1
		])

		# Perform geometric verification using RANSAC.
		try: 
			_, inliers = ransac(
					(locations_1_to_use, locations_2_to_use),
					AffineTransform,
					min_samples=3,
					residual_threshold=20,
					max_trials=1000)

			tf.logging.info('Found %d inliers' % sum(inliers))
			#print(sum(inliers))
			#print(maxNum)
			score = int(sum(inliers))
			#file1.write(name+'\t'+str(sum(inliers))+'\n')		
			num = name.split('/')[-1].split('.')[0]  
			if score>35 and score>maxNum:
				maxNum = score
				maxName = num2name[number]
				#print (maxName)
				#print (maxNum)
				#break

		except:
				#print('fail')
				a=0

	#print (maxName)
	#result_dic[data] = maxName
	return maxName

def main(unused_argv):
	#tf.logging.set_verbosity(tf.logging.INFO)
	name_list = []
	C=0
	for land_name in glob(feature_dir+'/*'):
		#print (land_name)
		name_list.append(land_name)
		C+=1
		#if C>10:
		#	break
	print(name_list)

	
		#break
	pool = multiprocessing.Pool(processes=32)
	maxList = pool.map(count_i, name_list)
	result_dic = {}
	i=0
	for land_name in name_list:
		data = land_name.split('/')[-1].split('.')[0]
		result_dic[data] = maxList[i]
		i+=1
	pickle.dump( result_dic, open( output_pickle+".p", "wb"))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.register('type', 'bool', lambda v: v.lower() == 'true')
	parser.add_argument(
			'--image_1_path',
			type=str,
			default='test_images/image_1.jpg',
			help="""
			Path to test image 1.
			""")
	parser.add_argument(
			'--image_2_path',
			type=str,
			default='test_images/image_2.jpg',
			help="""
			Path to test image 2.
			""")
	parser.add_argument(
			'--features_1_path',
			type=str,
			default='test_features/image_1.delf',
			help="""
			Path to DELF features from image 1.
			""")
	parser.add_argument(
			'--features_2_path',
			type=str,
			default='test_features/image_2.delf',
			help="""
			Path to DELF features from image 2.
			""")
	parser.add_argument(
			'--output_image',
			type=str,
			default='test_match.png',
			help="""
			Path where an image showing the matches will be saved.
			""")
	cmd_args, unparsed = parser.parse_known_args()
	app.run(main=main, argv=[sys.argv[0]] + unparsed)
