## AIDA code documents (Face, Flag, Landmark, ttl file generator)
Source provided by Brian Chen.

Face: https://github.com/davidsandberg/facenet
Face detection for image & key frames

```
CUDA_VISIBLE_DEVICES=0 python src/align/align_dataset_mtcnn.py \
[input directory of img or key frames] \
datasets/[output dir] \
--image_size 160 \
--margin 32
```

## Face Recognition for image & key frames

```
CUDA_VISIBLE_DEVICES=0 python src/classifier.py CLASSIFY datasets/[1.1 output directory] models/facenet/20180402-114759/20180402-114759.pb models/google500_2_classifier.pkl [result pickle file name] --batch_size 1000 > results/[result txt file name].txt
```

### Store bounding box

```
python src/bbox.py [1.1 output directory] [bbox pickle name]
```

## Building & Flag
Object detection for building and flag: https://github.com/tensorflow/models/tree/master/research/object_detection
Under the object detection directory in tensorflow, run obj_preprocess.ipynb
In line 10, change the folder path to img folder.
In line 11, set the saved pickle file name for flag detection.
The last block saves the building_list.txt

### Flag
Save cropped flag files:
Run save_flag_crop.ipynb under object_detection directory, 
line 7 set the output directory of the cropped flags
line 8 is the pickle file stored the object detection result generated from 2.2.1
Classify the flags
python src/label_image.py [output dir of 2.2.1] [output flag result pickle name]

### Building
Feature extraction: https://github.com/tensorflow/models/tree/master/research/delf
Under the delf directory in tensorflow (tensorflow/models/research/delf/delf/python/examples), run

```
CUDA_VISIBLE_DEVICES=0 python extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path [path to the building_list.txt created by obj_preprocess.ipynb] /building_list.txt \
  --output_dir [output feature directory]
```

### Feature matching

```
python match2.py [output feature directory from 2.3.1] [output result pickle name]
```

### Output ttl files
Read text entity from RPI ttl files
Run read_RPI_entity.ipynb
Block 2 line 12 is the ttl folder fenerated by RPI
Block 2 line 34 is the stored pickle file

Run create_ttl_m18.ipynb (Take m18 as an example)
Need to change the path in the first block when executing
parent_file: Parent child document relation given by LDC (parent_children.sorted.tab)
video_frame_mapping = video frame id mapping from LDC (masterShotBoundary.msb)
face_img_result = Output from 1.2, face classification pickle file for img
face_frame_result = Output from 1.2, face classification pickle file for video frame
bbox_img: Output from 1.3, bbox of image
bbox_frame = Output from 1.3, bbox of video frame
az_obj_graph = Object detection graph from Alireza (rdf_graphs_34.pkl)
az_obj_jpg = Object detection dictionary for img from Alireza (det_results_merged_34a_jpg.pkl)
az_obj_kf = Object detection dictionary for key frame from Alireza (det_results_merged_34b_kf.pkl)
Lorelei_path = External knowledge Lorelei path (entities.tab')
flag_result = Output from 2.2.2 (flag_m18_2.pickle)
landmark_result = Output from 2.3.2 (result_dic_m18_new.p)
RPI_entity = Output from 3.1 (PT003_r1.pickle)
input_img_path = jpg file directory path 
outputN = Output ttl folder name

The output file will be ttl files in the folder

## Docker

```
$ docker build . --tag aida-face
$ docker run -itd --name aida-face-env aida-face:latest /bin/bash
$ docker exec -it aida-face-env /bin/bash
```
