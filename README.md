# GAIA: A Cross-media Fine-grained Knowledge Extraction System

Latest version is in [UIUC-Text-IE](https://github.com/limanling/uiuc_ie_pipeline_fine_grained)(TextIE) and [GAIA-AIDA](https://github.com/GAIA-AIDA)(VisionIE).

Table of Contents
=================
  * [Overview](#overview)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)

## Overview
<p align="center">
  <img src="data/images/overview_all.png" alt="Photo" style="width="100%;"/>
</p>

## Requirements

#### Environment

- Docker

#### Pacakges
You can install the environment using `requirements.txt` for each component.

```pip
docker pull mongo
docker pull panx27/edl
docker pull limanling/uiuc_ie_m18
docker pull charlesztt/aida_event
docker pull dylandilu/event_coreference_xdoc
docker pull wangqy96/aida_nominal_coreference_en
docker pull frnkenstien/corenlp
pip install -r gaia_object_detection-master/requirements.txt
pip install -r gaia_face_building-master/requirements.txt
pip install -r gaia_grounding_merging-master/requirements.txt
```

## Quickstart

### Textual Entity Extraction and Linking, Relation Extraction, Event Extraction and corefernece

One single script to run text information extraction, including entity extraction, relation extraction and event extraction.

```bash
sh gaia_text_ie_m18/pipeline_sample.sh ${data_root}
```

Example files are in `gaia_text_ie_m18/data/testdata`.

### Visual Entity Extraction

```
python gaia_object_detection-master/tfobjdetect/ipynb/deployment/deploy_037a(b/c/d/e)
python gaia_object_detection-master/wsod/ipynb/deploy/dpl_034a(b)
python gaia_object_detection-master/model_fusion/ipynb/fusion/fuse_034a(b)
python gaia_object_detection-master/model_fusion/ipynb/export/ex_034
```

### Visual Entity Linking and Coreference

```
python gaia_face_building-master/src/align/align_dataset_mtcnn.py \
[input directory of img or key frames] \
datasets/[output dir] \
--image_size 160 \
--margin 32

python gaia_face_building-master/src/classifier.py CLASSIFY \ 
datasets/[1.1 output directory] \
models/facenet/20180402-114759/20180402-114759.pb \
models/google500_2_classifier.pkl \
[result pickle file name] \
--batch_size 1000 > results/[result txt file name].txt

python gaia_face_building-master/src/bbox.py [1.1 output directory] [bbox pickle name]

python gaia_face_building-master/extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path [path to the building_list.txt created by obj_preprocess.ipynb] /building_list.txt \
  --output_dir [output feature directory]
  
python gaia_face_building-master/match2.py [output feature directory from 2.3.1] [output result pickle name]
```

### Cross-media Fusion

This step synthesises the knowledge from text, images, and videos to construct a comprehensive multimedia knowledge base. The crossmedia fusion is done by grouding text to visual regions and the results of entity linking. Runing following steps for crossmedia fusion: 

```
Visual_Features.ipynb
Grounding.ipynb
Merge.ipynb
```

