# GAIA: A Cross-media Fine-grained Knowledge Extraction System

[GAIA: A Cross-media Fine-grained Knowledge Extraction System]() 

Table of Contents
=================
  * [Overview](#overview)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Citation](#citation)

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
pip install -r gaia_object_detection-master/requirements.txt
pip install -r gaia_face_building-master/requirements.txt
pip install -r gaia_grounding_merging-master/requirements.txt
```

## Quickstart

### Textual Entity Extraction and Linking, Relation Extraction, Event Extraction and corefernece

One single script to run text information extraction, including entity extraction, relation extraction and event extraction.

```bash
sh pipeline_sample.sh ${data_root}
```

Example files are in `gaia_text_ie_m18/data/testdata`.

### Visual Entity Extraction

```
python tfobjdetect/ipynb/deployment/deploy_037a(b/c/d/e)
python wsod/ipynb/deploy/dpl_034a(b)
python model_fusion/ipynb/fusion/fuse_034a(b)
python model_fusion/ipynb/export/ex_034
```

### Visual Entity Linking and Coreference

```
CUDA_VISIBLE_DEVICES=0 python gaia_face_building-master/src/align/align_dataset_mtcnn.py \
[input directory of img or key frames] \
datasets/[output dir] \
--image_size 160 \
--margin 32

CUDA_VISIBLE_DEVICES=0 python gaia_face_building-master/src/classifier.py CLASSIFY \ 
datasets/[1.1 output directory] \
models/facenet/20180402-114759/20180402-114759.pb \
models/google500_2_classifier.pkl \
[result pickle file name] \
--batch_size 1000 > results/[result txt file name].txt

python gaia_face_building-master/src/bbox.py [1.1 output directory] [bbox pickle name]

CUDA_VISIBLE_DEVICES=0 python gaia_face_building-master/extract_features.py \
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

## Citation
```
@InProceedings{ligaia:19,
    author  = {Manling Li and Ying Lin and Ananya Subburathinam and Spencer Whitehead and Xiaoman Pan and Di Lu and Qingyun Wang and Tongtao Zhang and Lifu Huang and Heng Ji and Alireza Zareian and Hassan Akbari and Brian Chen and Bo Wu and Emily Allaway and Shih-Fu Chang and Kathleen McKeown and Yixiang Yao and Jennifer Chen and Eric Berquist and Kexuan Sun and Xujun Peng and Ryan Gabbard and Marjorie Freedman and Pedro Szekely and T.K. Satish Kumar and Arka Sadhu and Ram Nevatia and Miguel Rodriguez and Yifan Wang and Yang Bai and Ali Sadeghian and Daisy Zhe Wang},
    title   = {GAIA at SM-KBP 2019 - A Multi-media Multi-lingual Knowledge Extraction and Hypothesis Generation System},
    booktitle = "Proceedings of TAC KBP 2019, the 26th International Conference on Computational Linguistics: Technical Papers",
    year    = "2019"
}
```
