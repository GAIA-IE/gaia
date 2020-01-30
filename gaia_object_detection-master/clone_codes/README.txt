-----
Pip

pip install -r req.txt

-----
Docker

$ docker build . --tag [TAG]
$ docker run -itd -p [HOST_PORT]:8082 --name [CONTAINER_NAME] [TAG] /bin/bash 
$ docker exec -it [CONTAINER_NAME] /bin/bash

\# jupyter notebook --allow-root --ip=0.0.0.0 --port=8082
-----

# in jpg folder: find '*.jpg*' > jpg.txt
# in keyframes folder: find '*.png*' > keyframe.txt

# Run model 1 on jpg data
python tfobjdetect/ipynb/deployment/deploy_037a (needs jpg.txt and path to jpgs) (outputs det_results_jpg_oi_filtered.pkl)

# Run model 2 on jpg data
python tfobjdetect/ipynb/deployment/deploy_037b (needs jpg.txt and path to jpgs) (outputs det_results_jpg_coco_filtered.pkl)

# Run model 1 on keyframe data
python tfobjdetect/ipynb/deployment/deploy_037c (needs keyframe.txt and masterShotBoundary.msb and path to keyframes) (outputs det_results_kf_oi_filtered.pkl AND kf_id2path.pkl)

# Run model 2 on keyframe data
python tfobjdetect/ipynb/deployment/deploy_037d (needs kf_id2path.pkl) (outputs det_results_kf_coco_filtered.pkl)

# Run model 3 on jpg data
python wsod/ipynb/deploy/dpl_034a (needs jpg.txt and path to jpgs) (outputs det_results_034a.pkl)

# Run model 3 on keyframe data
python wsod/ipynb/deploy/dpl_034b (needs kf_id2path.pkl) (outputs det_results_034b.pkl)

# get eval_jpg_voc_detn_fin_results.pkl and eval_representative_frames_voc_detn_fin_results.pkl from Arka

# postprocess Arka's results
python model_fusion/ipynb/import_rams/import_011 (needs eval_representative_frames_voc_detn_fin_results.pkl) outputs (eval_representative_frames_voc_detn_fin_results_renamed.pkl)


# Integrate jpg results
python model_fusion/ipynb/fusion/fuse_034a (needs det_results_jpg_oi_1_filtered.pkl
                                            det_results_jpg_coco_1_filtered.pkl
                                            det_results_dpl_034a.pkl
                                            eval_jpg_voc_detn_fin_results.pkl) (outputs det_results_merged_034a.pkl)


# Integrate keyframe results
python model_fusion/ipynb/fusion/fuse_034b (needs det_results_kf_oi_1_filtered.pkl
                                            det_results_kf_coco_1_filtered.pkl
                                            det_results_dpl_034b.pkl
                                            eval_representative_frames_voc_detn_fin_results_renamed.pkl) (outputs det_results_merged_034b.pkl)


# Integrate all results and postprocess to prepare it for AIF
python model_fusion/ipynb/export/ex_034 (needs det_results_merged_034a.pkl and det_results_merged_034b.pkl) (outputs aida_output_34.pkl)


# send aida_output_34.pkl to Brian's system 
