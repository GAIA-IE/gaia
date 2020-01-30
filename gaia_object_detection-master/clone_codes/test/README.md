Testing instructions.

Everything should be in github repo except the following .pb file which needs to copied manually.

```

./tfobjdetect/checkpoints/faster_rcnn_inception_resnet_v2_atrous_oid/frozen_inference_graph.pb
```

Example building and running 

```

$ docker build . --tag jan23-02
Sending build context to Docker daemon  617.7MB
Step 1/20 : FROM python:3.6.3
 ---> a8f7167de312
Step 2/20 : MAINTAINER Dan Napierski (ISI) <dan.napierski@toptal.com>
 ---> Using cache
 ---> 62306400abb0
Step 3/20 : WORKDIR /root/src/
 ---> Using cache
 ---> 8278bfb73f9e
Step 4/20 : RUN apt-get upgrade && apt-get update && apt-get -y install apt-utils unzip git python-pil python-lxml python-tk
 ---> Using cache
 ---> 1d949879333e
Step 5/20 : RUN pip install --upgrade pip
 ---> Using cache
 ---> d8a90d1f06b8
Step 6/20 : RUN mkdir tf
 ---> Using cache
 ---> 3249d82dc1b7
Step 7/20 : WORKDIR /root/src/tf
 ---> Using cache
 ---> 91cd6b613eec
Step 8/20 : RUN git clone https://github.com/tensorflow/models.git
 ---> Using cache
 ---> c2bf25cb90f0
Step 9/20 : ENV PYTHONPATH=/usr/local/bin/python:/root/src/tf/models/research:/root/src/tf/models/research/slim:.
 ---> Using cache
 ---> 590886d76136
Step 10/20 : WORKDIR /root/src/tf/models/research
 ---> Using cache
 ---> d5417e9c7880
Step 11/20 : RUN wget -O protobuf.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
 ---> Using cache
 ---> 54dffa9c7004
Step 12/20 : RUN unzip protobuf.zip
 ---> Using cache
 ---> ae29cccef9b2
Step 13/20 : RUN ./bin/protoc object_detection/protos/*.proto --python_out=.
 ---> Using cache
 ---> 3b0c9a7ed422
Step 14/20 : RUN apt-get install nano
 ---> Using cache
 ---> 105f9faaea83
Step 15/20 : WORKDIR /root/src/
 ---> Using cache
 ---> fdf91a1fc11e
Step 16/20 : COPY requirements.txt ./
 ---> Using cache
 ---> 1f64cd7af75e
Step 17/20 : RUN pip install -r requirements.txt
 ---> Using cache
 ---> 3a2f60abdb6c
Step 18/20 : RUN python /root/src/tf/models/research/object_detection/builders/model_builder_test.py
 ---> Using cache
 ---> 4c30f01f0b4d
Step 19/20 : COPY . .
 ---> 2405a1bdb83d
Step 20/20 : CMD [ "/bin/bash", "" ]
 ---> Running in 7efa95b1c83a
Removing intermediate container 7efa95b1c83a
 ---> 36fbb928eecb
Successfully built 36fbb928eecb
Successfully tagged jan23-02:latest
[napiersk@vista23 clone_codes]$ docker run -itd --name d5 jan23-02 /bin/bash
0a3840309a2ebff783fe47b2487adc086d6bf7eef765cd48bcbc5d7c24fcb1e0
[napiersk@vista23 clone_codes]$ docker exec -it d5 /bin/bash
root@0a3840309a2e:~/src# cd ./tfobjdetect/script/
root@0a3840309a2e:~/src/tfobjdetect/script# python ./deploy_037a.py 
2020-01-23 21:40:40.320998: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-01-23 21:41:40.040455: W tensorflow/core/framework/allocator.cc:122] Allocation of 377318400 exceeds 10% of system memory.
2020-01-23 21:41:40.304104: W tensorflow/core/framework/allocator.cc:122] Allocation of 377318400 exceeds 10% of system memory.
2020-01-23 21:41:42.398052: W tensorflow/core/framework/allocator.cc:122] Allocation of 159744000 exceeds 10% of system memory.
2020-01-23 21:41:42.725291: W tensorflow/core/framework/allocator.cc:122] Allocation of 159744000 exceeds 10% of system memory.
2020-01-23 21:41:43.296744: W tensorflow/core/framework/allocator.cc:122] Allocation of 159744000 exceeds 10% of system memory.
0 images processed out of 10. 
root@0a3840309a2e:~/src/tfobjdetect/script# find ~/ -name 'det_results_m18_jpg_oi_1_filtered.pkl'
/root/src/results/det_results_m18_jpg_oi_1_filtered.pkl
root@0a3840309a2e:~/src/tfobjdetect/script# ls -al /root/src/results/
total 188
drwxrwxr-x 1 root root     99 Jan 23 21:44 .
drwxr-xr-x 1 root root     67 Jan 23 21:37 ..
-rw-r--r-- 1 root root 130290 Jan 23 21:44 det_results_m18_jpg_oi_1.pkl
-rw-r--r-- 1 root root  57652 Jan 23 21:44 det_results_m18_jpg_oi_1_filtered.pkl
-rw-rw-r-- 1 root root      0 Jan 23 20:23 empty.txt
root@0a3840309a2e:~/src/tfobjdetect/script# pwd
/root/src/tfobjdetect/script
root@0a3840309a2e:~/src/tfobjdetect/script# exit

```
