FROM python:3.6.3

MAINTAINER Dan Napierski (ISI) <dan.napierski@toptal.com>

# Create app directory
WORKDIR /root/src/

# Install app dependencies
RUN apt-get upgrade && apt-get update && apt-get -y install apt-utils unzip git python-pil python-lxml python-tk
RUN pip install --upgrade pip

RUN mkdir tf
WORKDIR /root/src/tf
RUN git clone https://github.com/tensorflow/models.git
ENV PYTHONPATH=/usr/local/bin/python:/root/src/tf/models/research:/root/src/tf/models/research/slim:.

WORKDIR /root/src/tf/models/research
RUN wget -O protobuf.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.

RUN apt-get install nano

WORKDIR /root/src/
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Confirm Tensorflow Object Detection API installation
RUN python /root/src/tf/models/research/object_detection/builders/model_builder_test.py

# Bundle app source
COPY . .

CMD [ "/bin/bash", "" ]
