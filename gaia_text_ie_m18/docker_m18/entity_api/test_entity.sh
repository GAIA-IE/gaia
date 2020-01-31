#!/usr/bin/env bash

docker run -v `pwd`:`pwd` -w `pwd` -i -t -p 5500:5500 limanling/aida_entity \
    /opt/conda/envs/aida_entity/bin/python \
    entity_api/app.py

docker run -v `pwd`:`pwd` -w `pwd` -i -t -p 5500:5500 limanling/aida_entity \
    /opt/conda/envs/aida_entity/bin/python \
