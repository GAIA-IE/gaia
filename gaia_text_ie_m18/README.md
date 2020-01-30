# UIUC IE Pipeline
One single script to run text information extraction, including entity extraction, relation extraction and event extraction.

## Prerequisites
### Packages to install
1. Docker
Please do not set up UIUC IE Pipeline in a NAS, as the EDL needs MongoDB, which may lead to permission issues in a NAS.

### Download the latest docker images
Docker images will work as services (`mongo`, `panx27/edl`, `limanling/uiuc_ie_m18`， `charlesztt/aida_event`,  `dylandilu/event_coreference_xdoc`, and `wangqy96/aida_nominal_coreference_en`).
```bash
docker pull mongo
docker pull panx27/edl
docker pull limanling/uiuc_ie_m18
docker pull charlesztt/aida_event
docker pull dylandilu/event_coreference_xdoc
docker pull wangqy96/aida_nominal_coreference_en
docker pull frnkenstien/corenlp
```

### Download the latest models
Please download the models for EDL, and event extraction.
```
wget http://159.89.180.81/demo/resources/edl_data.tar.gz
tar zxvf edl_data.tar.gz
```
For event extraction models
```
cd ./aida_event
wget http://159.89.180.81/demo/resources/aida_event_data.tgz
tar -xzf aida_event_data.tgz
cd ..
```

## Deployment
Please ensure that you are under the root folder of this project, and after each of the following dockers (step 1~5) is started, please open a new terminal to continue with another docker (of course, under the same root folder).

Also please reserve the the following ports and ensure that no other programs/services are occupying these ports: `27017`, `2468`, `5500`, `5000`, `5234`, `9000`, `6001`, `6101` and `6201`.

Step 1. Start the EDL mongo database server

Please wait until you see "waiting for connections on port 27017" message appear on the screen.

```bash
docker run --rm -d -v ${PWD}/aida_pipeline_m18/edl_data/db:/data/db --name db mongo
```

Step 2. Start the nominal coreference server
```bash
docker run -i -t --rm -w /aida_nominal_coreference_en -p 2468:2468 wangqy96/aida_nominal_coreference_en python nominal_backend.py
```

Step 3. Start the name tagger
```bash
docker run -i -t --rm  -w /entity_api -p 5500:5500 limanling/uiuc_ie_m18 \
    /opt/conda/envs/aida_entity/bin/python \
    /entity_api/entity_api/app.py
```

Step 4. Start the event extractor

This step will take a few minutes, you can proceed after you see "Serving Flask app ..." message.
```bash
docker run -i -t --rm -v ${PWD}/aida_pipeline_m18/aida_event/aida_event_data:/tmp -w /aida_event -p 5234:5234 charlesztt/aida_event python gail_event.py
```

Step 5. Start the event coreference solution

This step will take a few minutes, you can proceed after you see "Serving Flask app "aida_event_coreference_backen_{eng, rus, ukr}"" message. Notice that the port 6001, 6101 and 6201 are for English, Russian and Ukrainian respectively.
```bash
docker run -i -t --rm -w /event_coreference_xdoc -p 6001:6001 dylandilu/event_coreference_xdoc python aida_event_coreference_backen_eng.py
```

Step 6. Prepare Stanford CoreNLP

Please start the CoreNLP Server under the CoreNLP folder.
```bash
docker run -p 9000:9000 --name coreNLP --rm -i -t frnkenstien/corenlp
```
Please test the CoreNLP Server is running successfully:
```bash
wget --post-data 'The quick brown fox jumped over the lazy dog.' 'localhost:9000/?properties={"annotators":"tokenize,ssplit,pos,lemma,ner,regexner,depparse,entitymentions","outputFormat":"json"}'
```

## Run the codes
* Make sure you have RSD (Raw Source Data, ending with `*.rsd.txt`) and LTF (Logical Text Format, ending with `*.ltf.xml`) files. 
	* If you have RSD files, please use the `aida_utilities/rsd2ltf.py` to generate the LTF files. 
	* If you have LTF files, please use the AIDA ltf2rsd tool (`LDC2018E62_AIDA_Month_9_Pilot_Eval_Corpus_V1.0/tools/ltf2txt/ltf2rsd.perl`) to generate the RSD files. 
* Edit the `pipeline_sample.sh` for your run, including `data_root` containing a subfolder `ltf` with your input LTF files and a subfolder `rsd` with your input RSD files. Then run the shell file. For example.

```bash
sh pipeline_sample.sh ${data_root}
```

