mention extraction: /nas/data/m1/liny9/aida/api/v1/
neeed requests, torch, flashtext, allennlp
```
conda create -n aida_entity python=3.6
conda install pytorch=1.3
-----conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install allennlp
pip install request
pip install flashtext

python app.py
```
request的格式
{
    "doc_id": "XXXXXXX",
    "ltf": "<ltf file content>",
    "rsd": "<rsd file content>",
    "lang": "<en|ru|uk>"
}


