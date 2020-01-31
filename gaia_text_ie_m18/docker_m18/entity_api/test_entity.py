import requests
import traceback
import json
import os


en_ltf_file = os.path.join(os.path.dirname(__file__), './testdata/en/IC001L4PK.ltf.xml')
en_rsd_file = os.path.join(os.path.dirname(__file__), './testdata/en/IC001L4PK.rsd.txt')
doc_id = 'IC001L4PK'
lang = 'en'
# ru_ltf_file = os.path.join(os.path.dirname(__file__), '../testdata/ru/IC001L4Q0.ltf.xml')
# ru_rsd_file = os.path.join(os.path.dirname(__file__), '../testdata/ru/IC001L4Q0.rsd.txt')
# uk_ltf_file = os.path.join(os.path.dirname(__file__), '../testdata/uk/IC001L4QB.ltf.xml')
# uk_rsd_file = os.path.join(os.path.dirname(__file__), '../testdata/uk/IC001L4QB.rsd.txt')

en_ltf_str = open(en_ltf_file, 'r').read()
en_rsd_str = open(en_rsd_file, 'r').read()
# ru_ltf_str = open(ru_ltf_file, 'r').read()
# ru_rsd_str = open(ru_rsd_file, 'r').read()
# uk_ltf_str = open(uk_ltf_file, 'r').read()
# uk_rsd_str = open(uk_rsd_file, 'r').read()

# en_json_file = os.path.join(os.path.dirname(__file__), '../testdata/en/en.sample.json')
# en_json_str = open(en_json_file, 'r').read()

ans = requests.post(
    'http://127.0.0.1:5500/tagging',

    data={
        'ltf': en_ltf_str,
        'rsd': en_rsd_str,
        'doc_id': doc_id,
        'lang': lang
    }
)

ans = json.loads(ans.text)
print('bio')
print(ans['bio'])
print('tab')
print(ans['tab'])
print('tsv')
print(ans['tsv'])


