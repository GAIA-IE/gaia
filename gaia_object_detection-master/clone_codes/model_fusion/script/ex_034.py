
# coding: utf-8

# In[1]:


import sys
import os
import pickle
import csv
import lmdb
import json

from io import BytesIO
import numpy as np

from rdflib import URIRef
from rdflib.namespace import ClosedNamespace


# In[2]:


score_threshold = 0.0


# In[3]:


sys.path.append("/home/alireza/aida/tools/AIDA-Interchange-Format/python")
from aida_interchange.Bounding_Box import Bounding_Box
import aida_interchange.aifutils as aifutils


# In[4]:


with open('../../../wsod/metadata/ont_m18/mapping2.pkl', 'rb') as fin:
    mid2ont, syn2mid, single_mids, mid2syn, class2ont, ont2name, class_names = pickle.load(fin)  


# In[5]:


ldc_entity_types = []
with open('../../../wsod/metadata/ont_m18/ldc_entity_types.txt', 'r') as fin:
    for line in fin:
        ldc_entity_types.append(line.strip())


# In[6]:


len(ldc_entity_types)


# In[7]:


ldc_event_types = []
with open('../../../wsod/metadata/ont_m18/ldc_event_types.txt', 'r') as fin:
    for line in fin:
        ldc_event_types.append(line.strip())


# In[8]:


len(ldc_event_types)


# In[9]:


short_to_long_map = {
    'BAL': 'Ballot',
    'COM': 'Commodity',
    'CRM': 'Crime',
    'FAC': 'Facility',
    'GPE': 'GeopoliticalEntity',
    'LAW': 'Law',
    'LOC': 'Location',
    'MON': 'Money',
    'ORG': 'Organization',
    'PER': 'Person',
    'RES': 'Result',
    'SID': 'Sides',
    'TTL': 'Title',
    'VAL': 'Value',
    'VEH': 'Vehicle',
    'WEA': 'Weapon',
}
long_to_short_map = {v: k for k, v in short_to_long_map.items()}


# In[10]:


allowed_to_have_name = ['PER', 'ORG', 'GPE', 'FAC', 'LOC', 'WEA', 'VEH', 'LAW']


# In[11]:


ldc_entity_types_new = []
for t in ldc_entity_types:
    sp = t.split('.')
    if sp[0] not in long_to_short_map:
        print(t)
    else:
        sp[0] = long_to_short_map[sp[0]]
    ldc_entity_types_new.append('.'.join(sp))


# In[12]:


LDC_ONTOLOGY = ClosedNamespace(
    uri=URIRef("https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LDCOntology#"),
    terms=ldc_entity_types_new + ldc_event_types
)


# In[13]:


with open('../../results/det_results_merged_34a.pkl', 'rb') as fin:
    det_results_jpg = pickle.load(fin)

with open('../../results/det_results_merged_34b.pkl', 'rb') as fin:
    det_results_vid = pickle.load(fin)
    


# In[14]:


with open('../../temp/imgsize_m18_jpg.pkl', 'rb') as fin:
    image_shape = pickle.load(fin)
with open('../../temp/imgsize_m18_kf.pkl', 'rb') as fin:
    image_shape.update(pickle.load(fin))


# In[15]:


root_to_leaf = {}
flag = False
with open('../../../../data/eval_m18/parent_children.sorted.tab', 'r') as fin:
    for line in fin:
        if not flag:
            flag = True
            continue
        row = line.split()
        if root_to_leaf.get(row[7]) == None:
            root_to_leaf[row[7]] = []
        root_to_leaf[row[7]].append(row[2])


# In[16]:


def add_detections_to_graph(g, detections, parent_id, imgid, is_keyframe):
    
    str_append = 'Keyframe' if is_keyframe else 'JPG'   
    if is_keyframe:
        vidid = imgid.split('_')[0]
    
    for ii, det in enumerate(detections):
        label = det['label']
        score = det['score']
        bbox = det['bbox']
        model = det['model']

        if score < score_threshold:
            continue

        if model == 'coco':
            sys = system_co
        elif model == 'voc':
            sys = system_pa
        elif model == 'oi':
            sys = system_oi
        elif model == 'ws':
            sys = system_ws
        elif model == 'coco/J':
            sys = system_co
        elif model == 'voc/J':
            sys = system_pa
        elif model == 'oi/J':
            sys = system_oi
        elif model == 'ws/J':
            sys = system_ws
        else:
            raise

        for iii, ont_id in enumerate(class2ont[label]):
            ont_name = ont2name[ont_id]

            labelrdf = LDC_ONTOLOGY.term(ont_name)

            if ont_name in ldc_entity_types_new:

                eid = f"http://www.columbia.edu/AIDA/DVMM/Entities/ObjectDetection/RUN00010/{str_append}/{imgid}/{ii}"

                entity = aifutils.make_entity(g, eid, sys)

                entity_dict[eid] = entity

                type_assertion = aifutils.mark_type(g, eid.replace('Entities', 'TypeAssertions') + f'/{iii}', entity, labelrdf, sys, score)

                bb = Bounding_Box((bbox[0], bbox[1]), (bbox[2], bbox[3]))

                if is_keyframe:
                    justif = aifutils.mark_keyframe_video_justification(g, [entity, type_assertion], vidid, imgid, bb, sys, score)
                else:
                    justif = aifutils.mark_image_justification(g, [entity, type_assertion], imgid, bb, sys, score)

                aifutils.add_source_document_to_justification(g, justif, parent_id)
                aifutils.mark_informative_justification(g, entity, justif)
                #aifutils.mark_private_data(g, entity, json.dumps({}), sys)
                #if ont_name.split('.')[0] in allowed_to_have_name:
                #    aifutils.mark_name(g, entity, class_names[label].split('(')[-1][:-1])

            else:            

                eid = f"http://www.columbia.edu/AIDA/DVMM/Events/ObjectDetection/RUN00010/{str_append}/{imgid}/{ii}"

                event = aifutils.make_event(g, eid, sys)

                event_dict[eid] = event

                type_assertion = aifutils.mark_type(g, eid.replace('Events', 'TypeAssertions') + f'/{iii}', event, labelrdf, sys, score)

                bb = Bounding_Box((1, 1), image_shape[imgid])

                if is_keyframe:
                    justif = aifutils.mark_keyframe_video_justification(g, [event, type_assertion], vidid, imgid, bb, sys, score)
                else:
                    justif = aifutils.mark_image_justification(g, [event, type_assertion], imgid, bb, sys, score)

                aifutils.add_source_document_to_justification(g, justif, parent_id)
                aifutils.mark_informative_justification(g, event, justif)
                    
                #aifutils.mark_private_data(g, event, json.dumps({}), sys)
                #aifutils.mark_name(g, event, class_names[label])


# In[17]:


id_set_jpg = set([item for item in det_results_jpg])
id_set_vid = set()
for imgid in det_results_vid:
    vidid = imgid.split('_')[0]
    id_set_vid.add(vidid)


# In[18]:


kb_dict = {}
entity_dict = {}
event_dict = {}

for root_doc in root_to_leaf:

    g = aifutils.make_graph()

    system_pa = aifutils.make_system_with_uri(g, "http://www.columbia.edu/AIDA/USC/Systems/ObjectDetection/FasterRCNN/PascalVOC")
    system_co = aifutils.make_system_with_uri(g, "http://www.columbia.edu/AIDA/DVMM/Systems/ObjectDetection/FasterRCNN-NASNet/COCO")
    system_oi = aifutils.make_system_with_uri(g, "http://www.columbia.edu/AIDA/DVMM/Systems/ObjectDetection/FasterRCNN-InceptionResNet/OpenImages")
    system_ws = aifutils.make_system_with_uri(g, "http://www.columbia.edu/AIDA/DVMM/Systems/ObjectDetection/MITWeaklySupervised-ResNet/OpenImages")
        
    for imgid in id_set_jpg & set(root_to_leaf[root_doc]):
        add_detections_to_graph(g, det_results_jpg[imgid], root_doc, imgid, is_keyframe=False)
        
    for imgid in det_results_vid:
        vidid = imgid.split('_')[0]
        if vidid in root_to_leaf[root_doc]:
            add_detections_to_graph(g, det_results_vid[imgid], root_doc, imgid, is_keyframe=True)        

    kb_dict[root_doc] = g


# In[19]:


with open('../../results/aida_output_34.pkl', 'wb') as fout:
    pickle.dump((kb_dict, entity_dict, event_dict), fout)


# In[19]:


export_dir = '../../results/aida_output_34'
if not os.path.isdir(export_dir):
    os.makedirs(export_dir)
for root_doc, g in kb_dict.items():
    with open(os.path.join(export_dir, root_doc+'.ttl'), 'w') as fout:
        serialization = BytesIO()
        g.serialize(destination=serialization, format='turtle')
        fout.write(serialization.getvalue().decode('utf-8'))

