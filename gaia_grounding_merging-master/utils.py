import re
import json
import os
import tensorflow as tf
from skimage.feature import peak_local_max
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from rdflib import RDF, URIRef
from rdflib.namespace import SKOS
from sklearn.cluster import DBSCAN

# nist_ont_pref = '.../SM-KBP/2018/ontologies/InterchangeOntology#'
nist_ont_pref = 'https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#'
justified_by_ = URIRef(nist_ont_pref+'justifiedBy')
entity_ = URIRef(nist_ont_pref+'Entity')
sys_ = URIRef(nist_ont_pref+'system')
hasName_ = URIRef(nist_ont_pref+'hasName')

#bbox generation config
rel_peak_thr = .3
rel_rel_thr = .3
ioa_thr = .6
topk_boxes = 3

#takes txt_id+offset+path and returns sentence that covers that offset in that txt_id
def offset2text(file_id, offset, path_dict):
    file_type = file_id.split('.')[1]
    if file_type == 'ltf':
        path = path_dict[file_id]
    elif file_type == 'mp4':
        path = path_dict[file_id]['asr']
    else:
        return []
    if not os.path.isfile(path):
        return []
    with open(path, 'r') as f:
        lines = f.readlines()
    text=[]
    offsets = list(map(int,offset.split('-')))

    for i,line in enumerate(lines):
        line = line.strip('\t')
        if line.strip('\n').find('<SEG ')==0:
            seg_data = line.strip('>\n').strip('<').split(' ')
            for entry in seg_data:
                if entry.find('start_char=')==0:
                    begin = int(entry[len('start_char='):].strip('"'))
                elif entry.find('end_char=')==0:
                    end = int(entry[len('end_char='):].strip('"'))
            if offsets[0]>=begin and offsets[1]<=end:
                splits = re.split(r'(<ORIGINAL_TEXT>|</ORIGINAL_TEXT>\n)', lines[i+1])
                if len(splits)>1:
                    text=splits[2]
                    break
    return text

#generating child and parent dicts
def create_dict(tab_path):
    with open(tab_path,'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
    parent_dict = {}
    child_dict = {}
    for line in lines[1:]:
        parent_id = line[7]
        child_id = line[2]+line[1]

        ##updating parent_dict
        if parent_id in parent_dict:
            parent_dict[parent_id].update([child_id])
        else:
            parent_dict[parent_id] = set([child_id])
            
        ##updating child_dict
        if child_id not in child_dict:
            child_dict[child_id] = [parent_id]
        else:
            child_dict[child_id].append(parent_id)
            
    return parent_dict, child_dict

#create entity dict from RPI result in turtle (AIF) format
def create_entity_dict(e2m_dict, path_dict, caption_alignment_path=[], filter_out=[]):
    if caption_alignment_path!=[]:
        with open(caption_alignment_path, 'r') as f:
            lines = f.readlines()
            lines_alignment = [line.strip('\n').split('\t') for line in lines]
            #caption  <img_id>  <txt_id>  <caption_offsets>
            #neighbor  <img_id>  <txt_id>  <ceiling_neighbor_offsets>  <floor_neighbor_offsets>
        
    id2mentions_dict = {}
    for entity_id in e2m_dict:
        entity_name = e2m_dict[entity_id]['name']
        en_sys_rdf = e2m_dict[entity_id]['sys_rdf']
        en_type_rdf = e2m_dict[entity_id]['type_rdf']
        entity_type = en_type_rdf.toPython().split('#')[1].split('.')[0]
        
        for mention_id in e2m_dict[entity_id]['mentions']:
            mention_name = e2m_dict[entity_id]['mentions'][mention_id]['name']
            mention_type = e2m_dict[entity_id]['mentions'][mention_id]['mention_type']
            source_type = e2m_dict[entity_id]['mentions'][mention_id]['source_type']
            language = e2m_dict[entity_id]['mentions'][mention_id]['language']
            translation = e2m_dict[entity_id]['mentions'][mention_id]['translation']
            if source_type == 'JPG':
                file_ext = '.ltf.xml'
            elif source_type == 'Keyframe':
                file_ext = '.mp4.ldcc'
            else:
                continue
            file_id = e2m_dict[entity_id]['mentions'][mention_id]['source']+file_ext
            offset = e2m_dict[entity_id]['mentions'][mention_id]['offset']
            
            if file_id not in path_dict or entity_type in filter_out or mention_type in filter_out:
#                 print('Filtered','file_id:',file_id,'entity_type:',entity_type,'mention_type',mention_type)
                continue
            
            if language!='English' and (translation == None or len(translation)==0):
                continue
                
            if file_id not in id2mentions_dict:
                id2mentions_dict[file_id] = {}
            
            if language=='English':
                sentence = offset2text(file_id, offset, path_dict)
            else:
                mention_name = ' '.join(translation)
                sentence = mention_name+' in the image.'
           
            if sentence==[]:
                print(file_id)
                print(offset)
                continue
            strip_chars = ['.',':',';',',','"','[',']','(',')','{','}','?','!',"'"]
            for strip_char in strip_chars:
                sentence = sentence.replace(strip_char,'')
                mention_name = mention_name.replace(strip_char,'')

            token_list = sentence.split()
            query_list = mention_name.split()

            if len(token_list)<3: #short sentences are ignored
                continue
            #for item in query_list: #debugging
            #    if item not in token_list:
            #        print(entity_id)
            #        print(item)
            #        print(sentence)


            idx_in_sen = [token_list.index(item) for item in query_list if item in token_list]
            
            tmp = {'name': mention_name,
                   'idx': idx_in_sen,
                   'mention_type': mention_type,
                   'source_type': source_type}
            
            if sentence not in id2mentions_dict[file_id]:
                id2mentions_dict[file_id].update({sentence:\
                    {entity_id: {'name' : entity_name,
                                 'type_rdf': en_type_rdf,
                                 'sys_rdf': en_sys_rdf,
                                 'source_type': source_type,
                                 'language': language,
                                 'mentions': {mention_id: tmp}}}})
            
            if entity_id not in id2mentions_dict[file_id][sentence]:
                id2mentions_dict[file_id][sentence].update(\
                    {entity_id: {'name' : entity_name,
                                 'type_rdf': en_type_rdf,
                                 'sys_rdf': en_sys_rdf,
                                 'source_type': source_type,
                                 'language': language,
                                 'mentions': {mention_id: tmp}}})
            
            if mention_id not in id2mentions_dict[file_id][sentence][entity_id]['mentions']:
                id2mentions_dict[file_id][sentence][entity_id]['mentions'].update({mention_id: tmp})

            #to do: add caption/neighbor flag to d2mentions_dict[txt_id][sentence]
        
    return id2mentions_dict

def get_entity2mention(graph,ltf_util):
    '''
    A function that gets graph and loads information in it.
    '''
    #get data and put in entity2mention dictionary
    entity2mention = {}
    mention2entity = {}
    
    entities = graph.subjects(predicate=RDF.type,object=entity_)

    for entity in entities:
        entity_hasName = list(graph.objects(predicate=hasName_,subject=entity))
        if len(entity_hasName)!=0:
            entity_name = entity_hasName[0].toPython()
        else:
            entity_name = 'no_name'

        entity_id = entity.toPython()
        en_sys_rdf = list(graph.objects(predicate=sys_,subject=entity))[0]
        en_asser_node = list(graph.subjects(predicate=RDF.subject,object=entity))[0]
        en_type_rdf = list(graph.objects(subject = en_asser_node,predicate=RDF.object))[0]
        
        entity2mention[entity_id] = {'mentions': {},
                                     'name': entity_name,
                                     'type_rdf': en_type_rdf,
                                     'sys_rdf': en_sys_rdf}
        
        
        just_by = graph.objects(predicate=justified_by_,subject=entity)        
        for just in just_by:
            mention_id = just.toPython()
            mention2entity[mention_id] = entity_id

            off_beg = list(graph.objects(subject=just,
                            predicate=URIRef(nist_ont_pref+'startOffset')))[0].toPython()
            off_end = list(graph.objects(subject=just,
                            predicate=URIRef(nist_ont_pref+'endOffsetInclusive')))[0].toPython()
            source = list(graph.objects(subject=just,
                            predicate=URIRef(nist_ont_pref+'source')))[0].toPython()
            pv_data_rdf = list(graph.objects(subject=just,
                            predicate=URIRef(nist_ont_pref+'privateData')))
            
           # Check error from ISI graph, 
#             print('entity_id',entity_id)          
#           print('graph')
#           for s, p, o in graph:
#                 print(s, p, o)

#             print(list(graph.objects(subject=just,
#                             predicate=SKOS.prefLabel)))  # Done: changed mention loading
            
            
                
            # Check Point: text mention from RPI 
            # Done: Check error for prefLabel missing 
            mention_name = None
            prefLabels = list(graph.objects(subject=just, predicate=SKOS.prefLabel))
            if len(prefLabels) == 0:
#                 print('prefLabel missing mention_id:',mention_id,', entity_id:', entity_id)
#                 print('mention_name:',mention_name,'\n')
                just_str = source + ':' + str(off_beg) + '-' + str(off_end) # 'HC000Q7NP:167-285'
                mention_name = ltf_util.get_str(just_str)
#                 continue
            else:
                mention_name = prefLabels[0].toPython()
            if mention_name == None:
#                 print('mention_name missing:', mention_name, ', entity_id:',entity_id)
                continue
                
#                 print('prefLabel',list(graph.objects(subject=just,
#                             predicate=SKOS.prefLabel))[0].toPython())
#                 print('mention_name',mention_name)
                
               
#             role_justi = g.value(subject=event_state, predicate=p_justi) # from Manling
            
            
            
            mention_type,f_t = None,None
            tr = None
            for pv_rdf in pv_data_rdf:
                dict_str=list(graph.objects(subject=pv_rdf,
                                predicate=URIRef(nist_ont_pref+'jsonContent')))[0].toPython()

                if 'justificationType' in json.loads(dict_str):
                    mention_type = json.loads(dict_str)['justificationType']
                    
                if 'fileType' in json.loads(dict_str):
                    f_t = json.loads(dict_str)['fileType']

                if 'translation' in json.loads(dict_str):
                    tr = json.loads(dict_str)['translation']
            
            
            # Done: type checking
#             if ( mention_name in ['27','3']):
#                 print('mention_type checking...', 'mention_name:',mention_name, 'en_type_rdf:',en_type_rdf,'mention_type:',mention_type, 'entity_id:',entity_id)
            
            #Done: missing checking
            flag_continue = False
            if mention_type == None:
            # justificationType for mention_type
#                 print('justificationType missing:', dict_str, ', entity_id:', entity_id)
                flag_continue = True
            if f_t == None:
            #  fileType for language
#                 print('fileType missing', dict_str,  'entity_id:',entity_id)
                flag_continue = True
            if flag_continue:
                continue
    
    
            elif f_t=='en_asr':
                source_type = 'Keyframe'
                language = 'English'
            elif f_t=='en_ocr':
                source_type = 'OCR'
                language = 'English'
            elif f_t=='en':
                source_type = 'JPG'
                language = 'English'
            elif f_t=='uk':
                source_type = 'JPG'
                language = 'Ukrainian'
            elif f_t=='ru':
                source_type = 'JPG'
                language = 'Russian'
            else:
                continue
            
            entity2mention[entity_id]['mentions'].update({mention_id: {'source': source,
                                                                       'offset': str(off_beg)+'-'+str(off_end),
                                                                       'name': mention_name,
                                                                       'translation': tr,
                                                                       'mention_type': mention_type,
                                                                       'source_type': source_type,
                                                                       'language': language}})
    return entity2mention


def load_text(parent_path,ltf_id):
    with open(os.path.join(parent_path,ltf_id), 'r') as f:
        lines = f.readlines()
    text=[]
    for line in lines:
        splits = re.split(r'(<ORIGINAL_TEXT>|</ORIGINAL_TEXT>\n)', line)
        if len(splits)>1:
            text.append(splits[2])
    return text

def heat2bbox(heat_map, original_image_shape):
    
    h, w = heat_map.shape
    
    bounding_boxes = []

    heat_map = heat_map - np.min(heat_map)
    heat_map = heat_map / np.max(heat_map)

    bboxes = []
    box_scores = []

    peak_coords = peak_local_max(heat_map, exclude_border=False, threshold_rel=rel_peak_thr) # find local peaks of heat map

    heat_resized = cv2.resize(heat_map, (original_image_shape[1],original_image_shape[0]))  ## resize heat map to original image shape
    peak_coords_resized = ((peak_coords + 0.5) * 
                           np.asarray([original_image_shape]) / 
                           np.asarray([[h, w]])
                          ).astype('int32')

    for pk_coord in peak_coords_resized:
        pk_value = heat_resized[tuple(pk_coord)]
        mask = heat_resized > pk_value * rel_rel_thr
        labeled, n = ndi.label(mask) 
        l = labeled[tuple(pk_coord)]
        yy, xx = np.where(labeled == l)
        min_x = np.min(xx)
        min_y = np.min(yy)
        max_x = np.max(xx)
        max_y = np.max(yy)
        bboxes.append((min_x, min_y, max_x, max_y))
        box_scores.append(pk_value) # you can change to pk_value * probability of sentence matching image or etc.


    ## Merging boxes that overlap too much
    box_idx = np.argsort(-np.asarray(box_scores))
    box_idx = box_idx[:min(topk_boxes, len(box_scores))]
    bboxes = [bboxes[i] for i in box_idx]
    box_scores = [box_scores[i] for i in box_idx]

    to_remove = []
    for iii in range(len(bboxes)):
        for iiii in range(iii):
            if iiii in to_remove:
                continue
            b1 = bboxes[iii]
            b2 = bboxes[iiii]
            isec = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
            ioa1 = isec / ((b1[2] - b1[0]) * (b1[3] - b1[1]))
            ioa2 = isec / ((b2[2] - b2[0]) * (b2[3] - b2[1]))
            if ioa1 > ioa_thr and ioa1 == ioa2:
                to_remove.append(iii)
            elif ioa1 > ioa_thr and ioa1 >= ioa2:
                to_remove.append(iii)
            elif ioa2 > ioa_thr and ioa2 >= ioa1:
                to_remove.append(iiii)

    for i in range(len(bboxes)): 
        if i not in to_remove:
            bounding_boxes.append({
                'score': box_scores[i],
                'bbox': bboxes[i],
                'bbox_normalized': np.asarray([
                    bboxes[i][0] / heat_resized.shape[1],
                    bboxes[i][1] / heat_resized.shape[0],
                    bboxes[i][2] / heat_resized.shape[1],
                    bboxes[i][3] / heat_resized.shape[0],
                ]),
            })
    
    return bounding_boxes

def img_heat_bbox_disp(image, heat_map, title='', en_name='', alpha=0.6, cmap='viridis', cbar='False', dot_max=False, bboxes=[], order=None, show=True):
    thr_hit = 1 #a bbox is acceptable if hit point is in middle 85% of bbox area
    thr_fit = .60 #the biggest acceptable bbox should not exceed 60% of the image
    H, W = image.shape[0:2]
    # resize heat map
    heat_map_resized = cv2.resize(heat_map, (H, W))

    # display
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, size=15)
    ax = plt.subplot(1,3,1)
    plt.imshow(image)
    if dot_max:
        max_loc = np.unravel_index(np.argmax(heat_map_resized, axis=None), heat_map_resized.shape)
        plt.scatter(x=max_loc[1], y=max_loc[0], edgecolor='w', linewidth=3)
    
    if len(bboxes)>0: #it gets normalized bbox
        if order==None:
            order='xxyy'
        
        for i in range(len(bboxes)):
            bbox_norm = bboxes[i]
            if order=='xxyy':
                x_min,x_max,y_min,y_max = int(bbox_norm[0]*W),int(bbox_norm[1]*W),int(bbox_norm[2]*H),int(bbox_norm[3]*H)
            elif order=='xyxy':
                x_min,x_max,y_min,y_max = int(bbox_norm[0]*W),int(bbox_norm[2]*W),int(bbox_norm[1]*H),int(bbox_norm[3]*H)
            x_length,y_length = x_max-x_min,y_max-y_min
            box = plt.Rectangle((x_min,y_min),x_length,y_length, edgecolor='w', linewidth=3, fill=False)
            plt.gca().add_patch(box)
            if en_name!='':
                ax.text(x_min+.5*x_length,y_min+10, en_name,
                verticalalignment='center', horizontalalignment='center',
                #transform=ax.transAxes,
                color='white', fontsize=15)
                #an = ax.annotate(en_name, xy=(x_min,y_min), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
                #plt.gca().add_patch(an)
            
    plt.imshow(heat_map_resized, alpha=alpha, cmap=cmap)
    
    #plt.figure(2, figsize=(6, 6))
    plt.subplot(1,3,2)
    plt.imshow(image)
    #plt.figure(3, figsize=(6, 6))
    plt.subplot(1,3,3)
    plt.imshow(heat_map_resized)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def filter_bbox(bbox_dict,order=None):
    thr_fit = .80 #the biggest acceptable bbox should not exceed 80% of the image
    if len(bbox_dict)>0: #it gets normalized bbox
        if order==None:
            order='xxyy'
        
        filtered_bbox = []
        filtered_bbox_norm = []
        filtered_score = []
        for i in range(len(bbox_dict)):
            bbox = bbox_dict[i]['bbox']
            bbox_norm = bbox_dict[i]['bbox_normalized']
            bbox_score = bbox_dict[i]['score']
            if order=='xxyy':
                x_min,x_max,y_min,y_max = bbox_norm[0],bbox_norm[1],bbox_norm[2],bbox_norm[3]
            elif order=='xyxy':
                x_min,x_max,y_min,y_max = bbox_norm[0],bbox_norm[2],bbox_norm[1],bbox_norm[3]
            if bbox_score>0:
                x_length,y_length = x_max-x_min,y_max-y_min
                if x_length*y_length<thr_fit:
                    filtered_score.append(bbox_score)
                    filtered_bbox.append(bbox)
                    filtered_bbox_norm.append(bbox_norm)
    return filtered_bbox, filtered_bbox_norm, filtered_score

def img_cap_batch_gen(imgs,sens,ids,key,path_dict,id2time_dict):
    if len(imgs)==0:
        return np.array([]),sens,[['no_image',()]]*len(sens)
    
    if key.split('.')[1]=='mp4':
        dtype = 'Keyframe'
        doc2time_dict = doc2time(key,path_dict)
    else:
        dtype = 'JPG'
    
    sen_batch = []
    img_batch = []
    img_info_batch = []
    for sen in sens:
        if dtype=='Keyframe':
            img_ids = sen2kfrm(sen,key,doc2time_dict,id2time_dict)
        elif dtype=='JPG':
            img_ids = ids
            
        for i,img_id in enumerate(img_ids):
            if dtype=='Keyframe':
                num = int(img_id.split('_')[1])
                img = imgs[num-1]
                ftype = '.mp4.ldcc'
            elif dtype=='JPG':
                img = imgs[i]
                ftype = '' #it already contains .jpg.ldcc
            sen_batch.append(sen)
            img_batch.append(cv2.resize(img,(299,299)))
            img_info_batch.append([img_id+ftype,(img.shape[0:2])])
    img_batch = np.array(img_batch)        
   
    return img_batch,sen_batch,img_info_batch

def pad_along_axis(array: np.ndarray, target_length, axis=0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array
    
    axis_nb = len(array.shape)
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b

def batch_split_run(sess,tensor_list,placeholders,inputs,text_flag,b_size_thr=100):
    img_batch,sen_batch = inputs
    if text_flag:
        L = len(sen_batch)
        pred_tensors = [tensor_list[0]]
    else:
        L = img_batch.shape[0]
        pred_tensors = tensor_list
    n_runs = int(L/b_size_thr)
    EN_embd, IMG_embd, EN_heat, EN_score, avg_EN_score, sen_score = ([],[],[],[],[],[])
    #split and calculate predictions
    if n_runs*b_size_thr==L:
        upper = n_runs
    else:
        upper = n_runs+1
    
    if upper==1:
        max_len = 0
        #doesn't need split, avoid tokenization, avoid padding
    else:
        max_lens = []
        for sen in sen_batch:
            max_lens.append(len(sen.split()))
        max_len = np.amax(max_lens)
        
    input_img, text_batch, mode = placeholders
    for n in range(upper):
        feed_dict = {text_batch: sen_batch[n*b_size_thr:(n+1)*b_size_thr], mode: 'test'}
        if not text_flag:
            feed_dict.update({input_img: img_batch[n*b_size_thr:(n+1)*b_size_thr,:]})
        preds = sess.run(pred_tensors, feed_dict)
        EN_embd.append(pad_along_axis(preds[0], max_len, axis=1))
        if not text_flag:
            IMG_embd.append(pad_along_axis(preds[1], max_len, axis=1))
            EN_heat.append(pad_along_axis(preds[2], max_len, axis=1))
            EN_score.append(pad_along_axis(preds[3], max_len, axis=1))
            avg_EN_score.append(preds[4])
            sen_score.append(preds[5])
    
    EN_embd = np.concatenate(EN_embd,axis=0)
    if not text_flag:
        IMG_embd = np.concatenate(IMG_embd,axis=0)
        EN_heat = np.concatenate(EN_heat,axis=0)
        EN_score = np.concatenate(EN_score,axis=0)
        avg_EN_score = np.concatenate(avg_EN_score,axis=0)
        sen_score = np.concatenate(sen_score,axis=0)
        
    return EN_embd, IMG_embd, EN_heat, EN_score, avg_EN_score, sen_score

#load jpg.ldcc from path
def ldcc_load(filename):
    with open(filename, 'rb') as fin:
        _ = fin.read(1024)
        imgbin = fin.read()
    imgbgr = cv2.imdecode(np.frombuffer(imgbin, dtype='uint8'), cv2.IMREAD_COLOR)
    if imgbgr is None:
        return None
    else:
        return imgbgr[:,:,[2,1,0]]

#if a child has multiple parents
def appnd_parents(parent_dict,child_dict,id_):
    childs = set()
    for parent_id in child_dict[id_]:
        childs.update(parent_dict[parent_id])
    return childs

#fetch image using child_id
def fetch_img(key, parent_dict, child_dict, path_dict, level='Child'):
    if 'mp4' in key and '_' in key:
        id_ = key.split('_')[0]+'.mp4.ldcc'
    else:
        id_ = key
    
    if id_ not in child_dict:
        print('id_ not in chlid_dict',id_)
        return [],[]
    #for now if input is mp4, only return its keyframes, not parent as well
    #if input is img_id_ii.mp4.ldcc, it only gives that frame
    elif 'mp4' in key and '_' in key:
        mp4_flag = False
        kfrm_flag = True
        n_kfrm = int(key.split('_')[1].split('.')[0])
        child_id_list = [id_]
    elif level=='Child' or key.find('mp4')!=-1:
        mp4_flag = True
        kfrm_flag = False
        child_id_list = [id_]
    elif level=='Parent':
        mp4_flag = False
        kfrm_flag = False
        child_id_list = appnd_parents(parent_dict,child_dict,id_)
    else:
        print('other')
        return [],[]
    
    imgs_in_parent = []
    ids_in_parent = []
    imgs_in_video = []
    ids_in_video = []
    key_error_num = 0
    for child_id in child_id_list:
        # Todo: fixed the bug of key missing for missing child_id
#             if (child_id=='HC000TJCP_36.mp4.ldcc'):
        if child_id not in path_dict.keys() and ('jpg' in child_id or 'mp4' in child_id ):
            print(child_id,'child_id not in path_dict:',child_id in path_dict.keys())
            key_error_num += 1
            continue
                
        if 'jpg' in child_id:
            filename = path_dict[child_id]
            img = ldcc_load(filename)
            if img is not None:
                imgs_in_parent.append(img)
                ids_in_parent.append(child_id)
        
        elif 'mp4' in child_id and kfrm_flag:
            filename = path_dict[child_id]['keyframe'][n_kfrm-1]
#             print('key',key,path_dict[child_id]['keyframe'])
#             print('filename',filename)
            if 'png.ldcc' in filename :
                img = ldcc_load(filename)
            else:
                img = cv2.imread(filename, cv2.IMREAD_COLOR)
            if img is not None:
                img = img[:,:,[2,1,0]]
                imgs_in_parent.append(img)
                ids_in_parent.append(child_id.split('.')[0]+'_'+str(n_kfrm)+'.mp4.ldcc')
         
        elif 'mp4' in child_id and mp4_flag:
            files = path_dict[child_id]['keyframe']
            for i,filename in enumerate(files):
                #note that path_dict should give a sorted list of keyframe paths
                if 'png.ldcc' in filename :
                    img = ldcc_load(filename)
                else:
                    img = cv2.imread(filename, cv2.IMREAD_COLOR)
                if img is not None:
                    img = img[:,:,[2,1,0]]
                    imgs_in_video.append(img)
                    ids_in_video.append(child_id.split('.')[0]+'_'+str(i+1)+'.mp4.ldcc')
            imgs_in_parent.extend(imgs_in_video)
            ids_in_parent.extend(ids_in_video)
    # Done: add the key missing counter
    if key_error_num>0:
        print('KeyError num',key_error_num)
    return imgs_in_parent, ids_in_parent

#create path dictionary from path
def create_path_dict(parent_path):
    files = os.listdir(parent_path)
    path_dict = {}
    for file in files:
        path_dict[file] = os.path.join(parent_path,file)
    return path_dict

#keyframe IDs to path
def create_dict_kfrm(keyframe_path, keyframe_msb, video_asr_path, video_map_path):
    with open(keyframe_msb,'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines] #split by whitespace
    id2dir_dict = {}
    dir2id_dict = {}
    for line in lines:
        dir_ = line[0]
        id_ = line[1].split('_')[0]
        if id_ not in id2dir_dict:
            id2dir_dict[id_] = {'dir': dir_}
        if dir_ not in dir2id_dict:
            dir2id_dict[dir_] = id_

    subdirs = [o for o in os.listdir(keyframe_path)]
    for dir_ in subdirs:
        if dir_.find('.')==0:
            continue
        id_ = dir2id_dict[dir_]
        id2dir_dict[id_].update({'keyframe_path': keyframe_path+'/'+dir_,
                                 'asr_path': video_asr_path+'/'+id_+'.ltf.xml',
                                 'map_path': video_map_path+'/'+id_+'.map'})
    return id2dir_dict

#create path dictionary from path for keyframes
def create_path_dict_kfrm(id2dir_dict_kfrm):
    path_dict = {}
    for key in id2dir_dict_kfrm:
        kfrm_dir = id2dir_dict_kfrm[key]['keyframe_path']
        asr_dir = id2dir_dict_kfrm[key]['asr_path']
        map_dir = id2dir_dict_kfrm[key]['map_path']
        files = os.listdir(kfrm_dir)
        #Done: fixed the bug of path index, listing file in files without order,skip the missing files ~70 key frames and ~80 png
#         paths = ['']*len(files)
        paths = [None] * len(files)
        for file in files:
            if file.find('png')==-1:
                continue
            num = int(file.split('.')[0].split('_')[2])
#             if num-1>len(paths)-1: # Test debug
#                 if num-1<0:
#                     print('aaa')
#                 else:
#                     print('file',file,'index',num-1,'paths_len',len(paths))
#                     continue
            #Done: fixed the bug of path index
            paths[num-1] = os.path.join(kfrm_dir,file)
#             paths.append(os.path.join(kfrm_dir,file))
        path_dict[key+'.mp4.ldcc'] = {'keyframe': paths, 'asr': asr_dir, 'map': map_dir}
    return path_dict

def batch_of_bbox(img, dict_,key_,score_thr,img_size=(299,299),acceptable_bbox_classes=[],filter_out=False):
    bb_ids = []
    bboxes_norm = []
    for ii,entry in enumerate(dict_[key_]):
        if filter_out:
            if entry['label'] not in acceptable_bbox_classes:
                continue

        if entry['score']>=score_thr:
            bb_ids.append(ii)
    img_batch = np.empty((len(bb_ids),img_size[0],img_size[1],3), dtype='float32')
    bboxes_norm = np.empty((len(bb_ids),4), dtype='float32')
    
    

    for i,bb_id in enumerate(bb_ids):
        # Done: Debug for cropping with bounding box
        bbox_norm = dict_[key_][bb_id]['bbox_normalized']
#         print('bbox output',dict_[key_][bb_id]['bbox'])
        bboxes_norm[i,:] = bbox_norm
        bbox = bbox_norm   
        img = cv2.resize(img,img_size)
        x, y, _ = np.shape(img)
#         print(np.shape(img),bbox)
#         print('recovered',int(bbox[1]*y), int(bbox[3]*y), int(bbox[0]*x),int(bbox[2]*x))
        cropped_img = img[int(bbox[1]*y):int(bbox[3]*y), int(bbox[0]*x):int(bbox[2]*x)] #1 3 0 2 
        cv2.imwrite('cropped/'+key_+'_cropped_'+str(i)+'.jpg', cropped_img)
        img_batch[i,:,:,:] = cv2.resize(cropped_img,img_size)
    return img_batch, bb_ids, bboxes_norm

def mask_fm_bbox(feature_map_size, bbox_norm, order='xyxy'):
    H,W = feature_map_size
    mask = np.zeros(feature_map_size, dtype='int32')
    if order == 'xyxy':
        y_min = int(bbox_norm[1]*H)
        y_max = int(bbox_norm[3]*H)
        x_min = int(bbox_norm[0]*W)
        x_max = int(bbox_norm[2]*W)
    elif order == 'xxyy':
        y_min = int(bbox_norm[2]*H)
        y_max = int(bbox_norm[3]*H)
        x_min = int(bbox_norm[0]*W)
        x_max = int(bbox_norm[1]*W)
    if x_min == x_max:
        if x_max<W:
            x_max+=1
        else:
            x_min-=1
    if y_min == y_max:
        if y_max<H:
            y_max+=1
        else:
            y_min-=1
    mask[y_min:y_max,x_min:x_max] = 1
    return mask
    
#takes txt_id+path and returns sentences and their timestamps
def doc2time(file_id, path_dict):
    file_type = file_id.split('.')[1]
    if file_type == 'mp4':
        asr_path = path_dict[file_id]['asr']
        map_path = path_dict[file_id]['map']
    else:
        return []
    if not os.path.isfile(asr_path) or not os.path.isfile(map_path):
        return []
    with open(asr_path, 'r') as f:
        lines = f.readlines()
    with open(map_path) as f:
        lines_map = f.readlines()
    id2time = {}
    for line in lines_map:
        entries = line.split('\t')
        id2time[entries[0]] = (float(entries[1]),float(entries[2]))
    strip_chars = ['.',':',';',',','"','[',']','(',')','{','}','?','!',"'"]
    text=[]
    sen2time = {}
    for i,line in enumerate(lines):
        line = line.strip('\t')
        if line.strip('\n').find('<SEG ')==0:
            seg_data = line.strip('>\n').strip('<').split(' ')
            for entry in seg_data:
                if entry.find('start_char=')==0:
                    begin = int(entry[len('start_char='):].strip('"'))
                elif entry.find('end_char=')==0:
                    end = int(entry[len('end_char='):].strip('"'))
                elif entry.find('id=')==0:
                    sen_id = entry[len('id='):].strip('"')
            splits = re.split(r'(<ORIGINAL_TEXT>|</ORIGINAL_TEXT>\n)', lines[i+1])
            if len(splits)>1:
                text=splits[2]
                for strip_char in strip_chars:
                    text = text.replace(strip_char,'')
                if len(text)>0:
                    sen2time[text] = id2time[sen_id]
    return sen2time

#takes kfrm_msb and returns timestamps of each keyframe
def id2time(kfrm_msb):
    with open(kfrm_msb,'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines] #split by whitespace
    id2range = {}
    for line in lines[1:]:
        id_=line[1]
        t_range = (float(line[4]),float(line[5]))
        file_id = id_.split('_')[0]+'.mp4.ldcc'
        if file_id not in id2range:
            id2range[file_id] = {id_: t_range}
        else:
            id2range[file_id].update({id_: t_range})
    return id2range

#takes sentence+file_id+timestamp dicts and returns keyframes that cover that sentence
def sen2kfrm(sen,file_id,doc2time_dict,id2time_dict):
    margin = 5
    if len(doc2time_dict)==0 or len(id2time_dict)==0:
        return []
    elif sen not in doc2time_dict or file_id not in id2time_dict:
        return []
    s0,s1 = doc2time_dict[sen]
    kfrm_time = id2time_dict[file_id]
    kfrm_cover = []
    for kfrm in kfrm_time:
        k0,k1 = kfrm_time[kfrm]
        if (k0-margin) <= s0 <= (k1+margin) or (k0-margin) <= s1 <= (k0+margin)\
            or (k0>s0 and k1>s0 and k0<s1 and k1<s1):
            kfrm_cover.append(kfrm)
    return kfrm_cover

def load_model(model_path,config):
    print('Model Loading...')
    new_graph = tf.Graph()
    sess = tf.InteractiveSession(graph = new_graph, config=config)
    _ = sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    print('pass')
    new_saver = tf.train.import_meta_graph(model_path+'.meta')
    print('----Model Loaded-----')
    new_saver.restore(sess, model_path)
    return sess, new_graph

def crop_resize_im(image, bbox, size, order='xxyy'):
    H,W,_ = image.shape
    if order=='xxyy':
        roi = image[int(bbox[2]*H):int(bbox[3]*H),int(bbox[0]*W):int(bbox[1]*W),:]
    elif order=='xyxy':
        roi = image[int(bbox[1]*H):int(bbox[3]*H),int(bbox[0]*W):int(bbox[2]*W),:]
#     print('mark')
    if np.size(roi) ==0:
        return None
    roi = cv2.resize(roi,size)
    return roi
   
def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def IoU(boxA, boxB):
    #order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def bbox_match(bbox_norms,id_,objdet_dict):
    thr = .7
    obj_bbox_thr = .2
    link_ids = []
    link_scores = []
    id_ = id_.split('.')[0]
    if id_ in objdet_dict:
        for i,bbox_norm in enumerate(bbox_norms):
            link_ids_i = []
            link_scores_i = []
            for j,obj in enumerate(objdet_dict[id_]):
                obj_bbox_norm = obj['bbox_normalized']
                obj_bbox_score = obj['score']
                if obj_bbox_score<obj_bbox_thr:
                    continue
                iou = IoU(bbox_norm,obj_bbox_norm)
                if iou>thr:
                    link_scores_i.append(iou)
                    link_ids_i.append(f"{id_}/{j}")
            link_ids.append(link_ids_i)
            link_scores.append(link_scores_i)
    return link_ids, link_scores

def IoU_inv(boxA, boxB):
    #order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return 1/(1+iou)

def men2en_grnd(men_dict_,dict_obj):
    #gathering all bboxes for same image (from different mentions)
    img_grnd_dict = {}
    for men in men_dict_:
        for img_id in men_dict_[men]['grounding']:
            grnd_dict = men_dict_[men]['grounding'][img_id]
            if len(grnd_dict)==0:
                continue
            n_b = len(grnd_dict['bbox'])
            if n_b==0:
                continue
            #print(n_b)
            heat_rep = [grnd_dict['heatmap']]*n_b
            men_score_rep = [np.mean(grnd_dict['men-img-score'])]*n_b
            g_feat_rep = [grnd_dict['grounding_features'].tolist()]*n_b
            if img_id not in img_grnd_dict:
                img_grnd_dict[img_id] = {'bbox': grnd_dict['bbox'],
                                         'bbox_norm': grnd_dict['bbox_norm'],
                                         'bbox_score': grnd_dict['bbox_score'],
                                         'heatmap': heat_rep,
                                         'men-img-score': men_score_rep,
                                         'grounding_features': g_feat_rep}
            else:
                img_grnd_dict[img_id]['bbox'].extend(grnd_dict['bbox'])
                img_grnd_dict[img_id]['bbox_norm'].extend(grnd_dict['bbox_norm'])
                img_grnd_dict[img_id]['bbox_score'].extend(grnd_dict['bbox_score'])
                img_grnd_dict[img_id]['heatmap'].extend(heat_rep)
                img_grnd_dict[img_id]['men-img-score'].extend(men_score_rep)
                img_grnd_dict[img_id]['grounding_features'].extend(g_feat_rep)
    
    for img_id in img_grnd_dict:
        db = DBSCAN(eps=0.6, min_samples=1, metric=IoU_inv)
        X = img_grnd_dict[img_id]['bbox_norm']
        db.fit(X)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        for item in img_grnd_dict[img_id]:
            data_list = img_grnd_dict[img_id][item]
            center_list = []
            if item=='bbox':
                dtp = 'int32'
            else:
                dtp = 'float32'
            for i in range(n_clusters_):
                w = list(map(int,labels==i))
                center_list.append(np.average(data_list,weights=w,axis=0).astype(dtp))
            
            #replace entries with centers of clusters
            img_grnd_dict[img_id][item] = center_list
        
        bbox_norm = img_grnd_dict[img_id]['bbox_norm']
        link_ids, link_scores = bbox_match(bbox_norm,img_id,dict_obj)
        img_grnd_dict[img_id].update({'link_ids': link_ids, 'link_scores': link_scores, 'system': 'Columbia_Vision'})
    
    return img_grnd_dict
