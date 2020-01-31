import os
import argparse

parser = argparse.ArgumentParser(description='Convert the visualization to brat file')
parser.add_argument('-i', '--input_file_path', help='Input file path', required=True)
parser.add_argument('-o', '--output_folder_path', help='Output folder path', required=True)
args = vars(parser.parse_args())

input_cs_file = args['input_file_path']
output_brat_folder = args['output_folder_path']

try:
    os.mkdir(output_brat_folder)
except:
    pass

for one_file in os.listdir(output_brat_folder):
    if '.rsd.ann' in one_file:
        one_file_path = os.path.join(output_brat_folder, one_file)
        os.remove(one_file_path)

uri_head = 'https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/SeedlingOntology#'

entity_dict = dict()
event_dict = dict()
file_content_dict = dict()

T_flag = 1
E_flag = 1

entity_mention_dict = dict()
trigger_mention_dict = dict()
arc_dict = dict()
T_flag_dict = dict()

last_event_search_key = ''
for one_line in open(input_cs_file):
    one_line = one_line.strip()
    one_line_list = one_line.split('\t')
    if len(one_line_list) < 3:
        continue
    # throw_relation
    if ':Entity' in one_line_list[0] and ':Entity' in one_line_list[2]:
        # This is a relation line, skip
        continue
    if ':Entity' in one_line_list[0]:
        if one_line_list[0] not in entity_dict:
            entity_dict[one_line_list[0]] = dict()
        if one_line_list[1] == 'type':
            entity_dict[one_line_list[0]]['type'] = one_line_list[2].replace(uri_head, '')
        if len(one_line_list) == 5:
            search_key = one_line_list[3]
            if search_key in entity_mention_dict:
                continue
            doc_id = search_key.split(':')[0]
            start_offset = int(search_key.split(':')[1].split('-')[0])
            end_offset = int(search_key.split(':')[1].split('-')[1])
            entity_mention_dict[search_key] = (entity_dict[one_line_list[0]], doc_id, start_offset, end_offset)
    if ':Event' in one_line_list[0]:
        if one_line_list[0] not in event_dict:
            event_dict[one_line_list[0]] = dict()
        if one_line_list[1] == 'type':
            event_dict[one_line_list[0]]['type'] = one_line_list[2].replace(uri_head, '')
        if len(one_line_list) == 5:
            if ':Entity' not in one_line_list[2]:
                search_key = one_line_list[3]
                if last_event_search_key != search_key:
                    last_event_search_key = search_key
                if search_key in trigger_mention_dict:
                    continue
                doc_id = search_key.split(':')[0]
                start_offset = int(search_key.split(':')[1].split('-')[0])
                end_offset = int(search_key.split(':')[1].split('-')[1])
                trigger_mention_dict[search_key] = (event_dict[one_line_list[0]], doc_id, start_offset, end_offset)
            else:
                search_key = one_line_list[3]
                if last_event_search_key not in arc_dict:
                    arc_dict[last_event_search_key] = list()
                doc_id = search_key.split(':')[0]
                argument_role = one_line_list[1].replace(uri_head, '').replace('.actual', '').split('_')[1]
                arc_dict[last_event_search_key].append((doc_id, event_dict[one_line_list[0]], argument_role, search_key))

for one_entry in entity_mention_dict:
    type_info = entity_mention_dict[one_entry][0]['type']
    doc_id = entity_mention_dict[one_entry][1]
    start_offset = entity_mention_dict[one_entry][2]
    end_offset = entity_mention_dict[one_entry][3]
    if doc_id not in file_content_dict:
        f_read = open(os.path.join(output_brat_folder, '%s.rsd.txt' % doc_id))
        f_content = f_read.read()
        file_content_dict[doc_id] = f_content
        f_read.close()
    text_string = file_content_dict[doc_id][start_offset:end_offset+1]
    f_write = open(os.path.join(output_brat_folder, '%s.rsd.ann' % doc_id),'a')
    flag_name = "T%d" % T_flag
    T_flag += 1
    one_temp_line = '%s\t%s %d %d\t%s\n' % (flag_name, type_info, start_offset, end_offset+1, text_string.replace('\n', ' '))
    T_flag_dict[one_entry] = flag_name
    f_write.write(one_temp_line)
    f_write.close()

for one_entry in trigger_mention_dict:
    type_info = trigger_mention_dict[one_entry][0]['type']
    doc_id = trigger_mention_dict[one_entry][1]
    start_offset = trigger_mention_dict[one_entry][2]
    end_offset = trigger_mention_dict[one_entry][3]
    if doc_id not in file_content_dict:
        f_read = open(os.path.join(output_brat_folder, '%s.rsd.txt' % doc_id))
        f_content = f_read.read()
        file_content_dict[doc_id] = f_content
        f_read.close()
    text_string = file_content_dict[doc_id][start_offset:end_offset + 1]
    f_write = open(os.path.join(output_brat_folder, '%s.rsd.ann' % doc_id),'a')
    flag_name = "T%d" % T_flag
    T_flag += 1
    one_temp_line = '%s\t%s %d %d\t%s\n' % (flag_name, type_info, start_offset, end_offset+1, text_string.replace('\n', ' '))
    T_flag_dict[one_entry] = flag_name
    f_write.write(one_temp_line)
    f_write.close()

for one_entry in arc_dict:
    doc_id = arc_dict[one_entry][0][0]
    one_temp_line_list = list()
    event_type = trigger_mention_dict[one_entry][0]['type']
    event_trigger_flag_name = T_flag_dict[one_entry]
    one_temp_line_list.append('%s:%s' % (event_type, event_trigger_flag_name))
    for one_item in arc_dict[one_entry]:
        role_name = one_item[2]
        argument_flag_name = T_flag_dict[one_item[3]]
        one_temp_line_list.append('%s:%s' % (role_name, argument_flag_name))
    flag_name = "E%d" % E_flag
    one_temp_line = '%s\t%s\n' % (flag_name, ' '.join(one_temp_line_list))
    E_flag += 1
    f_write = open(os.path.join(output_brat_folder, '%s.rsd.ann' % doc_id),'a')
    f_write.write(one_temp_line)
    f_write.close()