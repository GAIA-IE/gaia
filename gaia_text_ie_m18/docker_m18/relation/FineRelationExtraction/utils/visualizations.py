import os
import xml.etree.ElementTree as ET


def html_visualization(lang_id, cs_output_path, ltf_path, rsd_path, out_dir, outfname_form="{}.rel_vis.html"):
    # cs_folder_path = '/Volumes/Ann/univ/blender/supervised/ann_data_m9/eval_0627/visualization/rel/'
    # ltf_path = '/Volumes/Ann/univ/blender/supervised/ann_data_m9/eval_0627/source/uk/'
    # outfile = '/Volumes/Ann/univ/blender/supervised/ann_data_m9/eval_0627/visualization/rel/uk_rel.html'
    # rsd_path = '/Volumes/Ann/univ/blender/supervised/ann_data_m9/eval_0627/source/uk_rsd/'

    ltfdict = dict()
    rsddict = dict()
    sentences = []
    sentences.append(
        '<html><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8" /> </head><body>\n')
    count = 0

    with open(os.path.join(cs_output_path), 'r', encoding='utf-8') as csfile:
        cs = csfile.read()
        for rsdf in os.listdir(rsd_path):
            if '._' not in rsdf:
                ltff = rsdf.replace('.rsd.txt', '.ltf.xml')
                with open(os.path.join(rsd_path, rsdf), 'r', encoding='utf-8') as rsdfile, \
                        open(os.path.join(ltf_path, ltff), 'r', encoding='utf-8') as ltffile:
                    rsd = rsdfile.read()
                    ltf = ltffile.read()
                    rsddict[ltff.split('.')[0]] = rsd
                    ltfdict[ltff.split('.')[0]] = dict()
                    root = ET.fromstring(ltf)
                    for seg in root[0][0]:
                        sent = ""
                        start = 9999999999
                        end = -1
                        for child in seg:
                            if child.tag != 'TOKEN':
                                continue
                            token = child.text.strip()
                            sent += token + " "
                            start = int(child.get('start_char')) if int(child.get('start_char')) < start else start
                            end = int(child.get('end_char')) if int(child.get('end_char')) > end else end

                        sent = sent.replace('\n', ' ')
                        ltfdict[ltff.split('.')[0]][str(start) + '-' + str(end)] = sent

        for line in cs.split('\n'):
            elements = line.strip().split('\t')
            if len(elements) == 1:
                continue
            if len(elements) > 3:
                if elements[3].split(':')[0] not in rsddict:
                    continue
            if len(elements) == 3:
                trig_type = elements[2]

            if 'Relation' not in elements[0]:
                continue
            elif elements[1] == 'mention.actual':
                fname = elements[3].split(':')[0]
                start, end = elements[3].split(':')[1].split('-')
                for entry in ltfdict[fname]:
                    begin, finish = entry.split('-')
                    if begin <= start and finish <= end:
                        temp = ltfdict[elements[3].split(':')[0]][entry]
                temp = rsddict[fname][max(0, int(start) - 200): min(len(rsddict[fname]), int(end) + 200)].replace(
                    '\n', ' ')
                count = count + 1
                sentences.append('<br /><br />' + str(count) + ('-' * 70) + '<br />')
                sentences.append('<b>FILE:</b>' + fname)
                sentences.append('<b>RELATION OF TYPE ' + trig_type.split('#')[-1] + ' :</b>')
                sentences.append('<b>CONTEXT:</b>')
                sentences.append(rsddict[fname][int(start): int(end) + 1])
                sentences.append('<br />')
            elif ':Entity_' in elements[2] or ':Filler_' in elements[2] or ':Event_' in elements[2]:
                fname = elements[3].split(':')[0]
                sentences.append(
                    '<b>ARGUMENT OF TYPE ' + elements[1].split('#')[-1] + ' :</b> ' + elements[2] + ' ' + elements[
                        3])
                sentences.append(rsddict[fname][int(elements[3].split(':')[1].split('-')[0]):int(
                    elements[3].split(':')[1].split('-')[1]) + 1])
                sentences.append('<br />')

    outfname = os.path.join(out_dir, outfname_form.format(lang_id))
    with open(outfname, 'w', encoding='utf-8') as o:
        sentences.append('</body></html>')
        o.write('<br />'.join(sentences))


def brat_visualize_relation_results(result_entities, result_relations, result_events, outdir, rel_ontology):
    # entities = results[KE_ENT]
    # relations = results[KE_REL]
    # events = results.get(KE_EVT, {})
    entities = result_entities
    relations = result_relations
    events = result_events

    file_content_dict = dict()

    T_flag = 1
    R_flag = 1
    T_flag_dict = dict()

    ke_prov_strs = set()

    for entity_id, ent_mention_list in entities.items():
        for ent_mention in ent_mention_list:
            if ent_mention.get_provenance_offset_str() in ke_prov_strs:
                continue

            type_info = ent_mention.type
            doc_id = ent_mention.provenance
            start_offset = ent_mention.start_offset
            end_offset = ent_mention.end_offset
            if doc_id not in file_content_dict:
                f_read = open(os.path.join(outdir, '{}.rsd.txt'.format(doc_id)))
                f_content = f_read.read()
                file_content_dict[doc_id] = f_content
                f_read.close()
            text_string = file_content_dict[doc_id][start_offset:end_offset + 1]
            f_write = open(os.path.join(outdir, '{}.rsd.ann'.format(doc_id)), 'a')
            flag_name = "T%d" % T_flag
            T_flag += 1
            one_temp_line = '{}\t{} {} {}\t{}\n'.format(
                flag_name, type_info, start_offset, end_offset + 1, text_string.replace('\n', ' ')
            )
            # T_flag_dict[ent_mention.mention_id] = flag_name
            T_flag_dict[ent_mention.get_provenance_offset_str()] = flag_name
            f_write.write(one_temp_line)
            ke_prov_strs.add(ent_mention.get_provenance_offset_str())
            f_write.close()

    ke_prov_strs = set()

    for relation_id, relation_mention_list in relations.items():
        for rel_mention in relation_mention_list:
            if rel_mention.get_provenance_offset_str() in ke_prov_strs:
                continue

            doc_id = rel_mention.provenance
            # relation_type = rel_ontology.map2type_abbrev(rel_mention.get_full_type())
            relation_type = rel_ontology.map_type2abbrev(rel_mention.get_full_type())
            # relation_type = rel_mention.get_full_type()
            # relation_type = rel_mention.subtype if not rel_mention.subsubtype else rel_mention.subsubtype
            arg_ents = [rel_mention.retrieve_arg(curr_arg, entities, events=events) for curr_arg in rel_mention.args]

            arg_str = "Arg{}:{}"
            one_temp_line_list = list()
            one_temp_line_list.append(relation_type)
            for i, arg in enumerate(arg_ents, 1):
                # one_temp_line_list.append(arg_str.format(i, T_flag_dict[arg.mention_id]))
                one_temp_line_list.append(arg_str.format(i, T_flag_dict[arg.get_provenance_offset_str()]))
            flag_name = "R{}".format(R_flag)
            one_temp_line = '{}\t{}\n'.format(flag_name, ' '.join(one_temp_line_list))
            R_flag += 1
            f_write = open(os.path.join(outdir, '{}.rsd.ann'.format(doc_id)), 'a')
            f_write.write(one_temp_line)
            ke_prov_strs.add(rel_mention.get_provenance_offset_str())
            f_write.close()

    ke_prov_strs = set()

    for event_id, event_mention_list in events.items():
        for event_mention in event_mention_list:
            if event_mention.get_provenance_offset_str() in ke_prov_strs:
                continue

            type_info = event_mention.type
            doc_id = event_mention.provenance
            start_offset = event_mention.start_offset
            end_offset = event_mention.end_offset
            if doc_id not in file_content_dict:
                f_read = open(os.path.join(outdir, '{}.rsd.txt'.format(doc_id)))
                f_content = f_read.read()
                file_content_dict[doc_id] = f_content
                f_read.close()
            text_string = file_content_dict[doc_id][start_offset:end_offset + 1]
            f_write = open(os.path.join(outdir, '{}.rsd.ann'.format(doc_id)), 'a')
            flag_name = "T%d" % T_flag
            T_flag += 1
            one_temp_line = '{}\t{} {} {}\t{}\n'.format(
                flag_name, type_info, start_offset, end_offset + 1, text_string.replace('\n', ' ')
            )
            T_flag_dict[event_mention.mention_id] = flag_name
            f_write.write(one_temp_line)
            ke_prov_strs.add(event_mention.get_provenance_offset_str())
            f_write.close()





if __name__ == '__main__':
    from collections import Counter


    def read_dist_file(fname):
        curr_dist = Counter()
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rel_type, rel_type_count = line.split(",")
                    rel_type_count = int(rel_type_count)
                    curr_dist[rel_type] = rel_type_count
        return curr_dist


    def output_diff(outfname, dist1, dist2):
        differential = dist1 - dist2
        with open(outfname, "w", encoding="utf-8") as outf:
            for rel_type, rel_type_count in differential.most_common():
                outf.write("{},{}\n".format(rel_type, rel_type_count))
                # print("{},{}".format(rel_type, rel_type_count))


    ta_pairings = [
        ("./data/en_all.type_dist.relation.TA1a.txt", "./data/en_all.type_dist.relation.TA1b.txt"),
        ("./data/ru_all.type_dist.relation.TA1a.txt", "./data/ru_all.type_dist.relation.TA1b.txt"),
        ("./data/uk.type_dist.relation.TA1a.txt", "./data/uk.type_dist.relation.TA1b.txt")
    ]

    ta_diff_fnames = [
        "./data/en_all.type_dist.relation.DIFF.txt",
        "./data/ru_all.type_dist.relation.DIFF.txt",
        "./data/uk.type_dist.relation.DIFF.txt"
    ]

    for i, (ta1a_fname, ta1b_fname) in enumerate(ta_pairings):
        ta1a = read_dist_file(ta1a_fname)
        ta1b = read_dist_file(ta1b_fname)

        output_diff(ta_diff_fnames[i], ta1b, ta1a)




    # ta1a_fname = "./data/eval_dist.TA1a.txt"
    # with open(ta1a_fname, "r", encoding="utf-8") as ta1af:
    #     for line in ta1af:
    #         line = line.strip()
    #         if line:
    #             rel_type, rel_type_count = line.split(",")
    #             rel_type_count = int(rel_type_count)
    #             ta1a_total_relations += rel_type_count
    #             ta1a_dist[rel_type] = rel_type_count
    # print(ta1a_total_relations)
    # print(ta1a_total_relations)
    #
    # ta1b_fname = "./data/eval_dist.TA1b.txt"
    # ta1b_total_relations = 0
    # ta1b_dist = Counter()
    # with open(ta1b_fname, "r", encoding="utf-8") as ta1bf:
    #     for line in ta1bf:
    #         line = line.strip()
    #         if line:
    #             rel_type, rel_type_count = line.split(",")
    #             rel_type_count = int(rel_type_count)
    #             ta1b_total_relations += rel_type_count
    #             ta1b_dist[rel_type] = rel_type_count
    # print(ta1b_total_relations)
    #
    # print(ta1b_total_relations - ta1a_total_relations)
    #
    # differential = ta1b_dist - ta1a_dist
    #
    # for rel_type, rel_type_count in differential.most_common():
    #     print("{},{}".format(rel_type, rel_type_count))



    # cs_folder_path = '/Volumes/Ann/univ/blender/supervised/ann_data_m9/eval_0627/visualization/rel/'
    # ltf_path = '/Volumes/Ann/univ/blender/supervised/ann_data_m9/eval_0627/source/uk/'
    # outfile = '/Volumes/Ann/univ/blender/supervised/ann_data_m9/eval_0627/visualization/rel/uk_rel.html'
    # rsd_path = '/Volumes/Ann/univ/blender/supervised/ann_data_m9/eval_0627/source/uk_rsd/'

    # lang_id = "en_all"
    # cs_output_path = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/{0}/TEST{0}.fine_rel.cs".format(lang_id)
    # ltf_path  = "/data/m1/lim22/aida2019/LDC2019E42/source/en_all"
    # rsd_path = "/data/m1/lim22/aida2019/LDC2019E42/source/en_all_rsd"
    # # out_dir = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/{}".format(lang_id)
    # out_dir = "./"
    #
    # print(cs_output_path)
    # input()
    #
    # html_visualization(lang_id, cs_output_path, ltf_path, rsd_path, out_dir)
