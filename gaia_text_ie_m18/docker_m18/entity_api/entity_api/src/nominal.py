"""
Created on Thu Aug 30 17:44:51 2018

@author: Ananya
"""
import os
from flashtext import KeywordProcessor


def extract(type_files, rsd_str, ltf_str, lang, doc_id):
    entries = []
    count = -1

    for f in type_files:
        # gpe_ru.txt, fac_ru.txt, etc.. which contain the nominal keywords in lang
        enttype = os.path.basename(f).split('.')[0].upper().split('_')[0]
        keyword_processor = KeywordProcessor()

        # for each of the keyword files do....
        with open(f, 'r', encoding='utf-8') as keywordsfile:
            keywordsfull = keywordsfile.read().strip()
            keywords = keywordsfull.split('\n')

            # because of the way ltf files are tokenized, only look for full tokens:
            for k in keywords:
                keyword_processor.add_keyword(' ' + k + ' ')
                keyword_processor.add_keyword(' ' + k + '. ')
                keyword_processor.add_keyword(' ' + k + ', ')

                # look in each rsd file (faster than looking in ltf (approx 250 hours))

                # text = rsd.read()
                keywords_found = keyword_processor.extract_keywords(
                    rsd_str, span_info=True)

                # realign with ltf using math, and write output!
                for keyword in keywords_found:
                    count = count + 1
                    k1 = keyword[1]
                    k2 = keyword[2]

                    if (',' in keyword[0] or '.' in keyword[0]):
                        k2 = k2 - 1

                    st = k1 + 1
                    en = k2 - 2

                    entries.append('ELISA_IE' + '\t' + \
                                   lang.upper() + '_MENTION_' + str(
                        count).zfill(7) + '\t' + \
                                   rsd_str[k1 + 1:k2 - 1].strip() + '\t' + \
                                   doc_id + ':' + str(st) + '-' + str(
                        en) + '\t' + \
                                   'NIL' + '\t' + \
                                   enttype + '\t' + \
                                   'NOM' + '\t' + \
                                   '1.000')
    return '\n'.join(entries)

# with open(outfile, 'w', encoding='utf-8') as o:
#     o.write('\n'.join(entries))


def extract_nominal(ltf_str, rsd_str, lang,
                    type_files, doc_id):
    file_list = [os.path.join(type_files, 'fac_' + lang + '.txt'),
                 os.path.join(type_files, 'gpe_' + lang + '.txt'),
                 os.path.join(type_files, 'veh_' + lang + '.txt'),
                 os.path.join(type_files, 'wea_' + lang + '.txt'),
                 os.path.join(type_files, 'per_' + lang + '.txt'),
                 os.path.join(type_files, 'loc_' + lang + '.txt'),
                 os.path.join(type_files, 'org_' + lang + '.txt')]
    return extract(file_list, rsd_str, ltf_str, lang, doc_id)
