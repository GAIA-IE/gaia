import os
import ujson as json
import time
import random
import unicodedata
from collections import OrderedDict, Counter
from argparse import ArgumentParser

from utils.ltf_reading_utils import LTFUtil
from utils.finegrained_entity_utils import FineGrainedEntityUtil
from utils.dependency_parse_utils import DependencyParseUtil, DependencyPattern
from utils.visualizations import brat_visualize_relation_results, html_visualization


KE_ENT = "Entity"
KE_FILLER = "Entity_Filler"
KE_LANG_LABELS = {"en": "ENG", "en_asr": "ENG", "en_ocr_img": "ENG", "en_ocr_video": "ENG", "en_all": "ENG",
                  "ru": "RUS", "ru_asr": "RUS", "ru_ocr_img": "RUS", "ru_ocr_video": "RUS", "ru_all": "RUS",
                  "uk": "UKR", "uk_asr": "UKR", "uk_ocr_img": "UKR", "uk_ocr_video": "UKR"}
KE_ENT_FIELDS = ['tree_id', 'entitymention_id', 'entity_id', 'provenance', 'textoffset_startchar',
                 'textoffset_endchar', 'text_string', 'justification', 'type', 'level', 'kb_id', "link",
                 "confidence"]

KE_REL = "Relation"
KE_REL_FIELDS = ['tree_id', 'relationmention_id', 'relation_id', 'provenance', 'textoffset_startchar',
                 'textoffset_endchar', 'text_string', 'justification', 'type', 'subtype', 'attribute',
                 'start_date_type', 'start_date', 'end_date_type', 'end_date', 'kb_id', 'args']
KE_REL_ARG_FIELDS = ["tree_id", "relationmention_id", "slot_type", "arg_id"]

KE_EVT = "Event"

KE_ENT_CONV = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "GPE": "GeopoliticalEntity",
    "FAC": "Facility",
    "VEH": "Vehicle",
    "WEA": "Weapon",
    "MON": "Money",
    "TME": "Time",
    "TTL": "Title",
    "VAL": "NumericalValue",
    "URL": "Website",
    "SID": "Side",
    "COM": "Commodity",
    "LAW": "Law",
    "BAL": "Ballot",
    "RES": "Results",
    "CRM": "Crime",
    "EVENT": KE_EVT,
    "COL": "Color"
}


def form_cs_id(num, ke_label, lang_label, num_digits=7):
    if not lang_label:
        return "_".join([":" + ke_label, str(num).zfill(num_digits)])
    else:
        return "_".join([":" + ke_label, KE_LANG_LABELS[lang_label], str(num).zfill(7)])


class CSParser(object):
    REL_IDENT = "SeedlingOntology"
    FILLER_IDENT = "Filler"

    def __init__(self, delimiter='\t'):
        self.delim = delimiter
        self.num_entities = 0
        self.num_relations = 0

    def get_provenance(self, prov_str):
        prov, offsets = prov_str.split(':')
        startchar, endchar = int(offsets.split('-')[0]), int(offsets.split('-')[1])
        return prov, startchar, endchar

    def reset_counters(self):
        self.num_relations = 0
        self.num_entities = 0

    def _reset_event_mention(self, event_id, event_type, event_subtype):
        fresh_event_mention = EventMention()
        fresh_event_mention.id = event_id
        fresh_event_mention.type = event_type
        fresh_event_mention.subtype = event_subtype
        return fresh_event_mention

    def form_event_mentions(self, event_id, event_type, event_subtype, event_data):
        event_mention = self._reset_event_mention(event_id, event_type, event_subtype)

        ment_num = 0
        for entry in event_data:
            ev_info = entry[1].split(".")
            if ev_info[0] == "mention":
                if ment_num:
                    yield event_mention
                    event_mention = self._reset_event_mention(event_id, event_type, event_subtype)

                event_mention.text_string = entry[2].strip('"')
                event_mention.mention_id = event_id + "|" + entry[3]
                prov, startchar, endchar = self.get_provenance(entry[3])
                event_mention.provenance = prov
                event_mention.start_offset = startchar
                event_mention.end_offset = endchar
                ment_num += 1
            elif ev_info[0] != "canonical_mention":
                arg = ArgumentMention()
                arg.mention_id = event_mention.mention_id

                temp_slot_type = entry[1].split("#")
                if len(temp_slot_type) == 1:
                    temp_slot_type = temp_slot_type[0].split(".")[1].split("_")[1]
                else:
                    temp_slot_type = temp_slot_type[1].split(".")[1].split("_")[1]
                arg.slot_type = temp_slot_type
                # arg.slot_type = entry[1].split("#")[1].split(".")[1].split("_")[1]
                arg.arg_id = entry[2]
                arg.provenance, arg.start_offset, arg.end_offset = self.get_provenance(entry[3])
                event_mention.add_argument(arg)
        yield event_mention

    def parse_otf(self, lang_id, fname, link2fine_map, offset2link_map, fine_ent_taxonomy=None):
        with open(fname, 'r', encoding="utf-8") as f:
            type_info = {}
            link_info = {}
            curr_ent = {}
            curr_ev_id = "start"
            curr_event = []
            for line in f:
                line = line.rstrip()

                if line:
                    temp = line.split(self.delim)

                    if len(temp) > 1:
                        if temp[1] == "link":
                            link_info[temp[0]] = temp[2]
                            curr_ent_ments = curr_ent.pop(temp[0])
                            for ent_ment in curr_ent_ments:
                                if link2fine_map:
                                    ent_ment.link = link_info[temp[0]]
                                    # id_link = link_info[temp[0]].replace("LDC2015E42:", "")
                                    # leaf_fine_types = link2fine_map.get(id_link, [])
                                    # if fine_ent_taxonomy:
                                    #     ent_ment.fine_grained_types = fine_ent_taxonomy.get_all_types(leaf_fine_types)
                                    # else:
                                    #     ent_ment.fine_grained_types = leaf_fine_types
                                yield KE_ENT, ent_ment
                            continue
                        # if temp[1] == "link":
                        #     link_info[temp[0]] = temp[2]
                        #     continue

                        if temp[1] == "type":
                            temp_type = temp[2].split('#')
                            if len(temp_type) == 1:
                                if KE_EVT not in temp[0]:
                                    type_info[temp[0]] = KE_ENT_CONV[temp_type[0]]
                                else:
                                    type_info[temp[0]] = temp_type[0]
                            else:
                                type_info[temp[0]] = temp_type[1]
                            continue

                        if KE_ENT in temp[0] or CSParser.FILLER_IDENT in temp[0]:
                            # if CSParser.REL_IDENT not in temp[1]:
                            if KE_ENT not in temp[2] and CSParser.FILLER_IDENT not in temp[2] and KE_EVT not in temp[2]:
                                if temp[1] == "canonical_mention":
                                    continue
                                try:
                                    entity_id = temp[0]
                                    ment_id = entity_id + "|" + temp[3]
                                    ment_type = temp[1]
                                    text_str = temp[2].strip('"')
                                    prov, startchar, endchar = self.get_provenance(temp[3])
                                    ent_type = type_info[entity_id]
                                    ent_link = link_info.get(entity_id, None)
                                    ent_mention = EntityMention()
                                    ent_mention.id = entity_id
                                    ent_mention.confidence = temp[4]  # float(temp[4])
                                    ent_mention.mention_id = ment_id
                                    ent_mention.mention_type = ment_type
                                    ent_mention.type = ent_type
                                    ent_mention.link = ent_link
                                    ent_mention.text_string = text_str
                                    ent_mention.provenance = prov
                                    ent_mention.start_offset = startchar
                                    ent_mention.end_offset = endchar
                                except KeyError:
                                    print(temp)
                                    raise KeyError


                                self.num_entities += 1

                                if CSParser.FILLER_IDENT in entity_id:
                                    yield KE_ENT, ent_mention
                                elif entity_id not in link_info:
                                    curr_ent_ments = curr_ent.get(entity_id, [])
                                    curr_ent_ments.append(ent_mention)
                                    curr_ent[entity_id] = curr_ent_ments
                                else:
                                    ent_mention.link = link_info[entity_id]

                                if temp[3] in offset2link_map:
                                    id_link = offset2link_map[temp[3]]
                                    leaf_fine_types = link2fine_map.get(id_link, [])
                                    if fine_ent_taxonomy:
                                        ent_mention.fine_grained_types = fine_ent_taxonomy.get_all_types(
                                            leaf_fine_types
                                        )
                                    else:
                                        ent_mention.fine_grained_types = leaf_fine_types

                                yield KE_ENT, ent_mention
                                # else:
                                #     if link2fine_map:
                                #         ent_mention.link = link_info[entity_id]
                                #         id_link = link_info[entity_id].replace("LDC2015E42:", "")
                                #         leaf_fine_types = link2fine_map.get(id_link, [])
                                #         if fine_ent_taxonomy:
                                #             ent_mention.fine_grained_types = fine_ent_taxonomy.get_all_types(
                                #                 leaf_fine_types
                                #             )
                                #         else:
                                #             ent_mention.fine_grained_types = leaf_fine_types
                                #         # ent_mention.fine_grained_types = \
                                #         #     link2fine_map.get(id_link, [])
                                #
                                #     yield KE_ENT, ent_mention
                                # # yield KE_ENT, ent_mention

                            else:
                                rdfprefix, curr_type_str = temp[1].split('#')
                                curr_ment_type_list = curr_type_str.split('.')
                                if len(curr_ment_type_list) == 2:
                                    coarse_type, subtype = curr_ment_type_list
                                    subsubtype = None
                                elif len(curr_ment_type_list) == 3:
                                    coarse_type, subtype, subsubtype = curr_ment_type_list
                                else:
                                    raise AttributeError(
                                        "Relations must have `type` and `subtype`. `subsubtype` is optional. " + \
                                        "Recieved: {}".format(line)
                                    )

                                prov, startchar, endchar = self.get_provenance(temp[3])
                                ment_id = form_cs_id(self.num_relations, ke_label=KE_REL, lang_label=lang_id)
                                rel_mention = RelationMention()
                                rel_mention.id = ment_id
                                rel_mention.mention_id = ment_id
                                rel_mention.type = coarse_type
                                rel_mention.subtype = subtype
                                rel_mention.subsubtype = subsubtype
                                rel_mention.provenance = prov
                                rel_mention.start_offset = startchar
                                rel_mention.end_offset = endchar
                                rel_mention.confidence = temp[4]  # float(temp[4])

                                arg1 = ArgumentMention()
                                arg1.mention_id = ment_id
                                arg1.slot_type = "arg1"
                                arg1.arg_id = temp[0]
                                rel_mention.add_argument(arg1)

                                arg2 = ArgumentMention()
                                arg2.mention_id = ment_id
                                arg2.slot_type = "arg2"
                                arg2.arg_id = temp[2]
                                rel_mention.add_argument(arg2)

                                self.num_relations += 1

                                yield KE_REL, rel_mention
                        elif KE_EVT in temp[0]:
                            if curr_ev_id == "start":
                                curr_ev_id = temp[0]

                            if curr_ev_id == temp[0]:
                                if "type" in temp[1]:
                                    type_info[curr_ev_id] = temp[2].split('#')[1]
                                else:
                                    curr_event.append(temp)
                            else:
                                curr_ev_type, curr_ev_subtype = type_info[curr_ev_id].split(".")
                                for ev in self.form_event_mentions(
                                        curr_ev_id,
                                        curr_ev_type,
                                        curr_ev_subtype,
                                        curr_event):
                                    yield KE_EVT, ev
                                curr_ev_id = temp[0]
                                curr_event = [temp]

    def parse(self,
              lang_id,
              fname_list,
              fine_ent_type_fname=None,
              fine_ent_type_linking=None,
              fine_ent_type_taxonomy=None):
        link2fine_map = {}
        if fine_ent_type_fname:
            with open(fine_ent_type_fname, "r", encoding="utf-8") as fef:
                link2fine_map = json.load(fef)

        offset2link_map = {}
        if fine_ent_type_linking:
            with open(fine_ent_type_linking, "r", encoding="utf-8") as felf:
                for line in felf:
                    line = line.rstrip('\n')
                    tabs = line.split('\t')
                    offset = tabs[3]
                    link = tabs[4]
                    if not link.startswith('NIL'):
                        offset2link_map[offset] = link.replace("LDC2015E42:", "")

        for fname in fname_list:
            for ie_type, ke in self.parse_otf(lang_id, fname, link2fine_map, offset2link_map, fine_ent_type_taxonomy):
                yield ie_type, ke


class RelationOntology(object):
    def __init__(self,
                 arg_fname: str = "/home/whites5/AIDA/RelationExtraction/utils/relation_ontology_arguments.txt",
                 separator: str = ":",
                 arg_label_separator: str = "_",
                 type_separator: str = ".",
                 arg_constraint_separator=",",
                 **kwargs):
        self.arg_fname = arg_fname
        self.separator = separator
        self.arg_label_separator = arg_label_separator
        self.type_separator = type_separator
        self.arg_constraint_separator = arg_constraint_separator
        self.abbrev_name_mapping = {
            "APORA": "ArtifactPoliticalOrganizationReligiousAffiliation",
            "MORE": "MemberOriginReligionEthnicity",
            "OPRA": "OrganizationPoliticalReligiousAffiliation"
        }

        self.name_abbrev_mapping = {
            "ArtifactPoliticalOrganizationReligiousAffiliation": "APORA",
            "MemberOriginReligionEthnicity": "MORE",
            "OrganizationPoliticalReligiousAffiliation":  "OPRA"
            # "GeneralAffiliation": "GenAff",
            # "OwnershipPossession": "Own",
            # "OrganizationAffiliation": "OrgAff"
        }

        self.arg_mapping = {}
        self.abbrev_arg_mapping = {}
        self.load()

    def load(self):
        with open(self.arg_fname, 'r', encoding="utf-8") as af:
            for line in af:
                line = line.rstrip()
                if line:
                    arg_num, arg_label, arg_t_constraints = line.split(self.separator, 2)

                    rel_fulltype = arg_label.rsplit(self.arg_label_separator, 1)[0]
                    rel_arg_constraints = arg_t_constraints.split(self.arg_constraint_separator)
                    rel_arg_constraints = rel_arg_constraints + [KE_ENT_CONV[ent_t] for ent_t in rel_arg_constraints]

                    rel_args = self.arg_mapping.get(rel_fulltype, {})
                    rel_arg_num_data = rel_args.get(arg_num, {})
                    rel_arg_num_data["label"] = arg_label
                    rel_arg_num_data["arg_types"] = rel_arg_constraints
                    rel_args[arg_num] = rel_arg_num_data
                    self.arg_mapping[rel_fulltype] = rel_args

                    abbrev_rel_fulltype = self.map_type2abbrev(rel_fulltype)
                    self.abbrev_arg_mapping[abbrev_rel_fulltype] = rel_args

                    # arg_num, arg_label = line.split(self.separator, 1)
                    # rel_fulltype = arg_label.rsplit(self.arg_label_separator, 1)[0]
                    #
                    # rel_args = self.arg_mapping.get(rel_fulltype, {})
                    # rel_args[arg_num] = arg_label
                    # self.arg_mapping[rel_fulltype] = rel_args

    def get_types(self):
        return list(self.arg_mapping.keys())

    def map_type2abbrev(self, rel_mention_full_type):
        abbrev_rel_mention_full_type = rel_mention_full_type
        for name_rel_type, abbrev_rel_type in self.name_abbrev_mapping.items():
            abbrev_rel_mention_full_type = abbrev_rel_mention_full_type.replace(name_rel_type, abbrev_rel_type)
        return abbrev_rel_mention_full_type

    def map_abbrev2type(self, rel_mention):
        rel_fulltype = rel_mention.get_full_type()
        if rel_fulltype not in self.arg_mapping.keys():
            abbrev_expansion = self.abbrev_name_mapping.get(rel_mention.subtype, "")
            if abbrev_expansion:
                rel_mention.subtype = abbrev_expansion
        return rel_mention

    def map_arg_roles(self, rel_mention):
        rel_fulltype = rel_mention.get_full_type()
        if rel_fulltype in self.arg_mapping.keys() or rel_fulltype in self.abbrev_arg_mapping.keys():
            rel_type_arg_roles = self.arg_mapping.get(rel_fulltype, None)
            if rel_type_arg_roles is None:
                rel_type_arg_roles = self.abbrev_arg_mapping[rel_fulltype]
            for arg_mention in rel_mention.args:
                arg_mention.slot_type = rel_type_arg_roles[arg_mention.slot_type]["label"]

        return rel_mention

    def adjust(self, rel_mention):
        adj_rel_mention = self.map_abbrev2type(rel_mention)
        adj_rel_mention = self.map_arg_roles(adj_rel_mention)
        return adj_rel_mention

    def _check_arg_constraints(self, arg_slot_type_map, rel_arg_types):
        check_val = 0
        for arg_slot, arg_type in arg_slot_type_map.items():
            if arg_type in rel_arg_types[arg_slot]["arg_types"]:
                check_val += 1
        return int(check_val == 2)

    def arg_type_constraints(self, arg_slot_type_map, rel_mention_type, test_rev=False):
        if rel_mention_type in self.arg_mapping.keys() or rel_mention_type in self.abbrev_arg_mapping.keys():
            rel_type_arg_roles = self.arg_mapping.get(rel_mention_type, None)
            if rel_type_arg_roles is None:
                rel_type_arg_roles = self.abbrev_arg_mapping[rel_mention_type]
            met_constraint = self._check_arg_constraints(arg_slot_type_map, rel_type_arg_roles)
            if met_constraint == 0 and test_rev:
                arg_role_labels, arg_types = tuple(zip(*list(arg_slot_type_map.items())))
                rev_arg_slot_type_map = dict(zip(arg_role_labels[::-1], arg_types))
                rev_met_constraint = self._check_arg_constraints(rev_arg_slot_type_map, rel_type_arg_roles)
                if rev_met_constraint == 1:
                    return rev_met_constraint + 1
            return met_constraint
        return -1


class KnowledgeElementMention(object):
    RDF_PREFIX = "https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LDCOntology"

    def __init__(self, data=None):
        self._data_dict = data
        self.tree_id = None
        self.mention_id = None
        self.id = None
        self.provenance = None
        self.start_offset = None
        self.end_offset = None
        self.text_string = None
        self.justification = None
        self.type = None
        self.subtype = None
        self.subsubtype = None
        self.kb_id = None
        self.attribute = None
        self.confidence = None
        self.fine_grained_types = []

        if data:
            self.update(data)

    def _univ_update(self, data):
        curr_offsets = []
        remain_items = {}
        for field, value in data.items():
            if "mention_id" in field:
                self.mention_id = value
            elif "id" in field and any([ket in field.lower() for ket in ["entity", "relation", "event"]]):
                self.id = value
            elif "offset" in field:
                if len(curr_offsets) < 2:
                    curr_offsets.append(value)
                elif len(curr_offsets) >= 2:
                    raise ValueError("Too many offsets in entity data: {}".format(data))

                if len(curr_offsets) == 2:
                    curr_offsets.sort()
                    self.start_offset, self.end_offset = curr_offsets
            else:
                try:
                    getattr(self, field)
                    setattr(self, field, value)
                except AttributeError:
                    remain_items[field] = value
        return remain_items

    def update(self, data):
        self._univ_update(data)
        self._data_dict = data

    def update_full_type(self, rel_fulltype):
        rel_iso_type = rel_fulltype.replace(KnowledgeElementMention.RDF_PREFIX, "")
        temp = list(zip(["type", "subtype", "subsubtype"], rel_iso_type.split(".")))
        for type_level, val in temp:
            setattr(self, type_level, val)

    @classmethod
    def form_full_type(cls, m_type, m_subtype=None, m_subsubtype=None):
        type_hierarchy = [m_type]
        if m_subtype:
            type_hierarchy.append(m_subtype)

        if m_subsubtype:
            type_hierarchy.append(m_subsubtype)

        return ".".join(type_hierarchy)

    def get_full_type(self):
        return KnowledgeElementMention.form_full_type(self.type, self.subtype, self.subsubtype)

    def get_rdf_type(self, curr_type=None):
        if curr_type == None:
            return '#'.join([KnowledgeElementMention.RDF_PREFIX, self.get_full_type()])
        else:
            return '#'.join([KnowledgeElementMention.RDF_PREFIX, curr_type])

    @classmethod
    def form_provenance_offset_str(cls, provenance, start_offset, end_offset):
        return ':'.join([provenance, '-'.join([str(start_offset), str(end_offset)])])

    def get_provenance_offset_str(self):
        return KnowledgeElementMention.form_provenance_offset_str(self.provenance, self.start_offset, self.end_offset)
    # def get_provenance_offset_str(self):
    #     return ':'.join([self.provenance, '-'.join([str(self.start_offset), str(self.end_offset)])])


class EntityMention(KnowledgeElementMention):
    def __init__(self, data=None):
        self.mention_type = None  # Name vs nominal
        super().__init__(data)

    def update(self, data):
        remain_data = self._univ_update(data)
        for field, value in remain_data.items():
            if field == KE_ENT_FIELDS[9]:
                self.mention_type = value
        self._data_dict = data

    def __str__(self):
        return '{}\ttype\t{}\n{}\tcanonical_mention\t"{}"\t{}\t{}\n{}\tmention\t{}\t{}\t{}'.format(
            self.id, self.get_rdf_type(),
            self.id, self.text_string, self.get_provenance_offset_str(), self.confidence,
            self.id, self.text_string, self.get_provenance_offset_str(), self.confidence
        )

    def __eq__(self, other):
        return (isinstance(other, EntityMention) and
                self.provenance == other.provenance and
                self.start_offset == other.start_offset and
                self.end_offset == other.end_offset and
                self.type == other.type)

    def match(self, other):
        if self == other:
            return 1
        elif self.start_offset == other.start_offset and self.end_offset == other.end_offset:
            if self.type != other.type:
                return 2
        elif len(range(max(self.start_offset, other.start_offset), min(self.end_offset, other.end_offset) + 1)):
            if self.type == other.type:
                return 3
            elif self.type != other.type:
                return 4
        else:
            return 0


class ArgumentMention(KnowledgeElementMention):
    def __init__(self, data=None):
        self.slot_type = None
        self.arg_id = None
        self.unique_arg_id = str(time.time()) + "|" + str(random.random())
        super().__init__(data)

    def update(self, data):
        remain_data = self._univ_update(data)
        for field, value in remain_data.items():
            getattr(self, field)
            setattr(self, field, value)
        self._data_dict = data

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return (isinstance(other, ArgumentMention) and
                self.arg_id == other.arg_id and
                self.slot_type == other.slot_type)


class StructKnowledgeElementMention(KnowledgeElementMention):
    def __init__(self, data=None):
        self.start_date_type = None
        self.start_date = None
        self.end_date_type = None
        self.end_date = None
        self.political_status = None
        self.args = []
        super().__init__(data)

    def update(self, data):
        remain_data = self._univ_update(data)
        for field, value in remain_data.items():
            getattr(self, field)
            setattr(self, field, value)
        self._data_dict = data

    def add_argument(self, arg):
        if isinstance(arg, dict):
            arg = ArgumentMention(arg)
        self.args.append(arg)
        self.args.sort(key=lambda x: x.slot_type)

    def has_argument(self, entity_id):
        for arg in self.args:
            if arg.arg_id == entity_id:
                return True
        return False

    def retrieve_arg(self, curr_arg, entities, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class RelationMention(StructKnowledgeElementMention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __eq__(self, other):
        return (isinstance(other, RelationMention) and
                self.provenance == other.provenance and
                self.start_offset == other.start_offset and
                self.end_offset == other.end_offset and
                self.args == other.args and
                self.type == other.type and
                self.subtype == other.subtype)

    def __str__(self):
        arg1 = self.args[0].arg_id
        rdf_type = self.get_rdf_type()
        arg2 = self.args[1].arg_id
        prov_offset_str = self.get_provenance_offset_str()
        confidence_str = str(self.confidence)
        return '\t'.join([arg1, rdf_type, arg2, prov_offset_str, confidence_str])

    def str_cs(self, entities, ltf_retriever, events=None):
        rel_id = self.id  # self.args[0].mention_id
        arg1 = self.args[0]
        arg1_id = arg1.arg_id.replace(":Filler", ":Entity_Filler")
        arg1_type = self.get_rdf_type(arg1.slot_type)
        arg1_ent = self.retrieve_arg(arg1, entities, events=events)
        if arg1_ent is None:
            arg1_offset = 'None:None-None'
        else:
            arg1_offset = '{}:{}-{}'.format(arg1_ent.provenance, arg1_ent.start_offset, arg1_ent.end_offset)
        arg2 = self.args[1]
        arg2_id = arg2.arg_id.replace(":Filler", ":Entity_Filler")
        arg2_type = self.get_rdf_type(arg2.slot_type)
        arg2_ent = self.retrieve_arg(arg2, entities, events=events)
        if arg2_ent is None:
            arg2_offset = 'None:None-None'
        else:
            arg2_offset = '{}:{}-{}'.format(arg2_ent.provenance, arg2_ent.start_offset, arg2_ent.end_offset)
        rdf_type = self.get_rdf_type()
        prov_offset_str = self.get_provenance_offset_str()
        mention_str = ltf_retriever.get_str(prov_offset_str)
        confidence_str = str(self.confidence)

        if not mention_str:
            print(arg1_ent)
            print(arg2_ent)
            print(str(self))
            print("\n")

        return '{}\ttype\t{}\n' \
               '{}\tmention.actual\t"{}"\t{}\t{}\n' \
               '{}\tcanonical_mention.actual\t"{}"\t{}\t{}\n' \
               '{}\t{}\t{}\t{}\t{}\n' \
               '{}\t{}\t{}\t{}\t{}'.format(
                  rel_id, rdf_type,
                  rel_id, mention_str, prov_offset_str, confidence_str,
                  rel_id, mention_str, prov_offset_str, confidence_str,
                  rel_id, arg1_type, arg1_id, arg1_offset, confidence_str,
                  rel_id, arg2_type, arg2_id, arg2_offset, confidence_str)

    def flip_args(self):
        flipped_args = []
        for arg_ment in self.args:
            if arg_ment.slot_type == "arg1":
                arg_ment.slot_type = "arg2"
            elif arg_ment.slot_type == "arg2":
                arg_ment.slot_type = "arg1"
            flipped_args.append(arg_ment)
        self.args = flipped_args[::-1]

    def retrieve_arg(self, curr_arg, entities, **kwargs):
        arg_ment = None
        max_overlap = -1
        mentions = entities.get(curr_arg.arg_id, [])
        if not len(mentions):
            events = kwargs.pop("events", {})
            if events:
                mentions = events.get(curr_arg.arg_id, [])

        for ment in mentions:
            if curr_arg.arg_id == ment.id and ment.provenance == self.provenance:
                overlap_list = range(max(self.start_offset, ment.start_offset),
                                     min(self.end_offset, ment.end_offset) + 1)
                n_overlap = len(overlap_list)

                if n_overlap and n_overlap > max_overlap:
                    arg_ment = ment
                    max_overlap = n_overlap

        return arg_ment

    def match(self, my_entities, other, other_entities):
        my_arg_ents = list(filter(
            lambda x: x,
            [self.retrieve_arg(my_arg, my_entities)
             for my_arg in self.args]
        ))

        other_arg_ents = list(filter(
            lambda x: x,
            [self.retrieve_arg(other_arg, other_entities)
             for other_arg in other.args]
        ))

        matches = []
        already_matched = []
        for i, other_arg in enumerate(other_arg_ents):
            for j, my_arg in enumerate(my_arg_ents):
                if j not in already_matched:
                    matches.append(other_arg.match(my_arg))
                    already_matched.append(j)

        if len(matches) == len(other_arg_ents) and all([m > 0 for m in matches]):
            if all([m == 1 for m in matches]) and self.type == other.type:
                if self.subtype == other.subtype:
                    return 1
                elif self.subtype != other.subtype:
                    return 2
            elif all([m in [1, 3] for m in matches]) and self.type == other.type:
                if self.subtype == other.subtype:
                    return 3
                elif self.subtype != other.subtype:
                    return 4
        return 0


class EventMention(StructKnowledgeElementMention):
    def __init__(self, data=None):
        self.attribute2 = None
        super().__init__(data)

    def __eq__(self, other):
        return (isinstance(other, EventMention) and
                self.provenance == other.provenance and
                self.start_offset == other.start_offset and
                self.end_offset == other.end_offset and
                self.args == other.args and
                self.type == other.type and
                self.subtype == other.subtype,
                self.subsubtype == other.subsubtype)

    def retrieve_arg(self, curr_arg, entities, **kwargs):
        arg_ent = None
        min_dist = 10000
        max_overlap = -1
        event_start_offset = curr_arg.start_offset
        event_end_offset = curr_arg.end_offset
        if event_start_offset is None or event_end_offset is None:
            event_start_offset = kwargs["start_offset"]
            event_end_offset = kwargs["end_offset"]

        for ent in entities:
            if curr_arg.arg_id == ent.id:
                n_overlap = len(range(max(self.start_offset, ent.start_offset),
                                      min(self.end_offset, ent.end_offset) + 1))

                curr_dist = -1
                if event_start_offset >= ent.end_offset:
                    curr_dist = event_start_offset - ent.end_offset
                elif event_end_offset <= ent.start_offset:
                    curr_dist = ent.start_offset - event_end_offset

                if curr_dist > -1 and curr_dist < min_dist:
                    arg_ent = ent
                    min_dist = curr_dist
                elif n_overlap and n_overlap > max_overlap:
                    arg_ent = ent
                    max_overlap = n_overlap

        return arg_ent


class RelationRule(object):
    def __init__(self, coarse_type, subtype, subsubtypes, **kwargs):
        self.coarse_type = coarse_type
        self.subtype = subtype
        self.subsubtypes = subsubtypes

        self.full_subtype_name, self.full_subsubtype_names = self.create_fulltype_names(subtype, subsubtypes)

        self.dependency_patterns = {}
        if self.full_subtype_name and self.full_subsubtype_names:
            temp_all_patterns = kwargs.get("dependency_patterns", {})
            self.dependency_patterns = {
                ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
                for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
            }

    def create_fulltype_names(self, m_subtype=None, m_subsubtypes=None):
        fulltype_names = ["", []]
        if m_subtype:
            fulltype_names[0] = KnowledgeElementMention.form_full_type(self.coarse_type, m_subtype)

        if m_subsubtypes and m_subtype:
            fulltype_names[1] = [
                KnowledgeElementMention.form_full_type(self.coarse_type, m_subtype, ftype)
                    for ftype in m_subsubtypes
            ]

        return fulltype_names

    def __call__(self, *args, **kwargs):
        return self.run_rule(*args, **kwargs)

    def run_rule(self, *args, **kwargs):
        raise NotImplementedError

    def _match_type(self, coarse_type, subtype):
        if not self.subtype:
            return coarse_type == self.coarse_type
        else:
            return coarse_type == self.coarse_type and subtype == self.subtype

    def iso_type_str(self, rdf_type):
        rdf_link, ke_type_subtype = rdf_type.split('#')
        ke_type, ke_subtype = ke_type_subtype.split('.')
        return rdf_link, ke_type, ke_subtype

    def get_entity_relations(self, arg1, arg2, all_relations):
        ent_relations = []
        for _, rel_mentions in all_relations.items():
            rel_ment = rel_mentions[0]
            if ((rel_ment.args[0].arg_id in arg1.arg_id and rel_ment.args[1].arg_id in arg2.arg_id) or
                    (rel_ment.args[1].arg_id in arg1.arg_id and rel_ment.args[0].arg_id in arg2.arg_id)):
                ent_relations.append(rel_ment)
        return ent_relations

    def find_keyword_in_segment(self, segment_data, keywords):
        keyword_token_idxs = {}
        for k in keywords:
            k_toks = k.split()
            if k_toks[0] in segment_data.tokens:
                start_idx = segment_data.tokens.index(k_toks[0])
                if segment_data.tokens[start_idx:start_idx + len(k_toks)] == k_toks:
                    kw_startchar = segment_data.token_offsets[start_idx][0]
                    kw_endchar = segment_data.token_offsets[start_idx + len(k_toks) - 1][1]
                    keyword_token_idxs[k] = (kw_startchar, kw_endchar)
        return keyword_token_idxs

    def retrieve_segment_mentions(self, provenance, start_offset, end_offset, ke_mentions):
        segment_entities = []
        for ment in ke_mentions:
            if ment.provenance == provenance:
                overlap_list = range(max(start_offset, ment.start_offset),
                                      min(end_offset, ment.end_offset) + 1)
                n_overlap = len(overlap_list)

                if n_overlap:
                    segment_entities.append(ment)
        return segment_entities

    def coarse_constraint_eval(self, rel_mention, entities, relation_ontology,
                               arg_entities=None, events=None, rel_type_str=None):
        if not arg_entities:
            arg_entities = [rel_mention.retrieve_arg(a, entities, events=events) for a in rel_mention.args]
        arg_roles = [a.slot_type for a in rel_mention.args]
        try:
            arg_role_ent_types = dict(zip(arg_roles, [a_e.type for a_e in arg_entities]))
        except AttributeError as e:
            print(str(rel_mention))
            for x in arg_entities:
                print(str(x))
            raise AttributeError(e)

        if not rel_type_str:
            rel_type_str = self.full_subtype_name

        constraint_fit = relation_ontology.arg_type_constraints(
            arg_role_ent_types,
            rel_type_str,
            # self.full_subtype_name,
            test_rev=True
        )
        return constraint_fit, arg_roles, arg_entities


class PostProcessRelationRule(RelationRule):
    def __init__(self, coarse_type, subtype, subsubtypes):
        super().__init__(coarse_type, subtype, subsubtypes)
        self.tgt_subtype = ""
        self.src_subtypes = []

    def __call__(self, rel_mention, entities, relation_ontology, **kwargs):
        if not self._match_type(rel_mention.type, rel_mention.subtype):
            return rel_mention
        return self.run_rule(rel_mention, entities, relation_ontology, **kwargs)

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        raise NotImplementedError


class StandaloneRelationRule(RelationRule):
    def __init__(self, coarse_type, subtype, subsubtypes):
        super().__init__(coarse_type, subtype, subsubtypes)

    def __call__(self, segment_data, segment_dependency, entities, relation_ontology, **kwargs):
        return self.run_rule(segment_data, segment_dependency, entities, relation_ontology, **kwargs)

    def run_rule(self, segment_data, segment_dependency, entities, relation_ontology, **kwargs):
        raise NotImplementedError


# ==================================================
# Post-processing rules that operate on CS relation file.


class APORARule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="GeneralAffiliation",
            subtype="APORA",
            subsubtypes=["ControlTerritory", "NationalityCitizen", "OwnershipPossession"]
        )

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        # print(str(rel_mention))
        # print(constraint_fit)
        # input()

        arg1_role = arg_roles[0]
        arg2_role = arg_roles[1]

        for i in range(2):
            if i == 0:
                arg1_ent = arg_ents[0]
                arg2_ent = arg_ents[1]
            else:
                arg1_ent = arg_ents[1]
                arg2_ent = arg_ents[0]
            # if i == 0:
            #     arg1_ent = arg_ents[0]
            #     arg1_role = arg_roles[0]
            #
            #     arg2_ent = arg_ents[1]
            #     arg2_role = arg_roles[1]
            # else:
            #     arg1_ent = arg_ents[1]
            #     arg1_role = arg_roles[1]
            #
            #     arg2_ent = arg_ents[0]
            #     arg2_role = arg_roles[0]

            arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

            found_match = False

            if (arg1_ent.type in ["Location", "LOC", "Facility", "FAC"]
                    and arg2_ent.type in ["Side", "SID", "Person", "PER",
                                          "Organization", "ORG", "GeopoliticalEntity", "GPE"]
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[0]) > 0):
                rel_mention.subsubtype = self.subsubtypes[0]
                found_match = True

            elif (arg1_ent.type in ["Person", "PER", "Organization", "ORG"]
                    and arg2_ent.type in ["GeopoliticalEntity", "GPE"]
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[1]) > 0):
                rel_mention.subsubtype = self.subsubtypes[1]
                found_match = True

            elif (arg1_ent.type in ["Facility", "FAC", "Vehicle", "VEH", "Money", "MON"]
                    and arg2_ent.type in ["Side", "SID", "Person", "PER", "Organization", "ORG"]
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[2]) > 0):
                rel_mention.subsubtype = self.subsubtypes[2]
                found_match = True

            if found_match:
                if i == 1:
                    rel_mention.flip_args()
                break

        if not rel_mention.subsubtype and constraint_fit == 2:
            rel_mention.flip_args()

        return rel_mention


class MORERule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="GeneralAffiliation",
            subtype="MORE",
            subsubtypes=["Ethnicity", "NationalityCitizen"]
        )
        self.ethnicity_fine_types = {
            "EthnicGroup107967382",
            "EthnicMinority107967736"
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):

        # print("HERE")
        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        # print(str(rel_mention))
        # for ent in arg_ents:
        #     print(ent)

        arg1_role = arg_roles[0]
        arg2_role = arg_roles[1]
        for i in range(2):
            if i == 0:
                arg1_ent = arg_ents[0]
                arg2_ent = arg_ents[1]
            else:
                arg1_ent = arg_ents[1]
                arg2_ent = arg_ents[0]
            # if i == 0:
            #     arg1_ent = arg_ents[0]
            #     arg1_role = arg_roles[0]
            #
            #     arg2_ent = arg_ents[1]
            #     arg2_role = arg_roles[1]
            # else:
            #     arg1_ent = arg_ents[1]
            #     arg1_role = arg_roles[1]
            #
            #     arg2_ent = arg_ents[0]
            #     arg2_role = arg_roles[0]

            arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

            found_match = False

            if (arg1_ent.type in ["Person", "PER"]
                    and len(set(arg2_ent.fine_grained_types) & self.ethnicity_fine_types)
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[0]) > 0):
                rel_mention.subsubtype = self.subsubtypes[0]
                found_match = True
            elif (arg1_ent.type in ["Person", "PER", "Organization", "ORG"]
                    and arg2_ent.type in ["GeopoliticalEntity", "GPE"]
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[1]) > 0):
                rel_mention.subsubtype = self.subsubtypes[1]
                found_match = True

            if found_match:
                if i == 1:
                    rel_mention.flip_args()
                break

        if not rel_mention.subsubtype and constraint_fit == 2:
            rel_mention.flip_args()

        # print("\n AFTER: ")
        # print(rel_mention)
        # print(constraint_fit)
        #
        # input()

        return rel_mention


class OPRARule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="GeneralAffiliation",
            subtype="OPRA",
            subsubtypes=["NationalityCitizen"]
        )

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        arg1_role = arg_roles[0]
        arg2_role = arg_roles[1]

        for i in range(2):
            if i == 0:
                arg1_ent = arg_ents[0]
                arg2_ent = arg_ents[1]
            else:
                arg1_ent = arg_ents[1]
                arg2_ent = arg_ents[0]

            arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

            fine_constraint_fit = relation_ontology.arg_type_constraints(
                arg_role_ent_types,
                self.full_subsubtype_names[0]
            )

            found_match = False

            if (arg1_ent.type in ["Organization", "ORG"]
                    and arg2_ent.type in ["GeopoliticalEntity", "GPE"]
                    and fine_constraint_fit > 0):
                rel_mention.subsubtype = self.subsubtypes[0]
                found_match = True

            if found_match:
                if i == 1:
                    rel_mention.flip_args()
                break

        if not rel_mention.subsubtype and constraint_fit == 2:
            rel_mention.flip_args()

        return rel_mention


class OrganizationWebsiteRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="GeneralAffiliation",
            subtype="OrganizationWebsite",
            subsubtypes=["OrganizationWebsite"]
        )

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(
            rel_mention,
            entities,
            relation_ontology,
            rel_type_str=self.full_subsubtype_names[0]
        )

        arg1_role = arg_roles[0]
        arg2_role = arg_roles[1]

        for i in range(2):
            if i == 0:
                arg1_ent = arg_ents[0]
                arg2_ent = arg_ents[1]
            else:
                arg1_ent = arg_ents[1]
                arg2_ent = arg_ents[0]
            # if i == 0:
            #     arg1_ent = arg_ents[0]
            #     arg1_role = arg_roles[0]
            #
            #     arg2_ent = arg_ents[1]
            #     arg2_role = arg_roles[1]
            # else:
            #     arg1_ent = arg_ents[1]
            #     arg1_role = arg_roles[1]
            #
            #     arg2_ent = arg_ents[0]
            #     arg2_role = arg_roles[0]

            arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

            found_match = False

            if (rel_mention.subtype == "OrganizationWebsite"
                    and arg1_ent.type in ["Organization", "ORG"]
                    and arg2_ent.type in ["Website", "URL"]):  # Not checking arg type constraints. Correct?
                rel_mention.subsubtype = self.subsubtypes[0]
                found_match = True

            if found_match:
                if i == 1:
                    rel_mention.flip_args()
                break

        if not rel_mention.subsubtype and constraint_fit == 2:
            rel_mention.flip_args()

        return rel_mention


class MeasurementCountRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Measurement",
            subtype="",
            subsubtypes=["Count"]
        )
        self.tgt_subtype = "Size"
        self.src_subtypes = ["Count"]
        self.full_subtype_name, self.full_subsubtype_names = \
            self.create_fulltype_names(self.tgt_subtype, self.subsubtypes)

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        if rel_mention.subtype in self.src_subtypes:
            rel_mention.subtype = self.tgt_subtype

        if rel_mention.subtype != self.tgt_subtype:
            return rel_mention

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(
            rel_mention,
            entities,
            relation_ontology,
            rel_type_str=self.full_subsubtype_names[0]
        )

        if constraint_fit > 0:
            rel_mention.subsubtype = self.subsubtypes[0]

            if constraint_fit == 2:
                rel_mention.flip_args()

        return rel_mention


class EmploymentMembershipRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="OrganizationAffiliation",
            subtype="",
            subsubtypes=["Employment", "Membership"]
        )
        self.tgt_subtype = "EmploymentMembership"
        self.src_subtypes = ["InvestorShareholder", "StudentAlum", "Ownership"]
        self.full_subtype_name, self.full_subsubtype_names = \
            self.create_fulltype_names(self.tgt_subtype, self.subsubtypes)

        self.membership_fine_types = {
            'Alliance108293982',
            'Bloc108171094',
            'WorldOrganization108294696'
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        if rel_mention.subtype in self.src_subtypes:
            rel_mention.subtype = self.tgt_subtype

        if rel_mention.subtype != self.tgt_subtype:
            return rel_mention

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        # display_info = False
        # if (arg_ents[0].id in [":Entity_EDL_0018065", ":Entity_EDL_0001535"]
        #         and arg_ents[1].id in [":Entity_EDL_0018065", ":Entity_EDL_0001535"]):
        #     print(str(rel_mention))
        #     print(str(arg_ents[0]))
        #     print(str(arg_ents[1]))
        #     display_info = True

        arg1_role = arg_roles[0]
        arg2_role = arg_roles[1]

        for i in range(2):
            if i == 0:
                arg1_ent = arg_ents[0]
                arg2_ent = arg_ents[1]
            else:
                arg1_ent = arg_ents[1]
                arg2_ent = arg_ents[0]
            # if i == 0:
            #     arg1_ent = arg_ents[0]
            #     arg1_role = arg_roles[0]
            #
            #     arg2_ent = arg_ents[1]
            #     arg2_role = arg_roles[1]
            # else:
            #     arg1_ent = arg_ents[1]
            #     arg1_role = arg_roles[1]
            #
            #     arg2_ent = arg_ents[0]
            #     arg2_role = arg_roles[0]

            arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

            # if display_info:
            #     print(i)
            #     print(arg_roles)
            #     print(arg1_role)
            #     print(arg1_ent.type)
            #     print(arg2_role)
            #     print(arg2_ent.type)
            #     print(arg_role_ent_types)
            #     print("arg1 fine types: {}".format(arg1_ent.fine_grained_types))
            #     print("arg2 fine types: {}".format(arg2_ent.fine_grained_types))

            found_match = False

            # if arg1_ent.type == "Person" and arg2_ent.type in ["Organization", "GeopoliticalEntity"]:
            if relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subtype_name) > 0:
                if len(set(arg2_ent.fine_grained_types) & self.membership_fine_types):
                    rel_mention.subsubtype = self.subsubtypes[1]
                    found_match = True
                else:
                    rel_mention.subsubtype = self.subsubtypes[0]
                    found_match = True

            if found_match:
                if i == 1:
                    rel_mention.flip_args()

                # if display_info:
                #     if i == 1:
                #         print("found reverse match")
                #     else:
                #         print("found match")
                break

        if not rel_mention.subsubtype and constraint_fit == 2:
            rel_mention.flip_args()
            # if display_info:
            #     print("HERE")

        # if display_info:
        #     print(str(rel_mention))
        #     input()
        return rel_mention


class FounderRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="OrganizationAffiliation",
            subtype="Founder",
            subsubtypes=["Founder"]
        )

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        rel_mention.subsubtype = self.subsubtypes[0]

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(
            rel_mention,
            entities,
            relation_ontology,
            rel_type_str=self.full_subsubtype_names[0]
        )

        # display_info = False
        # if (arg_ents[0].id in [":Entity_EDL_0037273", ":Entity_EDL_0037271", ":Entity_EDL_0028132",
        #                        ":Entity_EDL_0028133", ":Entity_EDL_0037272", ":Entity_EDL_0037274",
        #                        ":Entity_EDL_0001506"]
        #         and arg_ents[1].id in [":Entity_EDL_0037273", ":Entity_EDL_0037271", ":Entity_EDL_0028132",
        #                        ":Entity_EDL_0028133", ":Entity_EDL_0037272", ":Entity_EDL_0037274",
        #                        ":Entity_EDL_0001506"]):
        #     print(str(rel_mention))
        #     print(str(arg_ents[0]))
        #     print(str(arg_ents[1]))
        #     print(constraint_fit)
        #     print(rel_ontology.arg_mapping)
        #     display_info = True
        #     input()

        # arg1_role = arg_roles[0]
        # arg2_role = arg_roles[1]
        #
        # for i in range(2):
        #     if i == 0:
        #         arg1_ent = arg_ents[0]
        #         arg2_ent = arg_ents[1]
        #     else:
        #         arg1_ent = arg_ents[1]
        #         arg2_ent = arg_ents[0]
        #     # if i == 0:
        #     #     arg1_ent = arg_ents[0]
        #     #     arg1_role = arg_roles[0]
        #     #
        #     #     arg2_ent = arg_ents[1]
        #     #     arg2_role = arg_roles[1]
        #     # else:
        #     #     arg1_ent = arg_ents[1]
        #     #     arg1_role = arg_roles[1]
        #     #
        #     #     arg2_ent = arg_ents[0]
        #     #     arg2_role = arg_roles[0]
        #
        #     arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}
        #
        #     found_match = False
        #
        #     if relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[0]) > 0:
        #         rel_mention.subsubtype = self.subsubtypes[0]
        #         found_match = True
        #
        #     if found_match:
        #         if i == 1:
        #             rel_mention.flip_args()
        #         break

        if constraint_fit == 2:
            rel_mention.flip_args()

        return rel_mention


class LeadershipRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="OrganizationAffiliation",
            subtype="Leadership",
            subsubtypes=["Government", "HeadOfState", "MilitaryPolice"]
        )

        self.military_police_fine_types = {
            "SecurityForce108210982",
            "LawEnforcementAgency108348815",
            "Military108199025",
            "Paramilitary108207209"
        }

        self.headofstate_fine_types = {
            "President110467179"
        }

        self.government_fine_types = {
            "ExecutiveDepartment108123167",
            "CountySeat108547143",
            "Parliament108319198"
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        arg1_role = arg_roles[0]
        arg2_role = arg_roles[1]

        for i in range(2):
            if i == 0:
                arg1_ent = arg_ents[0]
                arg2_ent = arg_ents[1]
            else:
                arg1_ent = arg_ents[1]
                arg2_ent = arg_ents[0]
            # if i == 0:
            #     arg1_ent = arg_ents[0]
            #     arg1_role = arg_roles[0]
            #
            #     arg2_ent = arg_ents[1]
            #     arg2_role = arg_roles[1]
            # else:
            #     arg1_ent = arg_ents[1]
            #     arg1_role = arg_roles[1]
            #
            #     arg2_ent = arg_ents[0]
            #     arg2_role = arg_roles[0]

            arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

            found_match = False

            if (arg1_ent.type in ["Person", "PER"]
                    and len(set(arg2_ent.fine_grained_types) & self.military_police_fine_types)
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[2]) > 0):
                rel_mention.subsubtype = self.subsubtypes[2]
                found_match = True
            elif (len(set(arg1_ent.fine_grained_types) & self.headofstate_fine_types)
                    and "Country108544813" in arg2_ent.fine_grained_types
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[1]) > 0):
                rel_mention.subsubtype = self.subsubtypes[1]
                found_match = True
            elif (arg1_ent.type in ["Person", "PER"]
                    and len(set(arg2_ent.fine_grained_types) & self.government_fine_types)
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[0]) > 0):
                rel_mention.subsubtype = self.subsubtypes[0]
                found_match = True

            if found_match:
                if i == 1:
                    rel_mention.flip_args()
                break

        if not rel_mention.subsubtype and constraint_fit == 2:
            rel_mention.flip_args()

        return rel_mention


class SubsidiaryRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="PartWhole",
            subtype="",
            subsubtypes=["NationalityCitizen", "OrganizationSubsidiary"]
        )
        self.tgt_subtype = "Subsidiary"
        self.src_subtypes = ["Membership", "Subsidiary"]
        self.full_subtype_name, self.full_subsubtype_names = \
            self.create_fulltype_names(self.tgt_subtype, self.subsubtypes)

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        if rel_mention.subtype in self.src_subtypes:
            rel_mention.subtype = self.tgt_subtype

        if rel_mention.subtype != self.tgt_subtype:
            return rel_mention

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        arg1_role = arg_roles[0]
        arg1_ent = arg_ents[0]

        arg2_role = arg_roles[1]
        arg2_ent = arg_ents[1]

        arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

        nat_cit_constraint_fit = relation_ontology.arg_type_constraints(
            arg_role_ent_types,
            self.full_subsubtype_names[0],
            test_rev=False
        )

        org_sub_constraint_fit = relation_ontology.arg_type_constraints(
            arg_role_ent_types,
            self.full_subsubtype_names[1],
            test_rev=False
        )

        if nat_cit_constraint_fit > 0:
            rel_mention.subsubtype = self.subsubtypes[0]
        elif org_sub_constraint_fit > 0:
            rel_mention.subsubtype = self.subsubtypes[1]

        return rel_mention


class RoleRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="PersonalSocial",
            subtype="",
            subsubtypes=["ProfessionalRole", "TitleFormOfAddress"]
        )
        self.tgt_subtype = "Role"
        self.src_subtypes = ["RoleTitle"]
        self.full_subtype_name, self.full_subsubtype_names = \
            self.create_fulltype_names(self.tgt_subtype, self.subsubtypes)

        self.professional_keywords = {
            "previous",
            "former",
            "next",
            "предыдущий",
            "предшествующий",
            "бывший",
            "следующий",
            "будущий"
        }

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        if rel_mention.subtype in self.src_subtypes:
            rel_mention.subtype = self.tgt_subtype

        if rel_mention.subtype != self.tgt_subtype:
            return rel_mention

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        depend_parse = kwargs.pop("dependency_parse", None)
        segment_data = kwargs.pop("segment_data", None)

        arg1_role = arg_roles[0]
        arg2_role = arg_roles[1]

        for i in range(2):
            if i == 0:
                arg1_ent = arg_ents[0]
                arg2_ent = arg_ents[1]
            else:
                arg1_ent = arg_ents[1]
                arg2_ent = arg_ents[0]
            # if i == 0:
            #     arg1_ent = arg_ents[0]
            #     arg1_role = arg_roles[0]
            #
            #     arg2_ent = arg_ents[1]
            #     arg2_role = arg_roles[1]
            # else:
            #     arg1_ent = arg_ents[1]
            #     arg1_role = arg_roles[1]
            #
            #     arg2_ent = arg_ents[0]
            #     arg2_role = arg_roles[0]

            arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

            found_match = False

            if (depend_parse
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[0]) > 0):

                arg2_start_idx = depend_parse.offsets2nodes(arg2_ent.start_offset, arg2_ent.end_offset)
                if segment_data.tokens[min(0, arg2_start_idx - 1)] in self.professional_keywords:
                    rel_mention.subsubtype = self.subsubtypes[0]
                    found_match = True

            elif (depend_parse
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[1]) > 0):

                all_dep_paths = depend_parse.get_offset_paths(
                    arg1_ent.start_offset, arg1_ent.end_offset,
                    arg2_ent.start_offset, arg2_ent.end_offset
                )
                if self.dependency_patterns[self.full_subsubtype_names[1]].match(all_dep_paths):
                    rel_mention.subsubtype = self.subsubtypes[1]
                    found_match = True

            if found_match:
                if i == 1:
                    rel_mention.flip_args()
                break

        if not rel_mention.subsubtype and constraint_fit == 2:
            rel_mention.flip_args()

        return rel_mention


class PersonalSocialUnspecifiedRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="PersonalSocial",
            subtype="",
            subsubtypes=["Political"]
        )
        self.tgt_subtype = "Unspecified"
        self.src_subtypes = ["Business", "Family", "Unspecified"]
        self.full_subtype_name, self.full_subsubtype_names = \
            self.create_fulltype_names(self.tgt_subtype, self.subsubtypes)

        self.political_keywords = {
            "allies",
            "ally"
        }

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        if rel_mention.subtype not in ["Role", "RoleTitle"] and rel_mention.subtype in self.src_subtypes:
            rel_mention.subtype = self.tgt_subtype

        if rel_mention.subtype != self.tgt_subtype:
            return rel_mention

        segment_data = kwargs.pop("segment_data", None)

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        arg1_ent = arg_ents[0]
        arg1_role = arg_roles[0]

        arg2_ent = arg_ents[1]
        arg2_role = arg_roles[1]

        arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

        if relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[0]) > 0:
            for token in segment_data.tokens:
                if token in self.political_keywords:
                    rel_mention.subsubtype = self.subsubtypes[0]

        return rel_mention


class LocatedNearRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Physical",
            subtype="",
            subsubtypes=["Surround"]
        )
        self.tgt_subtype = "LocatedNear"
        self.src_subtypes = ["OrganizationLocationOrigin", "LocatedNear"]
        self.full_subtype_name, self.full_subsubtype_names = \
            self.create_fulltype_names(self.tgt_subtype, self.subsubtypes)

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
                for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        org_loc_origin = False
        if rel_mention.subtype in self.src_subtypes:
            if rel_mention.subtype == "OrganizationLocationOrigin":
                org_loc_origin = True
            rel_mention.subtype = self.tgt_subtype

        if rel_mention.subtype != self.tgt_subtype:
            return rel_mention

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(rel_mention, entities, relation_ontology)

        if org_loc_origin:
            if constraint_fit == 2:
                rel_mention.flip_args()
            return rel_mention

        depend_parse = kwargs.pop("dependency_parse", None)
        segment_data = kwargs.pop("segment_data", None)

        arg1_role = arg_roles[0]
        arg2_role = arg_roles[1]

        for i in range(2):
            if i == 0:
                arg1_ent = arg_ents[0]
                arg2_ent = arg_ents[1]
            else:
                arg1_ent = arg_ents[1]
                arg2_ent = arg_ents[0]
            # if i == 0:
            #     arg1_ent = arg_ents[0]
            #     arg1_role = arg_roles[0]
            #
            #     arg2_ent = arg_ents[1]
            #     arg2_role = arg_roles[1]
            # else:
            #     arg1_ent = arg_ents[1]
            #     arg1_role = arg_roles[1]
            #
            #     arg2_ent = arg_ents[0]
            #     arg2_role = arg_roles[0]

            arg_role_ent_types = {arg1_role: arg1_ent.type, arg2_role: arg2_ent.type}

            found_match = False

            if (depend_parse
                    and relation_ontology.arg_type_constraints(arg_role_ent_types, self.full_subsubtype_names[0]) > 0):
                all_dep_paths = depend_parse.get_offset_paths(
                    arg1_ent.start_offset, arg1_ent.end_offset,
                    arg2_ent.start_offset, arg2_ent.end_offset
                )

                if all(all_dep_paths):
                    for dep_pat in self.dependency_patterns[self.full_subsubtype_names[0]]:
                        if dep_pat.match(all_dep_paths):
                            rel_mention.subsubtype = self.subsubtypes[0]
                            found_match = True
                            break

            if found_match:
                if i == 1:
                    rel_mention.flip_args()
                break

        if not rel_mention.subsubtype and constraint_fit == 2:
            rel_mention.flip_args()

        return rel_mention


class OrganizationHeadquartersRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Physical",
            subtype="",
            subsubtypes=["OrganizationHeadquarters"]
        )
        self.tgt_subtype = "OrganizationHeadquarters"
        self.src_subtypes = ["OrganizationHeadquarter", "OrganizationHeadquarters"]
        self.full_subtype_name, self.full_subsubtype_names = \
            self.create_fulltype_names(self.tgt_subtype, self.subsubtypes)

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        if rel_mention.subtype in self.src_subtypes:
            rel_mention.subtype = self.tgt_subtype
            rel_mention.subsubtype = self.subsubtypes[0]

        if rel_mention.subtype != self.tgt_subtype:
            return rel_mention

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(
            rel_mention,
            entities,
            relation_ontology,
            rel_type_str=self.full_subsubtype_names[0]
        )

        if constraint_fit == 2:
            rel_mention.flip_args()

        # if constraint_fit > 0:
        #     rel_mention.subsubtype = self.subsubtypes[0]
        #     if constraint_fit == 2:
        #         rel_mention.flip_args()

        return rel_mention


class ResidentRule(PostProcessRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Physical",
            subtype="Resident",
            subsubtypes=["Resident"]
        )

    def run_rule(self, rel_mention, entities, relation_ontology, **kwargs):
        rel_mention.subsubtype = self.subsubtypes[0]

        constraint_fit, arg_roles, arg_ents = self.coarse_constraint_eval(
            rel_mention,
            entities,
            relation_ontology,
            rel_type_str=self.full_subsubtype_names[0]
        )

        # display_info = False
        # if (arg_ents[0].id in [":Entity_EDL_0018428", ":Entity_EDL_0015666"]
        #         and arg_ents[1].id in [":Entity_EDL_0018428", ":Entity_EDL_0015666"]):
        #     print(str(rel_mention))
        #     print(str(arg_ents[0]))
        #     print(str(arg_ents[1]))
        #     print(constraint_fit)
        #     display_info = True
        #
        # input()

        if constraint_fit == 2:
            rel_mention.flip_args()

        return rel_mention


# ==================================================


# ==================================================
# Standalone rules that operate on ltf documents, not CS relation file.


class DeliberatenessRule(StandaloneRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Evaluate",
            subtype="Deliberateness",
            subsubtypes=["Accidental", "Deliberate"]
        )
        self.bad_edge_types = ["nmod", "nnmod:poss", "obl", "nmod:on", "nmod:in"]
        self.good_edge_types = ["ccomp", "xcomp", "nsubj"]

        self.deliberate_keywords = {
            "deliberately",
            "deliberate",
            "intentionally",
            "intentional",
            "conscious",
            "consciously",
            "intended",
            "willfully",
            "orchestrate",
            "orchestrated"
        }
        self.accidental_keywords = {
            "accidentally",
            "accidental",
            "coincidentally"
        }

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def _sort_by_entity_type(self, mention):
        if mention.type in ["PER", "Person"]:
            return 0
        elif mention.type in ["SID", "Side"]:
            return 1
        elif mention.type in ["ORG", "Organization"]:
            return 2
        elif mention.type in ["GPE", "GeopoliticalEntity"]:
            return 3
        else:
            return 4

    def _find_match(self, entity, event, segment_data, segment_dependency, kw_locs, relation_ontology,
                       cand_type, rel_id, lang_id):
        all_dep_paths = segment_dependency.get_offset_paths(
            entity.start_offset, entity.end_offset,
            event.start_offset, event.end_offset
        )
        if not any(all_dep_paths):
            return None

        arg_role_ent_types = {"arg1": entity.type, "arg2": KE_EVT}

        fulltype_name = cand_type
        fulltype_patterns = self.dependency_patterns[fulltype_name]

        # for fulltype_name, fulltype_patterns in self.dependency_patterns.items():
        if relation_ontology.arg_type_constraints(arg_role_ent_types, fulltype_name) > 0:
            for dep_pat in fulltype_patterns:
                if (not dep_pat.match(all_dep_paths)
                        and dep_pat.accept_path(all_dep_paths, bad_edge_type_list=self.bad_edge_types)):
                    for kw, kw_loc in kw_locs.items():
                        ent2kw_paths = segment_dependency.get_offset_paths(
                            entity.start_offset, entity.end_offset,
                            kw_loc[0], kw_loc[1]
                        )
                        if dep_pat.accept_path(ent2kw_paths, good_edge_type_list=self.good_edge_types):
                            ment_id = form_cs_id(
                                rel_id,
                                ke_label=KE_REL,
                                lang_label=lang_id
                            )

                            rel_mention = RelationMention()
                            rel_mention.id = ment_id
                            rel_mention.mention_id = ment_id
                            rel_mention.update_full_type(cand_type)
                            rel_mention.provenance = segment_data.provenance
                            rel_mention.start_offset = segment_data.start_offset
                            rel_mention.end_offset = segment_data.end_offset
                            rel_mention.confidence = "0.90"

                            arg1 = ArgumentMention()
                            arg1.mention_id = ment_id
                            arg1.slot_type = "arg1"
                            arg1.arg_id = entity.id
                            rel_mention.add_argument(arg1)

                            arg2 = ArgumentMention()
                            arg2.mention_id = ment_id
                            arg2.slot_type = "arg2"
                            arg2.arg_id = event.id
                            rel_mention.add_argument(arg2)
                            return rel_mention
        return None

    def run_rule(self, segment_data, segment_dependency, entities, relation_ontology, **kwargs):
        events = kwargs.pop("events", [])
        rel_id_offset = kwargs.pop("rel_id_offset", 0)
        lang_id = kwargs.pop("lang_id", None)

        results = {KE_ENT: [], KE_REL: []}

        if segment_dependency is None:
            return results

        kw_locs = self.find_keyword_in_segment(segment_data, self.deliberate_keywords)
        cand_type = KnowledgeElementMention.form_full_type(self.coarse_type, self.subtype, self.subsubtypes[1])
        if not len(kw_locs):
            kw_locs = self.find_keyword_in_segment(segment_data, self.accidental_keywords)
            cand_type = KnowledgeElementMention.form_full_type(self.coarse_type, self.subtype, self.subsubtypes[0])
            if not len(kw_locs):
                return None

        segment_entities = self.retrieve_segment_mentions(
            segment_data.provenance,
            segment_data.start_offset,
            segment_data.end_offset,
            entities
        )
        segment_entities = sorted(segment_entities, key=self._sort_by_entity_type)

        segment_events = self.retrieve_segment_mentions(
            segment_data.provenance,
            segment_data.start_offset,
            segment_data.end_offset,
            events
        )

        relations = []
        prev_matched = set()
        for event in segment_events:
            for i, entity in enumerate(segment_entities):
                if i not in prev_matched:
                    rel_mention = self._find_match(
                        entity,
                        event,
                        segment_data,
                        segment_dependency,
                        kw_locs,
                        relation_ontology,
                        cand_type,
                        rel_id_offset + len(relations),
                        lang_id
                    )

                    if rel_mention is not None:
                        relations.append(rel_mention)
                        prev_matched.add(i)
        results[KE_REL] = relations
        return results


class LegitimacyRule(StandaloneRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Evaluate",
            subtype="Legitimacy",
            subsubtypes=["Illegitimate", "Legitimate"]
        )
        self.bad_edge_types = ["nmod", "nnmod:poss", "obl", "nmod:on", "nmod:in"]
        self.good_edge_types = ["ccomp", "xcomp", "nsubj"]

        self.valid_event_types = {
            "Personnel",
            "Transaction",
            "Justice",
            "Inspection",
            "Government",
            # "Contact",
            "ArtifactExistence"
        }

        self.illegitimate_keywords = {
            "illegal",
            "illegally",
            "invalid",
            "sham",
            "cheat",
            "cheated",
            "flaws",
            "нелегитимным",
            "незаконно",
            "не визнає"
        }
        self.legitimate_keywords = {
            "legitimate",
            "valid",
            "respected",
            "respect",
            "виправданий"
        }

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def _sort_by_entity_type(self, mention):
        if mention.type in ["PER", "Person"]:
            return 0
        elif mention.type in ["SID", "Side"]:
            return 1
        elif mention.type in ["ORG", "Organization"]:
            return 2
        elif mention.type in ["GPE", "GeopoliticalEntity"]:
            return 3
        else:
            return 4

    def _find_match(self, entity, event, segment_data, segment_dependency, kw_locs, relation_ontology,
                       cand_type, rel_id, lang_id):
        all_dep_paths = segment_dependency.get_offset_paths(
            entity.start_offset, entity.end_offset,
            event.start_offset, event.end_offset
        )
        if not any(all_dep_paths):
            return None

        arg_role_ent_types = {"arg1": entity.type, "arg2": KE_EVT}
        fulltype_name = cand_type
        fulltype_patterns = self.dependency_patterns[fulltype_name]

        # for fulltype_name, fulltype_patterns in self.dependency_patterns.items():
        if relation_ontology.arg_type_constraints(arg_role_ent_types, fulltype_name) > 0:
            for dep_pat in fulltype_patterns:
                if (not dep_pat.match(all_dep_paths)
                        and dep_pat.accept_path(all_dep_paths, bad_edge_type_list=self.bad_edge_types)):
                    for kw, kw_loc in kw_locs.items():
                        ent2kw_paths = segment_dependency.get_offset_paths(
                            entity.start_offset, entity.end_offset,
                            kw_loc[0], kw_loc[1]
                        )
                        if dep_pat.accept_path(ent2kw_paths, good_edge_type_list=self.good_edge_types):
                            ment_id = form_cs_id(
                                rel_id,
                                ke_label=KE_REL,
                                lang_label=lang_id
                            )

                            rel_mention = RelationMention()
                            rel_mention.id = ment_id
                            rel_mention.mention_id = ment_id
                            rel_mention.update_full_type(cand_type)
                            rel_mention.provenance = segment_data.provenance
                            rel_mention.start_offset = segment_data.start_offset
                            rel_mention.end_offset = segment_data.end_offset
                            rel_mention.confidence = "0.90"

                            arg1 = ArgumentMention()
                            arg1.mention_id = ment_id
                            arg1.slot_type = "arg1"
                            arg1.arg_id = entity.id
                            rel_mention.add_argument(arg1)

                            arg2 = ArgumentMention()
                            arg2.mention_id = ment_id
                            arg2.slot_type = "arg2"
                            arg2.arg_id = event.id
                            rel_mention.add_argument(arg2)
                            return rel_mention
        return None

    def run_rule(self, segment_data, segment_dependency, entities, relation_ontology, **kwargs):
        events = kwargs.pop("events", [])
        rel_id_offset = kwargs.pop("rel_id_offset", 0)
        lang_id = kwargs.pop("lang_id", None)

        results = {KE_ENT: [], KE_REL: []}

        if segment_dependency is None:
            return results

        kw_locs = self.find_keyword_in_segment(segment_data, self.illegitimate_keywords)
        cand_type = KnowledgeElementMention.form_full_type(self.coarse_type, self.subtype, self.subsubtypes[0])
        if not len(kw_locs):
            kw_locs = self.find_keyword_in_segment(segment_data, self.legitimate_keywords)
            cand_type = KnowledgeElementMention.form_full_type(self.coarse_type, self.subtype, self.subsubtypes[1])
            if not len(kw_locs):
                return None

        segment_entities = self.retrieve_segment_mentions(
            segment_data.provenance,
            segment_data.start_offset,
            segment_data.end_offset,
            entities
        )
        segment_entities = sorted(segment_entities, key=self._sort_by_entity_type)

        segment_events = self.retrieve_segment_mentions(
            segment_data.provenance,
            segment_data.start_offset,
            segment_data.end_offset,
            events
        )

        relations = []
        prev_matched = set()
        for event in segment_events:
            if event.type not in self.valid_event_types:
                continue

            for i, entity in enumerate(segment_entities):
                if i not in prev_matched:
                    rel_mention = self._find_match(
                        entity,
                        event,
                        segment_data,
                        segment_dependency,
                        kw_locs,
                        relation_ontology,
                        cand_type,
                        rel_id_offset + len(relations),
                        lang_id
                    )

                    if rel_mention is not None:
                        relations.append(rel_mention)
                        prev_matched.add(i)
        results[KE_REL] = relations
        return results


class ColorRule(StandaloneRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Information",
            subtype="Color",
            subsubtypes=["Color"]
        )

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def run_rule(self, segment_data, segment_dependency, entities, relation_ontology, **kwargs):
        rel_id_offset = kwargs.pop("rel_id_offset", 0)
        lang_id = kwargs.pop("lang_id", None)

        results = {KE_ENT: [], KE_REL: []}

        if segment_dependency is None:
            return results

        segment_entities = self.retrieve_segment_mentions(
            segment_data.provenance,
            segment_data.start_offset,
            segment_data.end_offset,
            entities
        )

        segment_entities = [ent for ent in segment_entities if ent.type not in ["VAL", "NumericalValue"]]

        fulltype_name = self.full_subsubtype_names[0]

        relations = []
        prev_matched = set()
        for i, arg1_ent in enumerate(segment_entities):
            for j, arg2_ent in enumerate(segment_entities):
                temp_pair = "|".join([arg1_ent.get_provenance_offset_str(), arg2_ent.get_provenance_offset_str()])
                rev_temp_pair = "|".join([arg2_ent.get_provenance_offset_str(), arg1_ent.get_provenance_offset_str()])

                if i == j or temp_pair in prev_matched or rev_temp_pair in prev_matched:
                    continue

                arg_role_types = {"arg1": arg1_ent.type, "arg2": arg2_ent.type}

                if (relation_ontology.arg_type_constraints(arg_role_types, fulltype_name) > 0):

                    all_dep_paths = segment_dependency.get_offset_paths(
                        arg1_ent.start_offset, arg1_ent.end_offset,
                        arg2_ent.start_offset, arg2_ent.end_offset
                    )

                    for dep_pat in self.dependency_patterns[fulltype_name]:
                        if dep_pat.match(all_dep_paths):
                            ment_id = form_cs_id(
                                rel_id_offset + len(relations),
                                ke_label=KE_REL,
                                lang_label=lang_id
                            )

                            rel_mention = RelationMention()
                            rel_mention.id = ment_id
                            rel_mention.mention_id = ment_id
                            rel_mention.update_full_type(fulltype_name)
                            rel_mention.provenance = segment_data.provenance
                            rel_mention.start_offset = segment_data.start_offset
                            rel_mention.end_offset = segment_data.end_offset
                            rel_mention.confidence = "0.90"

                            arg1 = ArgumentMention()
                            arg1.mention_id = ment_id
                            arg1.slot_type = "arg1"
                            arg1.arg_id = arg1_ent.id
                            rel_mention.add_argument(arg1)

                            arg2 = ArgumentMention()
                            arg2.mention_id = ment_id
                            arg2.slot_type = "arg2"
                            arg2.arg_id = arg2_ent.id
                            rel_mention.add_argument(arg2)

                            relations.append(rel_mention)
                            prev_matched.add(temp_pair)
                            prev_matched.add(rev_temp_pair)
                            break
        results[KE_REL] = relations
        return results


class MakeRule(StandaloneRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Information",
            subtype="Make",
            subsubtypes=["Make"]
        )

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def run_rule(self, segment_data, segment_dependency, entities, relation_ontology, **kwargs):
        rel_id_offset = kwargs.pop("rel_id_offset", 0)
        lang_id = kwargs.pop("lang_id", None)

        results = {KE_ENT: [], KE_REL: []}

        if segment_dependency is None:
            return results

        segment_entities = self.retrieve_segment_mentions(
            segment_data.provenance,
            segment_data.start_offset,
            segment_data.end_offset,
            entities
        )

        segment_entities = [ent for ent in segment_entities if ent.type not in ["VAL", "NumericalValue", "TME", "Time"]]

        fulltype_name = self.full_subsubtype_names[0]

        relations = []
        prev_matched = set()
        for i, arg1_ent in enumerate(segment_entities):
            for j, arg2_ent in enumerate(segment_entities):

                temp_pair = "|".join([arg1_ent.get_provenance_offset_str(), arg2_ent.get_provenance_offset_str()])
                rev_temp_pair = "|".join([arg2_ent.get_provenance_offset_str(), arg1_ent.get_provenance_offset_str()])

                if i == j or temp_pair in prev_matched or rev_temp_pair in prev_matched:
                    continue

                arg1_ent_idx = segment_dependency.offsets2nodes(arg1_ent.start_offset, arg1_ent.end_offset)
                arg2_ent_idx = segment_dependency.offsets2nodes(arg2_ent.start_offset, arg2_ent.end_offset)

                try:
                    if arg1_ent.start_offset < arg2_ent.start_offset:
                        w_dist = arg2_ent_idx[0] - arg1_ent_idx[-1]
                    else:
                        w_dist = arg1_ent_idx[0] - arg2_ent_idx[-1]
                except IndexError:
                    # print("Error finding the following word indices:")
                    # print("ARG1:\n {}".format(arg1_ent))
                    # print("ARG2:\n {}".format(arg2_ent))
                    # print(segment_data)
                    # print(segment_dependency)
                    # print("\n\n")
                    w_dist = 10000

                if w_dist > 2:
                    continue

                arg_role_types = {"arg1": arg1_ent.type, "arg2": arg2_ent.type}

                if relation_ontology.arg_type_constraints(arg_role_types, fulltype_name) > 0:
                    all_dep_paths = segment_dependency.get_offset_paths(
                        arg1_ent.start_offset, arg1_ent.end_offset,
                        arg2_ent.start_offset, arg2_ent.end_offset
                    )

                    for dep_pat in self.dependency_patterns[fulltype_name]:
                        if dep_pat.match(all_dep_paths):
                            ment_id = form_cs_id(
                                rel_id_offset + len(relations),
                                ke_label=KE_REL,
                                lang_label=lang_id
                            )

                            rel_mention = RelationMention()
                            rel_mention.id = ment_id
                            rel_mention.mention_id = ment_id
                            rel_mention.update_full_type(fulltype_name)
                            rel_mention.provenance = segment_data.provenance
                            rel_mention.start_offset = segment_data.start_offset
                            rel_mention.end_offset = segment_data.end_offset
                            rel_mention.confidence = "0.90"

                            arg1 = ArgumentMention()
                            arg1.mention_id = ment_id
                            arg1.slot_type = "arg1"
                            arg1.arg_id = arg1_ent.id
                            rel_mention.add_argument(arg1)

                            arg2 = ArgumentMention()
                            arg2.mention_id = ment_id
                            arg2.slot_type = "arg2"
                            arg2.arg_id = arg2_ent.id
                            rel_mention.add_argument(arg2)

                            relations.append(rel_mention)
                            prev_matched.add(temp_pair)
                            prev_matched.add(rev_temp_pair)
                            break
        results[KE_REL] = relations
        return results


class MeasurementSizeRule(StandaloneRelationRule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            coarse_type="Measurement",
            subtype="Size",
            subsubtypes=["HeightLengthWidth", "Percentage", "Weight"]
        )

        self.numeric_ent_type = "VAL"

        self.height_width_keywords = {
            "meters",
            "meter",
            "inches",
            "centimeters",
            "feet",
            "foot",
            "miles",
            "kilometer",
            "kilometre",
            "метрів",
            "километров",
            "километр"
        }
        self.percentage_keywords = {
            "%", "percent", "per cent", "percentage", "процентов", "відсотків"
        }
        self.weight_keywords = {
            "pounds", "tons", "grams", "kilograms", "milligrams", "pound", "ton", "gram", "kilogram", "milligram", "kg",
        "mg", "lbs", "g", "фунтов", "кілограмів", "кг", "килограммов", "грамм", "грам", "міліграмів", "миллиграмм",
        "мг", "фунтів", "тонн"
        }

        self.finetype_kw_pairs = list(
            zip(
                self.full_subsubtype_names + [self.full_subtype_name],
                [self.height_width_keywords, self.percentage_keywords, self.weight_keywords, {}]
            )
        )

        temp_all_patterns = kwargs.get("dependency_patterns", {})
        self.dependency_patterns = {
            ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
            for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
        }

    def _find_match(self, arg1_entity, arg2_entity, segment_data, segment_dependency, keywords, relation_ontology,
                       cand_type, rel_id, lang_id):
        if len(keywords):
            found_kw = False
            for kw in keywords:
                if kw in arg1_entity.text_string:
                    found_kw = True
            if not found_kw:
                return None

        all_dep_paths = segment_dependency.get_offset_paths(
            arg1_entity.start_offset, arg1_entity.end_offset,
            arg2_entity.start_offset, arg2_entity.end_offset
        )

        arg_role_ent_types = {"arg1": arg1_entity.type, "arg2": arg2_entity.type}
        if relation_ontology.arg_type_constraints(arg_role_ent_types, cand_type) > 0:
            fulltype_patterns = self.dependency_patterns.get(cand_type, [])
            for dep_pat in fulltype_patterns:
                if dep_pat.match(all_dep_paths):
                    ment_id = form_cs_id(
                        rel_id,
                        ke_label=KE_REL,
                        lang_label=lang_id
                    )

                    rel_mention = RelationMention()
                    rel_mention.id = ment_id
                    rel_mention.mention_id = ment_id
                    rel_mention.update_full_type(cand_type)
                    rel_mention.provenance = segment_data.provenance
                    rel_mention.start_offset = segment_data.start_offset
                    rel_mention.end_offset = segment_data.end_offset
                    rel_mention.confidence = "0.90"

                    arg1 = ArgumentMention()
                    arg1.mention_id = ment_id
                    arg1.slot_type = "arg1"
                    arg1.arg_id = arg1_entity.id
                    rel_mention.add_argument(arg1)

                    arg2 = ArgumentMention()
                    arg2.mention_id = ment_id
                    arg2.slot_type = "arg2"
                    arg2.arg_id = arg2_entity.id
                    rel_mention.add_argument(arg2)
                    return rel_mention
        return None

    def run_rule(self, segment_data, segment_dependency, entities, relation_ontology, **kwargs):
        # Check out https://github.com/NEO-IE/Numberule
        results = {KE_ENT: [], KE_REL: []}

        if segment_dependency is None:
            return results

        rel_id_offset = kwargs.pop("rel_id_offset", 0)
        lang_id = kwargs.pop("lang_id", None)

        segment_entities = self.retrieve_segment_mentions(
            segment_data.provenance,
            segment_data.start_offset,
            segment_data.end_offset,
            entities
        )

        numerical_entities = []
        other_entities = []
        for ent in segment_entities:
            if ent.type == self.numeric_ent_type:
                numerical_entities.append(ent)
            else:
                other_entities.append(ent)

        relations = []
        prev_matched = set()
        for num_ent in numerical_entities:
            for i, seg_ent in enumerate(other_entities):
                if i in prev_matched:
                    continue

                for cand_type, cand_kw in self.finetype_kw_pairs:
                    rel_mention = self._find_match(
                        num_ent,
                        seg_ent,
                        segment_data,
                        segment_dependency,
                        cand_kw,
                        relation_ontology,
                        cand_type,
                        rel_id_offset + len(relations),
                        lang_id
                    )

                    if rel_mention is not None:
                        relations.append(rel_mention)
                        prev_matched.add(i)
                        break
        results[KE_REL] = relations
        return results
# class MeasurementSizeRule(StandaloneRelationRule):
#     def __init__(self, *args, **kwargs):
#         super().__init__(
#             coarse_type="Measurement",
#             subtype="Size",
#             subsubtypes=["HeightLengthWidth", "Percentage", "Weight"]
#         )
#
#         self.numeric_ent_type = "VAL"
#
#         self.height_width_keywords = {
#             "meters",
#             "meter",
#             "inches",
#             "centimeters",
#             "feet",
#             "foot",
#             "miles",
#             "kilometer",
#             "kilometre",
#             "метрів",
#             "километров",
#             "километр"
#         }
#         self.percentage_keywords = {
#             "%", "percent", "per cent", "percentage", "процентов", "відсотків"
#         }
#         self.weight_keywords = {
#             "pounds", "tons", "grams", "kilograms", "milligrams", "pound", "ton", "gram", "kilogram", "milligram", "kg",
#         "mg", "lbs", "g", "фунтов", "кілограмів", "кг", "килограммов", "грамм", "грам", "міліграмів", "миллиграмм",
#         "мг", "фунтів", "тонн"
#         }
#
#         temp_all_patterns = kwargs.get("dependency_patterns", {})
#         self.dependency_patterns = {
#             ft_name: [DependencyPattern(p) for p in temp_all_patterns[ft_name]]
#             for ft_name in [self.full_subtype_name] + self.full_subsubtype_names if ft_name in temp_all_patterns
#         }
#
#     def run_rule(self, segment_data, segment_dependency, entities, relation_ontology, **kwargs):
#         # Check out https://github.com/NEO-IE/Numberule
#         results = {KE_ENT: [], KE_REL: []}
#
#         if segment_dependency is None:
#             return results
#
#         rel_id_offset = kwargs.pop("rel_id_offset", 0)
#         lang_id = kwargs.pop("lang_id", None)
#
#         segment_entities = self.retrieve_segment_mentions(
#             segment_data.provenance,
#             segment_data.start_offset,
#             segment_data.end_offset,
#             entities
#         )
#
#         numerical_entities = []
#         other_entities = []
#         for ent in segment_entities:
#             if ent.type == self.numeric_ent_type:
#                 numerical_entities.append(ent)
#             else:
#                 other_entities.append(ent)
#
#         relations = []
#         prev_matched = set()
#         for num_ent in numerical_entities:
#             for i, seg_ent in enumerate(other_entities):
#
#                 found_rel = False
#                 if i not in prev_matched:
#                     all_dep_paths = segment_dependency.get_offset_paths(
#                         num_ent.start_offset, num_ent.end_offset,
#                         seg_ent.start_offset, seg_ent.end_offset
#                     )
#
#                     arg_role_ent_types = {"arg1": num_ent.type, "arg2": seg_ent.type}
#
#                     for fulltype_name, fulltype_patterns in self.dependency_patterns.items():
#                         if relation_ontology.arg_type_constraints(arg_role_ent_types, fulltype_name) > 0:
#                             for dep_pat in fulltype_patterns:
#                                 if dep_pat.match(all_dep_paths):
#                                     ment_id = form_cs_id(
#                                         rel_id_offset + len(relations),
#                                         ke_label=KE_REL,
#                                         lang_label=lang_id
#                                     )
#
#                                     rel_mention = RelationMention()
#                                     rel_mention.id = ment_id
#                                     rel_mention.mention_id = ment_id
#                                     rel_mention.update_full_type(fulltype_name)
#                                     rel_mention.provenance = segment_data.provenance
#                                     rel_mention.start_offset = segment_data.start_offset
#                                     rel_mention.end_offset = segment_data.end_offset
#                                     rel_mention.confidence = "0.90"
#
#                                     arg1 = ArgumentMention()
#                                     arg1.mention_id = ment_id
#                                     arg1.slot_type = "arg1"
#                                     arg1.arg_id = num_ent.id
#                                     rel_mention.add_argument(arg1)
#
#                                     arg2 = ArgumentMention()
#                                     arg2.mention_id = ment_id
#                                     arg2.slot_type = "arg2"
#                                     arg2.arg_id = seg_ent.id
#                                     rel_mention.add_argument(arg2)
#
#                                     relations.append(rel_mention)
#                                     prev_matched.add(i)
#                                     found_rel = True
#                                     break
#                         if found_rel:
#                             break
#                     if found_rel:
#                         break
#         results[KE_REL] = relations
#         return results


# ==================================================


def collect_lang_dependency_parses(dp_util, ltf_util):
    if dp_util.reuse_cache:
        return dp_util

    prov_offset_strs = []
    segments = []
    for seg_data in ltf_util.get_all_segments():
        prov_offset_strs.append(
            KnowledgeElementMention.form_provenance_offset_str(
                seg_data.provenance,
                seg_data.start_offset,
                seg_data.end_offset
            )
        )
        segments.append(seg_data.tokens)
    dp_util.run_parser(prov_offset_strs, segments)
    return dp_util


def collect_dependency_parses(depend_utils, ltf_utils):
    for lang_id, lang_ltf_util in ltf_utils.items():
        lang_dep_start_t = time.time()
        print("{} := Running dependency parsing.".format(lang_id))

        depend_utils[lang_id] = collect_lang_dependency_parses(depend_utils[lang_id], lang_ltf_util)

        print("\tDependency parse time: {} seconds\n".format(time.time() - lang_dep_start_t))
    return depend_utils


def run_postprocess_rules(relations, entities, rel_ontology, depend_util, dependency_patterns, ltf_util, lang_id):
    rule_list = [c(dependency_patterns=dependency_patterns) for c in PostProcessRelationRule.__subclasses__()]
    curr_type_dist = Counter(dict(zip(rel_ontology.get_types(), [0] * len(rel_ontology.get_types()))))
    updated_relations = OrderedDict()

    discarded_relations = OrderedDict()

    total_relations = 0
    total_discarded = 0
    total_updated = 0

    for rel_ment_id, rel_mentions in relations.items():
        updated_rel_mentions = []
        discarded_rel_mentions = []

        total_relations += len(rel_mentions)

        for j, rel_mention in enumerate(rel_mentions):
            updated_rel_ment = rel_mention

            seg_data = ltf_util.get_spec_doc_segment(
                rel_mention.provenance,
                rel_mention.start_offset,
                rel_mention.end_offset
            )

            for rule in rule_list:
                if not updated_rel_ment:
                    break

                updated_rel_ment = rule(
                    updated_rel_ment,
                    entities,
                    rel_ontology,
                    segment_data=seg_data,
                    dependency_parse=depend_util.get_dependency_parse(rel_mention.get_provenance_offset_str(), seg_data),
                    lang_id=lang_id
                    # all_mentions=relations
                )

            if updated_rel_ment:
                arg_entities = [rel_mention.retrieve_arg(a, entities) for a in rel_mention.args]
                arg_roles = [a.slot_type for a in rel_mention.args]
                constraint_fit = rel_ontology.arg_type_constraints(
                    dict(zip(arg_roles, [a_e.type for a_e in arg_entities])),
                    updated_rel_ment.get_full_type(),
                    test_rev=True
                )
                if constraint_fit > 0:
                    if constraint_fit == 2:
                        updated_rel_ment.flip_args()
                    updated_rel_ment = rel_ontology.adjust(updated_rel_ment)
                    updated_rel_mentions.append(updated_rel_ment)
                    curr_type_dist[updated_rel_ment.get_full_type()] += 1
                else:
                    print("Relation=BadConstraints={}".format(constraint_fit))
                    print(rel_mentions[j])
                    print("\n")
                    discarded_rel_mentions.append(rel_mentions[j])

            elif rel_mention:
                print("Relation=None")
                print(rel_mentions[j])
                print("\n")
                discarded_rel_mentions.append(rel_mentions[j])

        if len(updated_rel_mentions) > 0:
            updated_relations[rel_ment_id] = updated_rel_mentions
            total_updated += len(updated_rel_mentions)

        if len(discarded_rel_mentions) > 0:
            discarded_relations[rel_ment_id] = discarded_rel_mentions
            total_discarded += len(discarded_rel_mentions)

    assert total_relations == (total_updated + total_discarded), \
        "WARNING: Relations were dropped along the way!" + \
        "\ttotal_relations={}\n\ttotal_updated={}\n\ttotal_discarded={}\n".format(
            total_relations, total_updated, total_discarded
        )

    # if total_relations != (total_updated + total_discarded):
    #     print("WARNING: Relations were dropped along the way!")
    #     print("\ttotal_relations={}\n\ttotal_updated={}\n\ttotal_discarded={}\n".format(
    #         total_relations, total_updated, total_discarded)
    #     )

    return updated_relations, curr_type_dist, discarded_relations
    #     if len(updated_rel_mentions) == 0:
    #         del relations[rel_ment_id]
    #     else:
    #         relations[rel_ment_id] = updated_rel_mentions
    # return relations, curr_type_dist


def run_standalone_rules(
        doc_entities,
        doc_events,
        rel_ontology,
        ltf_util,
        depend_util,
        dependency_patterns,
        lang_id,
        id_offsets=None):
    if not id_offsets:
        id_offsets = {KE_ENT: 0, KE_REL: 0}

    standalone_results = {KE_ENT: {}, KE_REL: {}}
    rule_list = [c(dependency_patterns=dependency_patterns) for c in StandaloneRelationRule.__subclasses__()]
    curr_type_dist = Counter(dict(zip(rel_ontology.get_types(), [0] * len(rel_ontology.get_types()))))
    ere_counts = Counter(id_offsets)

    for seg_data in ltf_util.get_all_segments():
        seg_dep_parse = depend_util.get_dependency_parse(
            KnowledgeElementMention.form_provenance_offset_str(
                seg_data.provenance, seg_data.start_offset, seg_data.end_offset),
            seg_data
        )

        for rule in rule_list:
            rule_results = rule(
                seg_data,
                seg_dep_parse,
                doc_entities.get(seg_data.provenance, []),
                rel_ontology,
                events=doc_events.get(seg_data.provenance, []),
                lang_id=lang_id,
                rel_id_offset=ere_counts[KE_REL]
                # rel_id_offset=id_offsets[KE_REL] + ere_counts[KE_REL]
            )

            if rule_results:
                for ere in [KE_ENT, KE_REL]:
                    for pred_y in rule_results[ere]:
                        # pred_y.id = form_cs_id(ere_counts[ere] + id_offsets[ere], ke_label=ere, lang_label=lang_id)

                        if ere == KE_ENT:
                            curr_doc_entities = doc_entities.get(pred_y.provenance, [])
                            curr_doc_entities.append(pred_y)
                            doc_entities[pred_y.provenance] = curr_doc_entities
                        elif ere == KE_REL:
                            pred_y = rel_ontology.adjust(pred_y)
                            curr_type_dist[pred_y.get_full_type()] += 1

                        ere_standalone_results = standalone_results[ere].get(pred_y.id, [])
                        ere_standalone_results.append(pred_y)
                        standalone_results[ere][pred_y.id] = ere_standalone_results
                        ere_counts[ere] += 1

    return standalone_results, curr_type_dist


def safe_standalone_postprocess_merge(lang_id, ke_label, lang_postprocess_results, lang_standalone_results):
    if ke_label == KE_ENT:
        print("Warning performing merging on entities will not properly change the entity IDs in relations or events.")

    lang_merged_results = {}
    num_mentions = 0
    for ke_id, ke_ments in lang_postprocess_results.items():
        num_mentions += len(ke_ments)
        lang_merged_results[ke_id] = ke_ments

    current_count = int(num_mentions)
    for ke_id, ke_ments in lang_standalone_results.items():
        if ke_id in lang_merged_results:
            has_collision = True
            while has_collision:
                new_ke_id = form_cs_id(current_count, ke_label, lang_id)
                if new_ke_id not in lang_merged_results:
                    has_collision = False
                    for i, ment in enumerate(ke_ments):
                        ment.id = new_ke_id
                        ment.mention_id = new_ke_id
                        if ke_label in [KE_REL, KE_EVT]:
                            for j, a in enumerate(ment.args):
                                a.mention_id = new_ke_id
                                ment.args[j] = a
                        ke_ments[i] = ment
                    lang_merged_results[new_ke_id] = ke_ments
                else:
                    current_count += 1
        else:
            lang_merged_results[ke_id] = ke_ments
    return lang_merged_results


def fine_grained_relation_extraction(
        result_fdict,
        rel_ontology,
        depend_utils,
        dependency_patterns,
        ltf_utils,
        fine_ent_type_fnames=None,
        fine_ent_type_linking_fnames=None,
        fine_ent_type_taxonomy=None):
    results = {}
    discarded_results = {}
    result_parser = CSParser()

    timing_results = {}
    confidence_dist = {}
    type_dist = {}

    for lang_id, lang_result_flist in result_fdict.items():
        extract_start_t = time.time()

        lang_ltf_util = ltf_utils.get(lang_id, None)
        lang_dep_util = depend_utils.get(lang_id, None)
        lang_fine_ent_t_fname = fine_ent_type_fnames.get(lang_id, None)
        lang_fine_ent_t_linking_fname = fine_ent_type_linking_fnames.get(lang_id, None)
        lang_results = {KE_ENT: OrderedDict(), KE_REL: OrderedDict(), KE_EVT: OrderedDict()}
        lang_confidence_dist = {KE_ENT: Counter(), KE_REL: Counter(), KE_EVT: Counter()}

        print("{} := Collecting results from {}".format(lang_id, lang_result_flist))

        lang_doc_entities = {}
        lang_doc_events = {}
        for ere, pred_y in result_parser.parse(
                lang_id,
                lang_result_flist,
                lang_fine_ent_t_fname,
                lang_fine_ent_t_linking_fname,
                fine_ent_type_taxonomy):
            if ere in [KE_ENT, KE_REL, KE_EVT]:
                mention_data = lang_results[ere].get(pred_y.id, [])
                mention_data.append(pred_y)
                lang_results[ere][pred_y.id] = mention_data
                lang_confidence_dist[ere][pred_y.confidence] += 1

                if ere == KE_ENT:
                    lang_curr_doc_entities = lang_doc_entities.get(pred_y.provenance, [])
                    lang_curr_doc_entities.append(pred_y)
                    lang_doc_entities[pred_y.provenance] = lang_curr_doc_entities
                elif ere == KE_EVT:
                    lang_curr_doc_events = lang_doc_events.get(pred_y.provenance, [])
                    lang_curr_doc_events.append(pred_y)
                    lang_doc_events[pred_y.provenance] = lang_curr_doc_events

        print("{} := Running rules for fine-grained relation types.".format(lang_id))
        post_proc_start_t = time.time()
        (
            lang_fine_relations, lang_rel_dist, lang_discarded_relations
        ) = run_postprocess_rules(
            lang_results[KE_REL],
            lang_results[KE_ENT],
            rel_ontology,
            lang_dep_util,
            dependency_patterns,
            lang_ltf_util,
            lang_id
        )

        print("\tPost-process rule time: {} seconds".format(time.time() - post_proc_start_t))

        standalone_start_t = time.time()
        (
            s_lang_results, s_lang_rel_dist
        ) = run_standalone_rules(
            lang_doc_entities,
            lang_doc_events,
            rel_ontology,
            lang_ltf_util,
            lang_dep_util,
            dependency_patterns,
            lang_id,
            {KE_ENT: len(lang_results[KE_ENT]), KE_REL: len(lang_results[KE_REL])}
        )
        print("\tStandalone rule time: {} seconds".format(time.time() - standalone_start_t))

        # temp_fine_rels = {}
        # post_proc_rel_count = 0
        # for rel_ment_id, rel_ments in lang_fine_relations.items():
        #     temp_fine_rels[rel_ment_id] = rel_ments
        #     post_proc_rel_count += len(rel_ments)
        #
        # standalone_rel_count = 0
        # num_collisions = 0
        # for rel_ment_id, rel_ments in s_lang_results[KE_REL].items():
        #     standalone_rel_count += len(rel_ments)
        #     if rel_ment_id in temp_fine_rels:
        #         num_collisions += 1
        #         print("COLLISION: {}".format(rel_ment_id))
        #         print(rel_ments)
        #         print("=" * 50)
        #         print(lang_fine_relations[rel_ment_id])
        #         print("\n")

        id_overlap = set(lang_fine_relations.keys()) & set(s_lang_results[KE_REL].keys())
        assert len(id_overlap) == 0, \
            "Postprocess relations and standalone relations have collisions in their IDs: {}".format(id_overlap)

        lang_fine_relations.update(s_lang_results[KE_REL])  # TODO: Change this to use safe_merge and test.

        combined_lang_rel_dist = {}
        for reltype in rel_ontology.get_types():
            combined_lang_rel_dist[reltype] = lang_rel_dist[reltype] + s_lang_rel_dist[reltype]
        combined_lang_rel_dist = Counter(combined_lang_rel_dist)

        # print(post_proc_rel_count)
        # print(standalone_rel_count)
        # print(num_collisions)
        # print(sum(combined_lang_rel_dist.values()))
        # print(len(lang_fine_relations))
        assert sum(combined_lang_rel_dist.values()) == len(lang_fine_relations)

        results[lang_id] = dict(
            zip([KE_ENT, KE_REL, KE_EVT], [lang_results[KE_ENT], lang_fine_relations, lang_results[KE_EVT]])
        )

        discarded_results[lang_id] = dict(
            zip([KE_ENT, KE_REL, KE_EVT], [lang_results[KE_ENT], lang_discarded_relations, lang_results[KE_EVT]])
        )

        lang_duration = time.time() - extract_start_t
        timing_results[lang_id] = lang_duration

        confidence_dist[lang_id] = lang_confidence_dist
        type_dist[lang_id] = {KE_REL: combined_lang_rel_dist}

        result_parser.reset_counters()

        print("\tFinished processing {}.\n".format(lang_id))
        print("\t\tNumber of {} relations: {}".format(lang_id, sum(combined_lang_rel_dist.values())))
        print("\t\tTime: {} seconds\n".format(lang_duration))
        # print("\tFinished processing {}. (Time: {} seconds)\n".format(lang_id, lang_duration))

    return results, timing_results, confidence_dist, type_dist, discarded_results


def combine_hypo_relations(
        result_fdict,
        rel_ontology,
        fine_ent_type_fnames=None,
        fine_ent_type_linking_fnames=None,
        fine_ent_type_taxonomy=None
):
    results = {}
    result_parser = CSParser()
    timing_results = {}

    confidence_dist = {}
    type_dist = {}

    for lang_id, lang_result_flist in result_fdict.items():
        extract_start_t = time.time()

        lang_fine_ent_t_fname = fine_ent_type_fnames.get(lang_id, None)
        lang_fine_ent_t_linking_fname = fine_ent_type_linking_fnames.get(lang_id, None)
        lang_results = {KE_ENT: OrderedDict(), KE_REL: OrderedDict(), KE_EVT: OrderedDict()}
        lang_confidence_dist = {KE_ENT: Counter(), KE_REL: Counter(), KE_EVT: Counter()}
        lang_type_dist = {
            KE_ENT: Counter(),
            KE_REL: Counter(dict(zip(rel_ontology.get_types(), [0] * len(rel_ontology.get_types())))),
            KE_EVT: Counter()
        }

        print("{} := Collecting results from {}".format(lang_id, lang_result_flist))

        for ere, pred_y in result_parser.parse(
                lang_id,
                lang_result_flist,
                lang_fine_ent_t_fname,
                lang_fine_ent_t_linking_fname,
                fine_ent_type_taxonomy):
            if ere in [KE_ENT, KE_REL, KE_EVT]:
                if ere == KE_REL:
                    pred_y = rel_ontology.adjust(pred_y)
                mention_data = lang_results[ere].get(pred_y.id, [])
                mention_data.append(pred_y)
                lang_results[ere][pred_y.id] = mention_data
                lang_confidence_dist[ere][pred_y.confidence] += 1
                lang_type_dist[ere][pred_y.get_full_type()] += 1

        results[lang_id] = dict(
            zip([KE_ENT, KE_REL, KE_EVT], [lang_results[KE_ENT], lang_results[KE_REL], lang_results[KE_EVT]])
        )
        confidence_dist[lang_id] = lang_confidence_dist
        type_dist[lang_id] = lang_type_dist

        lang_duration = time.time() - extract_start_t
        timing_results[lang_id] = lang_duration

        result_parser.reset_counters()

        print("\tFinished processing {}.\n".format(lang_id))
        print("\t\tNumber of {} relations: {}".format(lang_id, sum(lang_type_dist[KE_REL].values())))
        print("\t\tTime: {} seconds".format(lang_duration))

    return results, timing_results, confidence_dist, type_dist


def output_results(
        results,
        outdirs,
        ltf_retrievers,
        outfname_form="{}.fine_rel.cs",
        # entity_outfname_form="{}.filler_entities.cs",
        output_old_format=False,
        old_outfname_form="{}.old_format.fine_rel.cs",
        output_tags=None
):
    output_files = OrderedDict()
    output_times = {}

    if output_tags is None:
        output_tags = {}

    for lang, lang_results in results.items():
        output_start_t = time.time()

        lang_rel_results = lang_results[KE_REL]
        lang_ent_results = lang_results[KE_ENT]
        lang_event_results = lang_results.get(KE_EVT, {})
        lang_ltf_retriever = ltf_retrievers[lang]
        lang_outdir = outdirs[lang]

        lang_outfname_form = add_tag_outfname_form(outfname_form, output_tags.get(lang, ""))
        lang_outfname = os.path.join(lang_outdir, lang_outfname_form.format(lang))

        print("{} := Saving results to {}".format(lang, lang_outfname))

        oldform_outf = None
        if output_old_format:
            lang_old_outfname_form = add_tag_outfname_form(old_outfname_form, output_tags.get(lang, ""))
            lang_old_outfname = os.path.join(lang_outdir, lang_old_outfname_form.format(lang))
            # lang_old_outfname = os.path.join(lang_outdir, old_outfname_form.format(lang))
            print("\tSaving {} old format results to {}".format(lang, lang_old_outfname))
            oldform_outf = open(lang_old_outfname, "w", encoding="utf-8")

        with open(lang_outfname, 'w', encoding="utf-8") as outf:
            for _, rel_mention in lang_rel_results.items():
                rel_mention = rel_mention[0]
                outf.write(
                    rel_mention.str_cs(lang_ent_results, lang_ltf_retriever, events=lang_event_results) + "\n"
                )

                if output_old_format and oldform_outf:
                    oldform_outf.write(str(rel_mention) + "\n")

        if output_old_format and oldform_outf:
            oldform_outf.close()

        output_files[lang] = lang_outfname

        lang_duration = time.time() - output_start_t
        output_times[lang] = lang_duration

        print("\tSaved {} results to {}. (Time: {} seconds)".format(lang, lang_outfname, lang_duration))
    print("\n".join([v for k, v in output_files.items()]) + "\n")
    return output_files, output_times


def output_dist(my_dist, outdirs, data_name, outfname_form="{}.{}_dist.{}.txt", output_tags=None):
    if output_tags is None:
        output_tags = {}

    for lang, lang_ere_dists in my_dist.items():
        lang_outfname_form = add_tag_outfname_form(outfname_form, output_tags.get(lang, ""))
        for ere, dist in lang_ere_dists.items():
            outfname = os.path.join(outdirs[lang], lang_outfname_form.format(lang, data_name, ere.lower()))
            with open(outfname, "w", encoding="utf-8") as outf:
                outf.write("\n".join(["{},{}".format(item, freq) for item, freq in dist.most_common()]))


def add_tag_outfname_form(outfname_form, output_tag):
    if output_tag:
        fname_data = os.path.splitext(outfname_form)
        return ".".join([fname_data[0], output_tag, fname_data[1].lstrip(".")])
    else:
        return outfname_form


def form_output_dirs(lang_ids, output_dir):
    outpaths = {}

    if not output_dir:
        return outpaths

    for lid in lang_ids:
        # lang_outdir = os.path.join(output_dir, lid, "{}".format(time.strftime('%Y%m%d', time.localtime())))
        lang_outdir = os.path.join(output_dir, "{}".format(lid))
        if not os.path.isdir(lang_outdir):
            os.makedirs(lang_outdir)
        outpaths[lid] = lang_outdir
    return outpaths


def load_arg_config(arg_config_resource, is_dict=False):
    if not is_dict:
        with open(arg_config_resource, "r", encoding="utf-8") as argf:
            arg_data = json.load(argf)
    else:
        arg_data = arg_config_resource

    finetype_util = FineGrainedEntityUtil(arg_data["hierarchy_dir"])
    rel_ontology = RelationOntology(arg_fname=arg_data["rel_ont_fname"])
    outdir = arg_data["outdir"]

    ltf_utils = {lang: LTFUtil(lang_ltf_dir) for lang, lang_ltf_dir in arg_data["ltf_utils"].items()}

    cs_fnames = arg_data["cs_fnames"]
    fine_ent_type_fnames = arg_data["fine_ent_type_fnames"]
    fine_ent_type_linking_fnames = arg_data["fine_ent_type_linking_fnames"]

    outtag = arg_data.get("outtag", None)

    # rsd_vis_dirs = arg_data.get("rsd_vis_dirs", {})
    rsd_vis_dirs = form_output_dirs(list(ltf_utils.keys()), arg_data.get("rsd_vis_dir", None))

    output_dirs = form_output_dirs(list(ltf_utils.keys()), outdir)

    depend_utils = {}
    for lang, dep_cfg in arg_data.get("dep_cfg", {}).items():
        depend_utils[lang] = DependencyParseUtil(
            lang_id=lang,
            cache_dir=output_dirs[lang],
            pos_rdr_fname=dep_cfg["pos_rdr"],
            pos_dict_fname=dep_cfg["pos_dict"],
            dp_model_path=dep_cfg["model_path"],
            dp_model_name=dep_cfg["model_name"],
            decode_alg=dep_cfg["decode_alg"],
            use_gpu=dep_cfg["use_gpu"],
            reuse_cache=dep_cfg["reuse_cache"]
        )

    with open(arg_data["depend_pattern_fname"], "r", encoding="utf-8") as d_pat_f:
        dependency_patterns = json.load(d_pat_f)

    return (
        finetype_util,
        rel_ontology,
        ltf_utils,
        depend_utils,
        dependency_patterns,
        cs_fnames,
        fine_ent_type_fnames,
        fine_ent_type_linking_fnames,
        output_dirs,
        rsd_vis_dirs,
        outtag
    )


def get_arg_config_defaults(args):
    arg_dict = {
        "hierarchy_dir": "./data/yago_taxonomy_wordnet_single_parent.json",
        "rel_ont_fname": "./data/relation_ontology_arg_labels_constraints.v8.txt",
        "depend_pattern_fname": "./data/dependency_patterns.json",
        "outdir": "",
        "rsd_vis_dir": "",

        "ltf_utils": {},

        "rsd_dirs": {},

        "cs_fnames": {},

        "fine_ent_type_fnames": {},

        "fine_ent_type_linking_fnames": {},

        "dep_cfg": {}
    }

    if "en" == args.lang_id:
        arg_dict["dep_cfg"][args.lang_id] = {
            "pos_rdr": "/relation/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllx.pos.RDR",
            "pos_dict": "/relation/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllx.pos.DICT",
            "model_path": "/relation/DependencyParse/models/en",
            "model_name": "biaffine.pt",
            "use_gpu": args.use_gpu,
            "decode_alg": "mst",
            "reuse_cache": args.reuse_cache
        }
    elif "ru" == args.lang_id:
        arg_dict["dep_cfg"][args.lang_id] = {
            "pos_rdr": "/relation/DependencyParse/data/ud-treebanks-v2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllx.pos.RDR",
            "pos_dict": "/relation/DependencyParse/data/ud-treebanks-v2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllx.pos.DICT",
            "model_path": "/relation/DependencyParse/models/ru",
            "model_name": "biaffine.pt",
            "use_gpu": args.use_gpu,
            "decode_alg": "mst",
            "reuse_cache": args.reuse_cache
        }
    elif "uk" == args.lang_id:
        arg_dict["dep_cfg"][args.lang_id] = {
            "pos_rdr": "/relation/DependencyParse/data/ud-treebanks-v2.3/UD_Ukrainian-IU/uk_iu-ud-train.conllx.pos.RDR",
            "pos_dict": "/relation/DependencyParse/data/ud-treebanks-v2.3/UD_Ukrainian-IU/uk_iu-ud-train.conllx.pos.DICT",
            "model_path": "/relation/DependencyParse/models/uk",
            "model_name": "biaffine.pt",
            "use_gpu": args.use_gpu,
            "decode_alg": "mst",
            "reuse_cache": args.reuse_cache
        }

    arg_dict["outdir"] = args.outdir
    arg_dict["ltf_utils"][args.lang_id] = args.ltf_dir
    arg_dict["rsd_dirs"][args.lang_id] = args.rsd_dir
    arg_dict["cs_fnames"][args.lang_id] = args.cs_fnames
    arg_dict["fine_ent_type_fnames"][args.lang_id] = args.fine_ent_type_json
    arg_dict["fine_ent_type_linking_fnames"][args.lang_id] = args.fine_ent_type_tab

    return arg_dict


if __name__ == "__main__":
    start_t = time.time()

    argparser = ArgumentParser()
    argparser.add_argument("--lang_id", help="Language ID", required=True)
    argparser.add_argument("--ltf_dir", help="LTF folder path.", required=True)
    argparser.add_argument("--rsd_dir", help="RSD folder path.", required=True)
    argparser.add_argument("--cs_fnames", nargs="+",
                           help="List of paths to input CS files.", required=True)
    argparser.add_argument("--fine_ent_type_tab",
                           help="Path to fine-grained entity type tab file.", required=True)
    argparser.add_argument("--fine_ent_type_json",
                           help="Path to fine-grained entity type json file.", required=True)
    argparser.add_argument("--outdir", help="Output folder path.", required=True)
    argparser.add_argument("--use_gpu",
                           action="store_true",
                           help="Use GPU for dependency parsing.")
    argparser.add_argument("--reuse_cache",
                           action="store_true",
                           help="Use cached dependency parsing output found in the same directory as the output directory.")
    argparser.add_argument("-v", "--visualize",
                           action="store_true", help="Additionally output BRAT visualization format.")
    argparser.add_argument("-c", "--confidence",
                           action="store_true", help="Output confidence distribution.")
    argparser.add_argument("-t", "--type",
                           action="store_true", help="Output type distribution.")
    argparser.add_argument("-d", "--discarded",
                           action="store_true", help="Output discarded coarse relations.")
    argparser.add_argument("-f", "--fine_grained",
                           action="store_true",
                           help="Run fine grained extraction.")
    argparser.add_argument("-m", "--merge_hypo",
                           action="store_true",
                           help="Merge hypothesis all relations instead of running fine grained extraction. (TA1b)")
    args = argparser.parse_args()

    print("----------")
    print("Parameters:")
    for arg in vars(args):
        print("\t{}: {}".format(arg, getattr(args, arg)))
    print("----------")

    assert args.fine_grained != args.merge_hypo, "Must provide either `-f` or `-m`, but not both."

    arg_config_res = get_arg_config_defaults(args)

    (
        finetype_util,
        rel_ontology,
        ltf_utils,
        depend_utils,
        depend_patterns,
        cs_fnames,
        fine_ent_type_fnames,
        fine_ent_type_linking_fnames,
        output_dirs,
        rsd_vis_dirs,
        outtag
    ) = load_arg_config(arg_config_res)

    hypo_outfname_form = "{}.hypo_fine_rel.cs"
    hypo_old_outfname_form = "{}.old_format.hypo_fine_rel.cs"

    fine_outfname_form = "{}.fine_rel.cs"
    fine_old_outfname_form = "{}.old_format.fine_rel.cs"

    disc_outfname_form = "{}.discarded_rel.cs"
    disc_old_outfname_form = "{}.old_format.discarded_rel.cs"

    conf_outfname_form = "{}.{}_dist.{}.txt"

    type_outfname_form = "{}.{}_dist.{}.txt"

    results = None

    if args.merge_hypo:
        conf_outfname_form = "{}.hypo_{}_dist.{}.txt"
        type_outfname_form = "{}.hypo_{}_dist.{}.txt"

        results, timing_results, confidence_dist, type_dist = combine_hypo_relations(
            cs_fnames,
            rel_ontology,
            fine_ent_type_fnames,
            fine_ent_type_linking_fnames,
            finetype_util
        )

        discarded_results = {}

        output_fnames, output_times = output_results(
            results,
            output_dirs,
            ltf_utils,
            outfname_form=hypo_outfname_form,
            output_old_format=True,
            old_outfname_form=hypo_old_outfname_form,
            output_tags=outtag
        )

    elif args.fine_grained:
        dep_start_t = time.time()
        depend_utils = collect_dependency_parses(depend_utils, ltf_utils)
        print("\tFull dependency parse time: {} seconds".format(time.time() - dep_start_t))

        (
            results, timing_results, confidence_dist, type_dist, discarded_results
        ) = fine_grained_relation_extraction(
            cs_fnames,
            rel_ontology,
            depend_utils,
            depend_patterns,
            ltf_utils,
            fine_ent_type_fnames,
            fine_ent_type_linking_fnames,
            finetype_util
        )

        output_fnames, output_times = output_results(
            results,
            output_dirs,
            ltf_utils,
            outfname_form=fine_outfname_form,
            output_old_format=True,
            old_outfname_form=fine_old_outfname_form,
            output_tags=outtag
        )
        # output_fnames, output_times = output_results(
        #     results,
        #     output_dirs,
        #     ltf_utils,
        #     outfname_form="{}.fine_rel.cs",
        #     output_old_format=True,
        #     old_outfname_form="{}.old_format.fine_rel.cs"
        # )

    if results:
        print("Total runtime: {}".format(time.time() - start_t))
        assert sorted(list(timing_results.keys())) == sorted(list(output_times.keys()))
        print("\n")
        for key, t in timing_results.items():
            total_t = t + output_times[key]
            print("{}: {} seconds".format(key, total_t))
        print("\n")

        if args.confidence:
            output_dist(
                confidence_dist,
                output_dirs,
                "confidence",
                outfname_form=conf_outfname_form,
                output_tags=outtag
            )
            # output_dist(confidence_dist, output_dirs, "confidence", outfname_form="{}.{}_dist.{}.txt")

        if args.type:
            output_dist(
                type_dist,
                output_dirs,
                "type",
                outfname_form=type_outfname_form,
                output_tags=outtag
            )
            # output_dist(type_dist, output_dirs, "type", outfname_form="{}.{}_dist.{}.txt")

        if args.visualize:
            for lid, lid_results in results.items():
                brat_visualize_relation_results(
                    lid_results[KE_ENT],
                    lid_results[KE_REL],
                    lid_results.get(KE_EVT, {}),
                    rsd_vis_dirs[lid],
                    rel_ontology
                )

        if args.discarded and not args.merge_hypo:
            output_results(
                discarded_results,
                output_dirs,
                ltf_utils,
                outfname_form=disc_outfname_form,
                output_old_format=True,
                old_outfname_form=disc_old_outfname_form,
                output_tags=outtag
            )
            # output_results(
            #     discarded_results,
            #     output_dirs,
            #     ltf_utils,
            #     outfname_form="{}.discarded_rel.cs",
            #     output_old_format=True,
            #     old_outfname_form="{}.old_format.discarded_rel.cs"
            # )


    # results = {
    #     "en_all": {KE_ENT: {}, KE_REL: {}, KE_EVT: {}},
    #     "ru_all": {KE_ENT: {}, KE_REL: {}, KE_EVT: {}},
    #     "uk": {KE_ENT: {}, KE_REL: {}, KE_EVT: {}}
    # }
    #
    # dists = {
    #     "en_all": {KE_ENT: {}, KE_REL: {}, KE_EVT: {}},
    #     "ru_all": {KE_ENT: {}, KE_REL: {}, KE_EVT: {}},
    #     "uk": {KE_ENT: {}, KE_REL: {}, KE_EVT: {}}
    # }
    #
    # print(outtag)
    # # outtag = {}
    #
    # # hypo_outfname_form = "{}.hypo_fine_rel.{}.cs" if outtag else "{}.hypo_fine_rel.cs"
    # # hypo_old_outfname_form = "{}.old_format.hypo_fine_rel.{}.cs" if outtag else "{}.old_format.hypo_fine_rel.cs"
    # #
    # # fine_outfname_form = "{}.fine_rel.{}.cs" if outtag else "{}.fine_rel.cs"
    # # fine_old_outfname_form = "{}.old_format.fine_rel.{}.cs" if outtag else "{}.old_format.fine_rel.cs"
    # #
    # # disc_outfname_form = "{}.discarded_rel.{}.cs" if outtag else "{}.discarded_rel.cs"
    # # disc_old_outfname_form = "{}.old_format.discarded_rel.{}.cs" if outtag else "{}.old_format.discarded_rel.cs"
    # #
    # # conf_outfname_form = "{}.{}_dist.{}.{}.txt" if outtag else "{}.{}_dist.{}.txt"
    # #
    # # type_outfname_form = "{}.{}_dist.{}.{}.txt" if outtag else "{}.{}_dist.{}.txt"
    #
    # # hypo_outfname_form = "{}.hypo_fine_rel.cs"
    # # hypo_old_outfname_form = "{}.old_format.hypo_fine_rel.cs"
    # #
    # # fine_outfname_form = "{}.fine_rel.cs"
    # # fine_old_outfname_form = "{}.old_format.fine_rel.cs"
    # #
    # # disc_outfname_form = "{}.discarded_rel.cs"
    # # disc_old_outfname_form = "{}.old_format.discarded_rel.cs"
    # #
    # # conf_outfname_form = "{}.{}_dist.{}.txt"
    # #
    # # type_outfname_form = "{}.{}_dist.{}.txt"
    #
    #
    # output_fnames, output_times = output_results(
    #     results,
    #     output_dirs,
    #     ltf_utils,
    #     outfname_form=hypo_outfname_form,
    #     output_old_format=True,
    #     old_outfname_form=hypo_old_outfname_form,
    #     output_tags=outtag
    # )
    #
    # output_fnames, output_times = output_results(
    #     results,
    #     output_dirs,
    #     ltf_utils,
    #     outfname_form=fine_outfname_form,
    #     output_old_format=True,
    #     old_outfname_form=fine_old_outfname_form,
    #     output_tags=outtag
    # )
    #
    # output_results(
    #     results,
    #     output_dirs,
    #     ltf_utils,
    #     outfname_form=disc_outfname_form,
    #     output_old_format=True,
    #     old_outfname_form=disc_old_outfname_form,
    #     output_tags=outtag
    # )
    #
    # output_dist(dists, output_dirs, "confidence", outfname_form=conf_outfname_form, output_tags=outtag)
    #
    # output_dist(dists, output_dirs, "type", outfname_form=type_outfname_form, output_tags=outtag)
