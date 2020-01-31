import ujson as json


class FineGrainedEntityUtil(object):
    def __init__(self, hierarchy_dir):
        super(FineGrainedEntityUtil, self).__init__()
        # self.hierarchy_dir = hierarchy_dir
        self.child_parents_dict = self.load_parent(hierarchy_dir)

    def load_parent(self, hierarchy_dir):
        child_parent_dict = {} # 1-1 mapping
        parent_child_dict = json.load(open(hierarchy_dir))
        for parent in parent_child_dict:
            for child in parent_child_dict[parent]:
                child_parent_dict[child] = parent

        child_parents_dict = {}
        for child in child_parent_dict:
            child_parents_dict[child] = []
            self._get_parents(child, child_parent_dict, child_parents_dict[child])

        return child_parents_dict

    def _get_parents(self, child, child_parent_dict, parents):
        if child not in child_parent_dict:
            return parents
        else:
            parent_type = child_parent_dict[child]
            parents.append(parent_type)
            self._get_parents(parent_type, child_parent_dict, parents)

    def get_all_types(self, type_list):
        alltypes = set()
        for type in type_list:
            alltypes.add(type)
            alltypes.update(self.child_parents_dict.get(type, []))
        return list(alltypes)

    def _prep_type_old(self, typestr):
        return typestr.replace("https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/SeedlingOntology#", "")

    # <entityid, fine_type_list>
    def entity_finegrain_by_json(self, entity_finegrain_json, entity_freebase_tab, entity_coarse,
                                 filler_coarse=None, add_coarse=True, add_parent=True):
        entity_dict = {}
        fine_mapping = json.load(open(entity_finegrain_json))
        offset_fb_mapping = self._load_offset_fine_mapping(entity_freebase_tab)
        self._entity_finegrain_by_json(offset_fb_mapping, fine_mapping, entity_dict, entity_coarse,
                                       add_coarse, add_parent)
        if filler_coarse is not None and len(filler_coarse) != 0:
            self._entity_finegrain_by_json(offset_fb_mapping, fine_mapping, entity_dict, filler_coarse,
                                           add_coarse, add_parent)
        return entity_dict

    def _entity_finegrain_by_json(self, offset_fb_mapping, fb_fine_mapping, entity_dict, coarse_file,
                                  add_coarse=True, add_parent=True):
        for line in open(coarse_file):
            if line.startswith(':Entity') or line.startswith(':Filler'):  # (????)
                line = line.rstrip('\n')
                tabs = line.split('\t')
                if tabs[0] not in entity_dict:
                    entity_dict[tabs[0]] = set()  # []
                if tabs[1] == 'type':
                    if add_coarse:
                        type_str = self._prep_type_old(tabs[2])
                        entity_dict[tabs[0]].add(type_str)  # append(type_str)
                elif 'mention' in tabs[1]:
                    offset = tabs[3]
                    if offset in offset_fb_mapping:
                        link = offset_fb_mapping[offset]
                        if link in fb_fine_mapping:
                            fine_type = fb_fine_mapping[link]
                            # print(fine_type)
                            if add_parent:
                                entity_dict[tabs[0]].update(
                                    self.get_all_types(fine_type))  # .extend(self.get_all_types(fine_type))
                            else:
                                entity_dict[tabs[0]].update(fine_type)  # .extend(fine_type)
                # if tabs[1] == 'link':
                #     link = tabs[2].replace("LDC2015E42:", "")
                #     if link in fb_fine_mapping:
                #         entity_dict[tabs[0]].extend(self.get_all_types(fb_fine_mapping[link]))
                #         # entity_dict[tabs[0]].extend(fine_mapping[link])

    def _load_offset_fine_mapping(self, entity_coarse_freebase):
        offset_fine_mapping = {}
        for line in open(entity_coarse_freebase):
            line = line.rstrip('\n')
            tabs = line.split('\t')
            offset = tabs[3]
            link = tabs[4]
            if not link.startswith('NIL'):
                offset_fine_mapping[offset] = link
        return offset_fine_mapping


if __name__ == '__main__':
    hierarchy_dir = '/nas/data/m1/panx2/workspace/kbp18/data/prepare/output/yago_taxonomy_wordnet_single_parent.json'
    finetype_util = FineGrainedEntityUtil(hierarchy_dir)

    type_list = ["Capital108518505", "BusinessDistrict108539072"]
    print(finetype_util.get_all_types(type_list))
