import os
import ujson as json
import linecache
import subprocess
import itertools
import networkx as nx
from collections import namedtuple
from networkx.readwrite import json_graph


UDP_PYTHON = "/home/whites5/anaconda2/envs/udp/bin/python2.7"
POS_TAGGER_SCRIPT = "/home/whites5/AIDA/DependencyParse/RDRPOSTagger/pSCRDRtagger/aida_pos_tagger.py"

DEPEND_PARSER_SCRIPT = "/home/whites5/AIDA/DependencyParse/NeuroNLP2/examples/aida_depend.py"
DEPEND_PUNCT_SET = ['.', '``', "''", ':', ',']

DependNode = namedtuple("DependNode", ["token", "pos"])
DependEdge = namedtuple("DependEdge", ["source", "etype", "target"])


class DependencyPattern(object):
    def __init__(self, dep_pattern_list):
        self._pattern = self._convert_to_pattern(dep_pattern_list)

    def __repr__(self):
        return "DependencyPattern({})".format(self._pattern)

    def __str__(self):
        return self.__repr__()

    def read_token_file(self, token_fname):
        token_data = []
        with open(token_fname, "r", encoding="utf-8") as tok_f:
            for line in tok_f:
                line = line.strip()
                if line:
                    token_data.append(line)
        return token_data

    def load_token_data(self, token_data_list):
        all_token_data = []
        for tok in token_data_list:
            if tok.endswith(".txt"):
                all_token_data.extend(self.read_token_file(tok))
            else:
                all_token_data.append(tok)
        return all_token_data

    def _convert_to_pattern(self, dep_pattern):
        pattern = []
        for u_data, etype_vals, v_data in dep_pattern:
            u_node = DependNode(token=self.load_token_data(u_data.get("token", ["*"])), pos=u_data.get("pos", ["*"]))

            v_node = DependNode(token=self.load_token_data(v_data.get("token", ["*"])), pos=v_data.get("pos", ["*"]))

            pattern.append(
                DependEdge(source=u_node, etype=etype_vals, target=v_node)
            )
        return pattern

    def _element_match(self, element_pattern_list, elem):
        if "*" in element_pattern_list:
            return True

        for elem_pat in element_pattern_list:
            if elem_pat in elem:
                return True

        return False

    def _triple_match(self, dependency_edge, pattern_edge, rev_match=False):
        if not rev_match:
            src_dep_comp_tok = dependency_edge.source.token
            src_dep_comp_pos = dependency_edge.source.pos
            tgt_dep_comp_tok = dependency_edge.target.token
            tgt_dep_comp_pos = dependency_edge.target.pos
        else:
            src_dep_comp_tok = dependency_edge.target.token
            src_dep_comp_pos = dependency_edge.target.pos
            tgt_dep_comp_tok = dependency_edge.source.token
            tgt_dep_comp_pos = dependency_edge.source.pos

        source_match = (
                self._element_match(pattern_edge.source.token, src_dep_comp_tok)
                and self._element_match(pattern_edge.source.pos, src_dep_comp_pos)
        )

        etype_match = self._element_match(pattern_edge.etype, dependency_edge.etype)
        target_match = (
                self._element_match(pattern_edge.target.token, tgt_dep_comp_tok)
                and self._element_match(pattern_edge.target.pos, tgt_dep_comp_pos)
        )
        # source_match = (
        #     ("*" in pattern_edge.source.token or src_dep_comp_tok in pattern_edge.source.token)
        #     and ("*" in pattern_edge.source.pos or src_dep_comp_pos in pattern_edge.source.pos)
        #
        # )
        # etype_match = ("*" in pattern_edge.etype or dependency_edge.etype in pattern_edge.etype)
        # target_match = (
        #     ("*" in pattern_edge.target.token or tgt_dep_comp_tok in pattern_edge.target.token)
        #     and ("*" in pattern_edge.target.pos or tgt_dep_comp_pos in pattern_edge.target.pos)
        # )
        return source_match and etype_match and target_match

    def accept_path(self, path_list, bad_edge_type_list=None, good_edge_type_list=None):
        if not path_list:
            return False

        found_good = False
        if not bad_edge_type_list:
            bad_edge_type_list = []

        if not good_edge_type_list:
            good_edge_type_list = []
            found_good = True

        for path in path_list:
            if path:
                for edge in path:
                    if edge.etype in bad_edge_type_list:
                        return False
                    elif edge.etype in good_edge_type_list:
                        found_good = True
        if found_good:
            return True
        else:
            return False

    def match(self, dependency_path_list):
        for dependency_path in dependency_path_list:
            if dependency_path:
                if all([self._triple_match(d_edge, p_edge) for p_edge, d_edge in zip(self._pattern, dependency_path)]):
                    return True
                elif all([self._triple_match(d_edge, p_edge, rev_match=True)
                          for p_edge, d_edge in zip(self._pattern, dependency_path[::-1])]):
                    return True
        return False


class DependencyInstance(object):
    def __init__(self, dep_graph_data, segment_data=None, undirected=True):
        self.dep_graph = json_graph.node_link_graph(dep_graph_data)
        if undirected:
            self.dep_graph = self.dep_graph.to_undirected()

        if segment_data is None:
            self.sentence = list(list(zip(*sorted(self.dep_graph.nodes(data="token"), key=lambda x: x[0])))[1])
        else:
            self.sentence = segment_data.tokens
            self.adjust_graph(segment_data)

        self.provenance, offset_str = self.dep_graph.graph["provenance"].split(":")
        self.start_offset, self.end_offset = int(offset_str.split("-")[0]), int(offset_str.split("-")[1])

    def __repr__(self):
        return "DependencyInstance({})".format(json_graph.node_link_data(self.dep_graph))

    def __str__(self):
        return self.__repr__()

    @property
    def full_provenance(self):
        return self.dep_graph.graph["provenance"]

    @property
    def edges(self, **kwargs):
        return self.dep_graph.edges(**kwargs)

    @property
    def root(self):
        return self.dep_graph.graph["root"], self.dep_graph.nodes[self.dep_graph.graph["root"]]

    def adjust_graph(self, segment_data):
        for i, token_w_offsets in enumerate(zip(segment_data.tokens, segment_data.token_offsets)):
            token, (token_start_offset, token_end_offset) = token_w_offsets
            self.dep_graph.nodes[i]["token"] = token
            self.dep_graph.nodes[i]["start_offset"] = token_start_offset
            self.dep_graph.nodes[i]["end_offset"] = token_end_offset

    def offsets2nodes(self, start_offset, end_offset):
        span_nodes = []
        for u, u_data in self.dep_graph.nodes(data=True):
            if u_data["start_offset"] >= start_offset and u_data["end_offset"] <= end_offset:
                span_nodes.append(u)

        return sorted(span_nodes)

    def get_path(self, token1_idx, token2_idx):
        path_nodes = None
        if nx.has_path(self.dep_graph, source=token1_idx, target=token2_idx):
            path_nodes = nx.shortest_path(self.dep_graph, source=token1_idx, target=token2_idx)
        elif nx.has_path(self.dep_graph, source=token2_idx, target=token1_idx):
            path_nodes = nx.shortest_path(self.dep_graph, source=token2_idx, target=token1_idx)
        else:
            return path_nodes

        path_data = []
        for i in range(1, len(path_nodes)):
            u, v = path_nodes[i - 1], path_nodes[i]
            u_data = self.dep_graph.nodes[u]
            u_node = DependNode(token=u_data["token"], pos=u_data["pos_tag"])

            v_data = self.dep_graph.nodes[v]
            v_node = DependNode(token=v_data["token"], pos=v_data["pos_tag"])

            path_data.append(
                DependEdge(source=u_node, etype=self.dep_graph.edges[u, v]["type"], target=v_node)
            )
        return path_data

    def get_path_set(self, u_nodes, v_nodes, provide_nodes=False):
        for u, v in itertools.product(u_nodes, v_nodes):
            if not provide_nodes:
                yield self.get_path(u, v)
            else:
                yield self.get_path(u, v), u, v

    def get_offset_paths(self, arg1_start_offset, arg1_end_offset, arg2_start_offset, arg2_end_offset):
        arg1_node_span = self.offsets2nodes(arg1_start_offset, arg1_end_offset)
        arg2_node_span = self.offsets2nodes(arg2_start_offset, arg2_end_offset)
        if len(arg1_node_span) and len(arg2_node_span):
            all_paths = [p for p in self.get_path_set(arg1_node_span, arg2_node_span)]
        else:
            all_paths = []
        return all_paths


class DependencyParseUtil(object):
    def __init__(self,
                 lang_id,
                 cache_dir,
                 pos_rdr_fname,
                 pos_dict_fname,
                 dp_model_path,
                 dp_model_name,
                 decode_alg,
                 use_gpu,
                 reuse_cache=True):
        self.pos_rdr_fname = pos_rdr_fname
        self.pos_dict_fname = pos_dict_fname
        self.cache_dir = cache_dir
        self.cache_fname = os.path.join(cache_dir, "{}.dep_parse.txt".format(lang_id))
        self.dep_log_fname = os.path.join(cache_dir, "{}.log.dep_parse.out".format(lang_id))
        self.dep_input_fname = os.path.join(cache_dir, "{}.doc_segments.txt".format(lang_id))
        self.reuse_cache = reuse_cache
        self.cached_parses = {} if not reuse_cache else self._index_depend_parse()

        self.pos_comm = [UDP_PYTHON, POS_TAGGER_SCRIPT, "tag", pos_rdr_fname, pos_dict_fname]

        self.dp_comm = [
            UDP_PYTHON, DEPEND_PARSER_SCRIPT,
            "--model_path", dp_model_path,
            "--model_name", dp_model_name,
            "--punctuation", *DEPEND_PUNCT_SET,
            "--decode", decode_alg,
            "test_phase",
            "--parser", "biaffine",
            "--input_data", self.dep_input_fname,
            "--output_path", self.cache_fname
        ]

        if use_gpu:
            self.dp_comm.insert(6, "--gpu")

    def run_pos(self, sentence_list, as_list=False):
        all_sentences = "\n".join([" ".join(s) for s in sentence_list])
        tagged_sents = subprocess.run(
            self.pos_comm,
            input=all_sentences,
            universal_newlines=True,
            encoding="utf-8",
            check=True,
            stdout=subprocess.PIPE
        ).stdout.strip().split("\n")
        if as_list:
            return [[tuple(tok_tag.split("||")) for tok_tag in tag_sent.split()] for tag_sent in tagged_sents]
        else:
            return tagged_sents

    def _index_depend_parse(self):
        indexed_parses = {}
        with open(self.cache_fname, "r", encoding="utf-8") as cf:
            for i, line in enumerate(cf):
                line = line.strip()
                if line:
                    seg_id = line.split("\t")[0]
                    indexed_parses[seg_id] = i
        return indexed_parses

    def run_parser(self, offset_strs, sentences):
        pos_tagged_sents = self.run_pos(sentences)

        with open(self.dep_input_fname, "w", encoding="utf-8") as dinf:
            dinf.write("\n".join(["\t".join((sid, pos_s)) for sid, pos_s in zip(offset_strs, pos_tagged_sents)]) + "\n")

        with open(self.dep_log_fname, "w", encoding="utf-8") as deplogf:
            temp = subprocess.run(
                self.dp_comm,
                encoding="utf-8",
                stdout=deplogf,
                env={"CUDA_VISIBLE_DEVICES": "3"}
            )

        self.cached_parses = self._index_depend_parse()

    def get_dependency_parse(self, doc_seg_id, segment_data):
        idx = self.cached_parses.get(doc_seg_id, None)
        dep_g = None
        if idx is not None:
            line = linecache.getline(self.cache_fname, idx + 1).strip()
            prov, dep_g_str = line.split("\t", 1)
            dep_g = DependencyInstance(json.loads(dep_g_str), segment_data)

        return dep_g


if __name__ == "__main__":
    lang_id = "en"
    aida_cache_dir = "/home/whites5/AIDA/RelationExtraction/utils/tmp"
    aida_pos_rdr = "/data/m1/whites5/AIDA/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllx.pos.RDR"
    aida_pos_dict = "/data/m1/whites5/AIDA/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllx.pos.DICT"
    dep_mod_path = "/data/m1/whites5/AIDA/DependencyParse/models/"
    dep_mod_name = "biaffine.pt"
    use_gpu = True
    decode_alg = "mst"

    # examples = ["Hi , my name is Spencer .".split(), "The boy wants to run to the store .".split()]
    # example_ids = ["HC00002ZN:0-100", "HC0001234:123-456"]

    examples = [["UKRAINE", "Battle", "for", "Kramatorsk", "|", "Kim", "Vinnell", "|", "Al", "Jazeera", "English"]]
    example_ids = ["IC0015YFI:1-64"]

    dp_util = DependencyParseUtil(
        lang_id,
        aida_cache_dir,
        aida_pos_rdr,
        aida_pos_dict,
        dep_mod_path,
        dep_mod_name,
        decode_alg,
        use_gpu,
        reuse_cache=False
    )

    dp_util.run_parser(example_ids, examples)
    # dep_parse = dp_util.get_dependency_parse("HC00002ZN", "0", "100")
    # dep_parse = dp_util.get_dependency_parse("HC0001234:123-456")
    dep_parse = dp_util.get_dependency_parse(example_ids[0])

    print(dep_parse.edges(data=True))
    print(dep_parse.root)

    dep_path = dep_parse.get_path(3, 5)
    print(dep_path)
