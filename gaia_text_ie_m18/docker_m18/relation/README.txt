Coarse grained relations:
    Runs Ge's system on the given input and is self-contained.
    Provided bash scripts show arguments used to run the script.

Fine grained relations:
    `EVALfine_grained_relations.py` is the main script that should be run. You run this with a configuration file, examples of which can be found in the `arg_configs` directory.
    This module requires dependency parsing (which requires POS tagging) for all languages. The dependency parsing and POS tagging scripts are called directly from the `dependency_parse_utils.py`.


Dependency parsing library:
    https://github.com/XuezheMax/NeuroNLP2
    My code requires a modified testing script and output format.


POS tagging library:
    http://rdrpostagger.sourceforge.net/
    Also has a modified testing script and output format.

need to change:
(1)DependencyParse/RDRPOSTagger/aida_pos.py,
PYTHON
(2)EVALfine_grained_relations.py
"pos_rdr": "/data/m1/whites5/AIDA/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllx.pos.RDR",
            "pos_dict": "/data/m1/whites5/AIDA/DependencyParse/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllx.pos.DICT",
            "model_path": "/data/m1/whites5/AIDA/DependencyParse/models/en",

 how to run preprocess ahead?