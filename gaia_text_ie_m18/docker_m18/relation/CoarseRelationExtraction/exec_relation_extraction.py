import io
import os
import time
import shutil
import argparse

import en_core_web_sm
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc

from utils import backbones

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        corrected_words = []
        i = 0
        while i < len(words) - 1:
            if words[i + 1] == "0/0" and words[i] == "0":
                corrected_words.append(" ".join(words[i:i + 2]))
                i += 1
            else:
                corrected_words.append(words[i])
            i += 1
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(corrected_words)
        return Doc(self.vocab, words=corrected_words, spaces=spaces)
        # spaces = [True] * len(words)
        # return Doc(self.vocab, words=words, spaces=spaces)


def aida_relation_en(input_arg_dict):
    # Load spacy wrapper
    spacy_model = en_core_web_sm.load()
    # custom_tokenizer = Tokenizer(spacy_model.vocab, {}, None, None, None)
    custom_tokenizer = WhitespaceTokenizer(spacy_model.vocab)
    spacy_model.tokenizer = custom_tokenizer

    # Load pre-trained model
    word_dic, type_dic, neural_model = backbones.load_model(
        train_file=os.path.join(CURRENT_DIR, "data/ere_filtered_train.txt"),
        eval_file=os.path.join(CURRENT_DIR, "data/ere_filtered_test.txt"),
        model_path=os.path.join(CURRENT_DIR, "data/filter_ere_dp_mask_5000.636.pkl")
    )

    # process edl and convert to intermediate files
    inter_flag = backbones.json2inter(input_arg_dict['edl_tab'], input_arg_dict['input'], type_dic,
                                      intermediate_path=input_arg_dict["intermediate_path"])

    # extract shortest dependency path
    dep_path_flag = backbones.extract_shortest_path(
        spacy_model,
        intermediate_path=input_arg_dict["intermediate_path"],
        depend_out_path=input_arg_dict["depend_out_path"]
    )

    # extract relations
    extract_rel_flag = backbones.extract_relations(
        neural_model,
        word_dic,
        type_dic,
        T=input_arg_dict["sm_temp"],
        test_file=input_arg_dict["intermediate_path"],
        result_path=input_arg_dict["pre_postprocess_result_path"],
        depend_out_path=input_arg_dict["depend_out_path"]
    )

    # post processing and extract sponsor relation
    post_flag = backbones.post_processing(
        plain_text=input_arg_dict["intermediate_path"],
        results=input_arg_dict["pre_postprocess_result_path"],
        skip_sponsor=True,
        postprocess_result_path=input_arg_dict["postprocess_result_path"]
    )

    # # extract sponsor relations
    # sponsor_flag = backbones.extract_sponsor(spacy_model)

    # generate final cs file
    out_cs_list = backbones.generate_relation_cs(
        input_arg_dict['edl_cs'],
        plain_text_file=input_arg_dict["intermediate_path"],
        final_results_file=input_arg_dict["postprocess_result_path"]
    )
    # shutil.rmtree("temp")

    return out_cs_list


def aida_relation_ru(input_arg_dict):
    word_dic_ru, type_dic_ru, neural_model_ru = backbones.load_ru(
        train_file=os.path.join(CURRENT_DIR, "data/convert_ere_ru_train.txt"),
        eval_file=os.path.join(CURRENT_DIR, "data/convert_ere_ru_test.txt"),
        model_path=os.path.join(CURRENT_DIR, "data/ru_new_piece_5000.724.pkl")
    )

    # process edl and convert to intermediate files
    inter_flag = backbones.json2inter(input_arg_dict['edl_tab'], input_arg_dict['input'], type_dic_ru,
                                      intermediate_path=input_arg_dict["intermediate_path"])

    # extract relations
    extract_rel_flag = backbones.extract_relations_cl(
        neural_model_ru,
        word_dic_ru,
        type_dic_ru,
        batch_size=63,
        T=input_arg_dict["sm_temp"],
        test_file=input_arg_dict["intermediate_path"],
        result_path=input_arg_dict["postprocess_result_path"]
    )

    # generate final cs file
    out_cs_list = backbones.generate_relation_cs(
        input_arg_dict['edl_cs'],
        plain_text_file=input_arg_dict["intermediate_path"],
        final_results_file=input_arg_dict["postprocess_result_path"]
    )

    # shutil.rmtree("temp")

    return out_cs_list


def aida_relation_uk(input_arg_dict):
    word_dic_uk, type_dic_uk, neural_model_uk = backbones.load_uk(
        train_file=os.path.join(CURRENT_DIR, "data/convert_ere_uk_train.txt"),
        eval_file=os.path.join(CURRENT_DIR, "data/convert_ere_uk_test.txt"),
        model_path=os.path.join(CURRENT_DIR, "data/uk_new_piece_5000.682.pkl")
    )

    # process edl and convert to intermediate files
    inter_flag = backbones.json2inter(input_arg_dict['edl_tab'], input_arg_dict['input'], type_dic_uk,
                                      intermediate_path=input_arg_dict["intermediate_path"])

    # extract relations
    extract_rel_flag = backbones.extract_relations_cl(
        neural_model_uk,
        word_dic_uk,
        type_dic_uk,
        T=input_arg_dict["sm_temp"],
        test_file=input_arg_dict["intermediate_path"],
        result_path=input_arg_dict["postprocess_result_path"]
    )

    # generate final cs file
    out_cs_list = backbones.generate_relation_cs(
        input_arg_dict['edl_cs'],
        plain_text_file=input_arg_dict["intermediate_path"],
        final_results_file=input_arg_dict["postprocess_result_path"]
    )

    # shutil.rmtree("temp")

    return out_cs_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Call the aida relation API acquire output')
    parser.add_argument('-l', '--list', help='LTF file list path', required=True)
    parser.add_argument('-f', '--ltf_folder', help='LTF folder path', required=True)
    parser.add_argument('-e', '--edl_cs', help='EDL CS file path', required=True)
    parser.add_argument('-t', '--edl_tab', help='EDL tab file path', required=True)
    # parser.add_argument('-d', '--output_dir', help='Output dir path', required=True)
    parser.add_argument('-o', '--output_path', help='Output CS file path', required=True)
    parser.add_argument('-i', '--lang_id', help='Language ID', required=True)
    parser.add_argument('-s', '--sm_temp', help='Temperature of softmax.', default=1.0, type=float)

    args = vars(parser.parse_args())

    print("----------")
    print("Parameters:")
    for ar in args:
        print("\t{}: {}".format(ar, args[ar]))
    print("----------")

    input_file_list_file_path = args['list']
    input_ltf_folder_path = args['ltf_folder']
    input_edl_cs_file_path = args['edl_cs']
    input_edl_tab_file_path = args['edl_tab']
    output_relation_output_dir = os.path.dirname(args['output_path']) #args['output_dir']
    output_relation_output_file_path = args['output_path'] #os.path.join(output_relation_output_dir, args['output_path'])
    T = args["sm_temp"]

    temp_dict = dict()
    temp_dict['edl_cs'] = io.open(input_edl_cs_file_path, encoding='utf-8').read().strip("\n")
    temp_dict['edl_tab'] = io.open(input_edl_tab_file_path, encoding='utf-8').read().strip("\n")
    temp_dict['input'] = dict()
    temp_dict["sm_temp"] = T
    for one_line in io.open(input_file_list_file_path):
        one_line = one_line.strip()
        base_name = one_line.replace(".ltf.xml", "")
        one_ltf_xml_file_path = os.path.join(input_ltf_folder_path, one_line)
        temp_dict['input'][base_name] = io.open(one_ltf_xml_file_path, encoding="utf-8").read()

    temp_dict["temp_dir"] = "./{}_temp".format(args["lang_id"])

    if not os.path.exists(output_relation_output_dir):
        os.mkdir(output_relation_output_dir)
    if not os.path.exists(temp_dict["temp_dir"]):
        os.mkdir(temp_dict["temp_dir"])

    temp_dict["intermediate_path"] = os.path.join(temp_dict["temp_dir"], "AIDA_plain_text.txt")
    temp_dict["depend_out_path"] = os.path.join(temp_dict["temp_dir"], "dp.pkl")
    temp_dict["pre_postprocess_result_path"] = os.path.join(temp_dict["temp_dir"], "AIDA_results.txt")
    temp_dict["postprocess_result_path"] = os.path.join(temp_dict["temp_dir"], "results_post_sponsor.txt")

    start_t = time.time()

    assert temp_dict["sm_temp"] == 1.0

    relation_results = None
    if args["lang_id"] == "en":
        relation_results = aida_relation_en(temp_dict)
    elif args["lang_id"] == "ru":
        relation_results = aida_relation_ru(temp_dict)
    elif args["lang_id"] == "uk":
        relation_results = aida_relation_uk(temp_dict)
    else:
        relation_results = aida_relation_en(temp_dict)

    print("{} := Relation extraction runtime: {}".format(args["lang_id"], time.time() - start_t))
    print("\t{} := Number of relations: {}".format(args["lang_id"], len(relation_results)))

    with open(output_relation_output_file_path, "w", encoding="utf-8") as outf:
        outf.write('\n'.join(relation_results))
        # outf.write('\n'.join(sorted(relation_results)))

    shutil.rmtree(temp_dict["temp_dir"])
