import os
import codecs
import subprocess
import argparse


CONV_SCRIPT_PYTHON = "/opt/conda/envs/py36/bin/python"
LTF2RSD_SCRIPT = "./ltf2rsd.py"

UDP_PYTHON = "/opt/conda/envs/py27/bin/python"
POS_TAGGER_DIR = "./pSCRDRtagger"
POS_TAGGER_SCRIPT = "RDRPOSTagger.py"


def run_ltf2rsd(ltf_dir, out_dir):
    subprocess.check_output([CONV_SCRIPT_PYTHON,
                             LTF2RSD_SCRIPT,
                             "--dir",
                             "--dat",
                             ltf_dir,
                             out_dir])


def convert_ud2pos(ud_fname, pos_fname, ud_separator=None, tag_separator="/", skip_comment=True):
    with codecs.open(pos_fname, 'w', encoding="utf-8") as posf:
        with codecs.open(ud_fname, 'r', encoding="utf-8") as udf:
            current_doc = []
            for line in udf:
                line = line.rstrip()
                if skip_comment and line.startswith('#'):
                    continue
                if line:
                    segs = line.strip().split(ud_separator)
                    token, pos_tag = segs[1], segs[4]
                    current_doc.append((token, pos_tag))
                elif current_doc:
                    token_pos_pairs = [tag_separator.join([tok, pos_t]) for tok, pos_t in current_doc]
                    outstr = " ".join(token_pos_pairs) + "\n"
                    posf.write(outstr)
                    # posf.write(outstr.encode("utf-8"))
                    current_doc = []
            if current_doc:
                token_pos_pairs = [tag_separator.join([tok, pos_t]) for tok, pos_t in current_doc]
                outstr = " ".join(token_pos_pairs)
                posf.write(outstr)
                # posf.write(outstr.encode("utf-8"))


def convert_pos2ud(pos_fname, ud_fname, ud_separator=None, tag_separator="/", blanc="_"):
    with codecs.open(ud_fname, 'w', encoding="utf-8") as udf:
        with codecs.open(pos_fname, 'r', encoding="utf-8") as posf:
            for line in posf:
                line = line.rstrip()
                if line:
                    # print(pos_fname)
                    temp = [tuple(token_pos_pair.rsplit(tag_separator, 1)) for token_pos_pair in line.split()]
                    # print(temp)
                    for i, (token, pos_tag) in enumerate(temp, 1):
                        cpos_tag = pos_tag.split("_")[0]
                        x = "\t".join([str(i), token, token, cpos_tag, pos_tag, blanc, str(i - 1), "obj", blanc, blanc])
                        x = x + "\n"
                        udf.write(x)
                        # udf.write(x.encode("utf-8"))
                    udf.write("\n")


def run_postagger(data_fname, training=True, model_fname=None, lexicon_fname=None):
    os.chdir(POS_TAGGER_DIR)
    if training:
        subprocess.check_output([UDP_PYTHON,
                                 POS_TAGGER_SCRIPT,
                                 "train",
                                 data_fname])
        curr_outfname = (data_fname + ".RDR", data_fname + ".DICT")
    elif model_fname and lexicon_fname:
        subprocess.check_output([UDP_PYTHON,
                                 POS_TAGGER_SCRIPT,
                                 "tag",
                                 model_fname,
                                 lexicon_fname,
                                 data_fname])
        curr_outfname = data_fname + ".TAGGED"
    else:
        raise ValueError("model_fname and lexicon_fname must be provided to run tagger.")
    os.chdir("../")
    return curr_outfname


def train_postagger(data_fname, convert=True):
    if convert:
        pos_fname = data_fname + ".pos"
        convert_ud2pos(data_fname, pos_fname)
    else:
        pos_fname = data_fname
    trained_model = run_postagger(pos_fname)
    print("Trained model location: {}".format(trained_model))


def aida_postagging(ltf_dir, rsd_dir, model_fname, lexicon_fname):
    run_ltf2rsd(ltf_dir, rsd_dir)
    rsd_flist = [os.path.join(rsd_dir, rsd_fname) for rsd_fname in os.listdir(rsd_dir)
                 if rsd_fname.endswith(".rsd_tok.txt")]
    for rsd_fname in rsd_flist:
        out_tagged_fname = run_postagger(rsd_fname, False, model_fname, lexicon_fname)
        ud_tagged_fname = rsd_fname + ".UD"
        convert_pos2ud(out_tagged_fname, ud_tagged_fname)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='AIDA Multi-lingual POS Tagging')
    args_parser.add_argument('--data_path', help='Path (directory or file) to input data. Must be conllx format!',
                             required=True)
    args_parser.add_argument('--rsd_data_path',
                             help='Path to output tokenized rsd files resulting from ltf to rsd conversion.')
    args_parser.add_argument('--model_fname', help='Trained POS tagger file.')
    args_parser.add_argument('--lexicon_fname', help='POS tagger lexicon file.')
    args_parser.add_argument('--train', action='store_true', help='Train POS tagger.')
    args = args_parser.parse_args()

    if args.train:
        train_postagger(args.data_path)
    else:
        aida_postagging(args.data_path, args.rsd_data_path, args.model_fname, args.lexicon_fname)
