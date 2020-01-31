import os
import traceback

from flask import Flask, jsonify, request
from torch.utils.data import DataLoader
from load import load_lstm_cnn_elmo_model, load_lstm_cnn_model, \
    load_attn_fet_model
from src.data import BioDataset, FetDataset
from src.util import convert_result, eng_nam_post_process, convert_bio2tab, \
    merge_bio, ltf2bio, tab2bio, bio2cfet, restore_order
from src.nominal import extract_nominal


app = Flask(__name__)


gpu = False
model_dir = './entity_api/models/'
# model_dir = os.path.join(os.path.dirname(__file__), 'models')
elmo_option = os.path.join(model_dir, 'eng.original.5.5b.json')
elmo_weight = os.path.join(model_dir, 'eng.original.5.5b.hdf5')
nominal_type_dir = os.path.join(model_dir, 'nominal_text')
# Preload models
eng_nam_model, eng_nam_vocabs = load_lstm_cnn_elmo_model(
    os.path.join(model_dir, 'eng.nam.mdl'), elmo_option, elmo_weight)
eng_nom_5type_model, eng_nom_5type_vocabs = load_lstm_cnn_elmo_model(
    os.path.join(model_dir, 'eng.nom.5type.mdl'), elmo_option, elmo_weight)
eng_nom_wv_model, eng_nom_wv_vocabs = load_lstm_cnn_elmo_model(
    os.path.join(model_dir, 'eng.nom.wv.mdl'), elmo_option, elmo_weight)
eng_pro_model, eng_pro_vocabs = load_lstm_cnn_elmo_model(
    os.path.join(model_dir, 'eng.pro.mdl'), elmo_option, elmo_weight)
rus_nam_5type_model, rus_nam_5type_vocabs = load_lstm_cnn_model(
    os.path.join(model_dir, 'rus.nam.5type.mdl'))
rus_nam_wv_model, rus_nam_wv_vocabs = load_lstm_cnn_model(
    os.path.join(model_dir, 'rus.nam.wv.mdl'))
ukr_nam_5type_model, ukr_nam_5type_vocabs = load_lstm_cnn_model(
    os.path.join(model_dir, 'ukr.nam.5type.mdl'))
ukr_nam_wv_model, ukr_nam_wv_vocabs = load_lstm_cnn_model(
    os.path.join(model_dir, 'ukr.nam.wv.mdl'))
eng_fet_model, eng_fet_vocabs = load_attn_fet_model(
    os.path.join(model_dir, 'eng.fet.attnfet.mdl'), gpu=gpu)
rus_fet_model, rus_fet_vocabs = load_attn_fet_model(
    os.path.join(model_dir, 'rus.fet.attnfet.mdl'), gpu=gpu)
ukr_fet_model, ukr_fet_vocabs = load_attn_fet_model(
    os.path.join(model_dir, 'ukr.fet.attnfet.mdl'), gpu=gpu)


def run_tagger(model, vocabs, bio_str, to_bio=True, separator=' ', conf=True, elmo=True):
    test_set = BioDataset(1, 0, -1, test_mode=True)
    test_set.process(bio_str, vocabs, fallback=True)
    label_itos = {i: s for s, i in vocabs['label'].items()}
    results = []
    for batch in DataLoader(test_set, batch_size=20, shuffle=False,
                            collate_fn=test_set.batch_process):
        token_ids, char_ids, elmo_ids, seq_lens, tokens, spans, ori = batch
        preds, lstm_out, conf_score = model.predict(token_ids,
                                                    elmo_ids if elmo else char_ids,
                                                    seq_lens,
                                                    return_hidden=False,
                                                    return_conf_score=True)
        preds = [[label_itos[l] for l in ls] for ls in preds]
        if elmo:
            results.append((preds, tokens, seq_lens.tolist(), spans, conf_score))
        else:
            results.append(restore_order(
                [preds, tokens, seq_lens.tolist(), spans, conf_score], ori))
    return convert_result(results, to_bio, separator=separator, conf=conf).strip()


def run_classifier(model, vocabs, json_str):
    test_set = FetDataset(gpu=gpu)
    test_set.process(json_str, vocabs)
    label_itos = {i: s for s, i in vocabs['label'].items()}

    results = []
    for batch in DataLoader(test_set, batch_size=100, shuffle=False,
                                   collate_fn=test_set.batch_process):
        (
            token_ids, men_mask, ctx_mask, dist, gather, men_ids, mentions,
            seq_lens
        ) = batch
        preds, scores = model.predict_aida(token_ids, men_mask, ctx_mask,
                                           dist, gather, seq_lens,
                                           top_only=True)
        preds = preds.int().data.tolist()
        for mid, m, p, s in zip(men_ids, mentions, preds, scores):
            p = [(label_itos[i], v) for i, (l, v) in enumerate(zip(p, s)) if l][
                0]
            results.append('{}\t{}\t{}\t{}'.format(
                mid,
                m,
                p[0],
                p[1]))
    return '\n'.join(results).strip()


@app.route('/tagging', methods=['POST'])
def tagging():
    try:
        ltf_str = request.form['ltf']
        rsd_str = request.form['rsd']
        doc_id = request.form['doc_id']
        bio_str = ltf2bio(ltf_str, doc_id)
        lang_code = request.form['lang']
        if lang_code == 'en':
            # NAM
            eng_nam_bio_str = run_tagger(eng_nam_model, eng_nam_vocabs, bio_str)
            eng_nam_bio_str = eng_nam_post_process(eng_nam_bio_str)
            eng_nam_tab_str = convert_bio2tab(eng_nam_bio_str, conf_col=2)
            # NOM
            eng_nom_5type_bio_str = run_tagger(eng_nom_5type_model, eng_nom_5type_vocabs, bio_str)
            eng_nom_wv_bio_str = run_tagger(eng_nom_wv_model, eng_nom_wv_vocabs, bio_str)
            eng_nom_bio_str = merge_bio(eng_nom_5type_bio_str, eng_nom_wv_bio_str)
            eng_nom_tab_str = convert_bio2tab(eng_nom_bio_str, nom=True, conf_col=2)
            # PRO
            eng_pro_bio_str = run_tagger(eng_pro_model, eng_pro_vocabs, bio_str)
            eng_pro_tab_str = convert_bio2tab(eng_pro_bio_str, pro=True, conf_col=2)
            # Merge NAM, NOM, PRO
            eng_all_bio_str = merge_bio(eng_nam_bio_str, eng_nom_bio_str)
            eng_all_bio_str = merge_bio(eng_all_bio_str, eng_pro_bio_str)
            # eng_all_tab_str = '{}\n{}\n{}'.format(eng_nam_tab_str.strip('\n'),
            #                                       eng_nom_tab_str.strip('\n'),
            #                                       eng_pro_tab_str.strip('\n'))
            # BIO to CFET json
            eng_fet_json_str = bio2cfet(eng_all_bio_str)
            eng_fet_tsv_str = run_classifier(eng_fet_model, eng_fet_vocabs, eng_fet_json_str)
            result = {
                'bio': eng_all_bio_str,
                'nam_tab': eng_nam_tab_str,
                'nom_tab': eng_nom_tab_str,
                'pro_tab': eng_pro_tab_str,
                'tsv': eng_fet_tsv_str
            }
            return jsonify(result)
        elif lang_code == 'ru':
            # NAM
            rus_nam_5type_bio_str = run_tagger(rus_nam_5type_model, rus_nam_5type_vocabs,
                                        bio_str, elmo=False)
            rus_nam_wv_bio_str = run_tagger(rus_nam_wv_model, rus_nam_wv_vocabs, bio_str,
                                     elmo=False)
            rus_nam_bio_str = merge_bio(rus_nam_5type_bio_str, rus_nam_wv_bio_str)
            rus_nam_tab_str = convert_bio2tab(rus_nam_bio_str, conf_col=2)
            # NOM
            rus_nom_tab_str = extract_nominal(ltf_str, rsd_str, 'ru',
                                              nominal_type_dir, doc_id)
            rus_nom_bio_str = tab2bio(rus_nom_tab_str, bio_str, True)
            # Merge NAM, NOM
            rus_all_bio_str = merge_bio(rus_nam_bio_str, rus_nom_bio_str)
            # rus_all_tab_str = '{}\n{}\n'.format(rus_nam_tab_str.strip('\n'),
            #                                     rus_nom_tab_str.strip('\n'))
            # BIO to CFET json
            rus_fet_json_str = bio2cfet(rus_all_bio_str)
            rus_fet_tsv_str = run_classifier(rus_fet_model, rus_fet_vocabs,
                                             rus_fet_json_str)
            result = {
                'bio': rus_all_bio_str,
                'nam_tab': rus_nam_tab_str,
                'nom_tab': rus_nom_tab_str,
                'tsv': rus_fet_tsv_str
            }
            return jsonify(result)
        elif lang_code == 'uk':
            # NAM
            ukr_nam_5type_bio_str = run_tagger(ukr_nam_5type_model, ukr_nam_5type_vocabs,
                                        bio_str, elmo=False)
            ukr_nam_wv_bio_str = run_tagger(ukr_nam_wv_model, ukr_nam_wv_vocabs,
                                     bio_str, elmo=False)
            ukr_nam_bio_str = merge_bio(ukr_nam_5type_bio_str, ukr_nam_wv_bio_str)
            ukr_nam_tab_str = convert_bio2tab(ukr_nam_bio_str, conf_col=2)
            # NOM
            ukr_nom_tab_str = extract_nominal(ltf_str, rsd_str, 'uk',
                                              nominal_type_dir, doc_id)
            ukr_nom_bio_str = tab2bio(ukr_nom_tab_str, bio_str, True)

            # Merge NAM, NOM
            ukr_all_bio_str = merge_bio(ukr_nam_bio_str, ukr_nom_bio_str)
            # ukr_all_tab_str = '{}\n{}\n'.format(ukr_nam_tab_str.strip('\n'),
            #                                     ukr_nom_tab_str.strip('\n'))
            # BIO to CFET json
            ukr_fet_json_str = bio2cfet(ukr_all_bio_str)
            ukr_fet_tsv_str = run_classifier(ukr_fet_model, ukr_fet_vocabs,
                                             ukr_fet_json_str)
            result = {
                'bio': ukr_all_bio_str,
                'nam_tab': ukr_nam_tab_str,
                'nom_tab': ukr_nom_tab_str,
                'tsv': ukr_fet_tsv_str
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'unknown language code: {}'.format(lang_code)}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/typing', methods=['POST'])
def typing():
    try:
        json_str = request.form['json']
        lang_code = request.form['lang']
        if lang_code == 'en':
            tsv_str = run_classifier(eng_fet_model, eng_fet_vocabs, json_str)
            return jsonify({'tsv': tsv_str})
        elif lang_code == 'ru':
            tsv_str = run_classifier(rus_fet_model, rus_fet_vocabs, json_str)
            return jsonify({'tsv': tsv_str})
        elif lang_code == 'uk':
            tsv_str = run_classifier(ukr_fet_model, ukr_fet_vocabs, json_str)
            return jsonify({'tsv': tsv_str})
        else:
            return jsonify({'error': 'unknown language code: {}'.format(lang_code)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
