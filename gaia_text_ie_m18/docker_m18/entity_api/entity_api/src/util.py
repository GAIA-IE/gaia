import sys
import json
import xml.etree.ElementTree as ET
from collections import defaultdict


def convert_result(results, to_bio=True, separator=' ', conf=True):
    def bioes_2_bio_tag(tag):
        if tag.startswith('S-'):
            tag = 'B-' + tag[2:]
        elif tag.startswith('E-'):
            tag = 'I-' + tag[2:]
        return tag

    bio_str = ''
    if conf:
        for p_b, t_b, l_b, s_b, c_b in results:
            for p_s, t_s, l_s, s_s, c_s in zip(p_b, t_b, l_b, s_b, c_b):
                p_s = p_s[:l_s]
                c_s = c_s[:l_s]
                for p, t, s, c in zip(p_s, t_s, s_s, c_s):
                    if to_bio:
                        p = bioes_2_bio_tag(p)
                    c = c.item()
                    bio_str += separator.join(
                        [str(i) for i in [t, s, c, 'O', p]]) + '\n'
                bio_str += '\n'
    else:
        for p_b, t_b, l_b, s_b in results:
            for p_s, t_s, l_s, s_s in zip(p_b, t_b, l_b, s_b):
                p_s = p_s[:l_s]
                for p, t, s in zip(p_s, t_s, s_s):
                    if to_bio:
                        p = bioes_2_bio_tag(p)
                    bio_str += separator.join(
                        [str(i) for i in [t, s, 'O', p]]) + '\n'
                bio_str += '\n'

    return bio_str


def eng_nam_post_process(input_bio, separator=' '):
    veh = ['mh17', 'mh-17', 'mh370', 'mh-370']
    output_bio = ''
    lines = input_bio.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            output_bio += line + '\n'
            continue
        items = line.split(separator)
        if any([items[0].lower().startswith(v) for v in veh]):
            items[-1] = 'B-VEH'
        output_bio += separator.join(items) + '\n'
    return output_bio.strip('\n')


def bio2tab(bio_str, nom=False, pro=False, conf_col=-1, separator=' '):
    print('=> bio2tab...')
    bio_sents = bio_str.split('\n\n')

    tab_result = []
    current_doc_id = ''
    label_index = 0
    b_tag_count = 0
    num_tokens = 0
    num_sents = 0
    doc_set = set()
    for sent in bio_sents:
        sent = sent.strip()
        if not sent:
            continue
        num_sents += 1
        sent_mentions = []
        tokens = sent.splitlines()
        current_mention = []
        for i, t in enumerate(tokens):
            num_tokens += 1
            t_split = t.split(separator)
            text = t_split[0]
            doc_id, offset = t_split[1].split(':')
            start_char, end_char = offset.split('-')
            pred = t_split[-1].split('-')
            conf = 1.0 if conf_col == -1 else float(t_split[conf_col])
            if len(pred) == 2:
                pred_tag, pred_type = pred
            else:
                pred_tag, pred_type = ('O', None)

            if pred_tag == 'O':
                if current_mention:
                    sent_mentions.append(current_mention)
                    current_mention = []
            elif pred_tag == 'B':
                b_tag_count += 1
                if current_mention:
                    sent_mentions.append(current_mention)
                current_mention = [(text, doc_id, start_char,
                                    end_char, pred_type, conf)]
            elif pred_tag == 'I' and current_mention:
                current_mention.append((text, doc_id, start_char,
                                        end_char, pred_type, conf))
            if i == len(tokens) - 1 and current_mention:
                sent_mentions.append(current_mention)

        for i, mention in enumerate(sent_mentions):
            mention_text = ''
            entity_type = ''
            mention_start_char = 0
            mention_end_char = 0
            mention_conf = 0
            for j, token in enumerate(mention):
                text, doc_id, start_char, end_char, pred_type, conf = token
                mention_conf += conf
                if j == 0:
                    mention_text += text
                    mention_start_char = int(start_char)
                else:
                    mention_text += ' ' * (int(start_char) -
                                           int(mention_end_char)
                                           - 1) + text
                mention_end_char = int(end_char)
                entity_type = pred_type

            if len(mention_text) != (mention_end_char -
                                     mention_start_char) + 1:
                print(mention_text, 'mention text offset error!')
                continue

            if doc_id != current_doc_id:
                current_doc_id = doc_id
                label_index = 0

            if nom:
                mention_type = 'NOM'
            elif pro:
                mention_type = 'PRO'
            else:
                mention_type = 'NAM'

            tab_line = '\t'.join(['bio2tab',
                                  '%s-%d' % (doc_id, label_index),
                                  mention_text,
                                  '%s:%s-%s' % (doc_id,
                                                mention_start_char,
                                                mention_end_char),
                                  'NIL',
                                  entity_type,
                                  mention_type,
                                  str(mention_conf / len(mention)),
                                  # '1.0'
                                  ])
            tab_result.append(tab_line)

            label_index += 1

        sys.stdout.write('%d docs, %d sentences, %d tokens processed.\r' %
                         (len(doc_set), num_sents, num_tokens))
        sys.stdout.flush()

    print('%d docs, %d sentences, %d tokens processed in total.' %
          (len(doc_set), num_sents, num_tokens))

    if b_tag_count != len(tab_result):
        print('number of B tag in bio and tab entries does not match:%d vs %d'
              % (b_tag_count, len(tab_result)))

    tab_str = '\n'.join(tab_result) + '\n'

    return tab_str


def convert_bio2tab(input_bio, nom=False, pro=None,
                    conf_col=-1, separator=' '):
    tab_str = bio2tab(input_bio, nom, pro, conf_col, separator)
    return tab_str.strip('\n')


def sent_in_bio(input_bio, separator=' '):
    sent = []
    for line in input_bio.splitlines():
        line = line.strip()
        if line:
            sent.append(line.split(separator))
        elif sent:
            yield sent
            sent = []


def names_in_sent(sent):
    cur_name = []
    for idx, line in enumerate(sent):
        label = line[-1]
        if label == 'O':
            if cur_name:
                yield cur_name
            cur_name = []
        elif label.startswith('B-'):
            if cur_name:
                yield  cur_name
            cur_name = [(idx, line)]
        else:
            cur_name.append((idx, line))
    if cur_name:
        yield cur_name


def merge_bio(input_bio_1, input_bio_2, separator=' '):
    not_overlapped = 0
    conflict_label = 0
    output_bio = ''
    for sent_1, sent_2 in zip(sent_in_bio(input_bio_1, separator),
                              sent_in_bio(input_bio_2, separator)):
        # all spans should be equal
        assert [t[1] for t in sent_1] == [t[1] for t in sent_2]

        for name in names_in_sent(sent_2):
            start, end = name[0][0], name[-1][0]
            conflict = False
            for i in range(start, end + 1):
                if sent_1[i][-1] != 'O':
                    conflict = True
                    break
            if conflict:
                conflict_label += 1
                print('conflict: {}'.format(sent_1[0][1]))
            else:
                for token in name:
                    sent_1[token[0]][-1] = token[1][-1]
        output_bio += '\n'.join([separator.join(i) for i in sent_1]) + '\n\n'

    print(
        '%d tokens are not overlapped between bio_a and bio_b' % not_overlapped)
    print('%d labels are conflicted.' % conflict_label)
    return output_bio


def ltf2bio(input_ltf, doc_id):
    def load_ltf(ltf_str):
        doc_tokens = []
        root = ET.fromstring(ltf_str)
        doc_id = root.find('DOC').get('id')
        for seg in root.find('DOC').find('TEXT').findall('SEG'):
            sent_tokens = []
            seg_text = seg.find('ORIGINAL_TEXT').text
            seg_start = int(seg.get('start_char'))
            for token in seg.findall('TOKEN'):
                token_text = token.text
                start_char = int(token.get('start_char'))
                end_char = int(token.get('end_char'))

                assert seg_text[
                       start_char - seg_start:end_char - seg_start + 1] == token_text, \
                    'ltf2bio load_ltf token offset error.'

                sent_tokens.append((token_text, doc_id, start_char, end_char))
            doc_tokens.append(sent_tokens)

        return doc_tokens

    doc_tokens = load_ltf(input_ltf)
    bio = []
    for sent in doc_tokens:
        sent_res = []
        for token in sent:
            t_text = token[0]
            if not t_text.strip():
                continue
            if t_text is None:
                t_text = ''
            t_start_char = token[2]
            t_end_char = token[3]
            sent_res.append(' '.join([t_text, '{}:{}-{}'.format(doc_id,
                                                                t_start_char,
                                                                t_end_char)]))
        bio.append('\n'.join(sent_res))

    return '\n\n'.join(bio)


def tab2bio(input_tab, input_bio, test_mode=False):
    bio_str = ''
    annotation = defaultdict(list)
    for line in input_tab.splitlines():
        segs = line.split('\t')
        span = segs[3]
        doc_id = span.split(':')[0]
        start, end = span.split(':')[1].split('-')
        start, end = int(start), int(end)
        type = segs[5]
        annotation[doc_id].append((start, end, type))

    for line in input_bio.splitlines():
        if line:
            segs = line.split()
            token, span = segs[0], segs[1]
            # if test_mode:
            #     token, span, _ = segs[0], segs[1], segs[-1]
            #     # token, span = segs[0], segs[1]
            # else:
            #     token, span, _, _ = segs[0], segs[1], segs[-2], segs[-1]
            doc_id = span.split(':')[0]
            start, end = span.split(':')[1].split('-')
            start, end = int(start), int(end)
            if doc_id not in annotation:
                bio_str += '{} {} O\n'.format(token, span)
            else:
                found = False
                for s, e, t in annotation[doc_id]:
                    if s == start and e == end:
                        # w.write('{} {} {} B-{}\n'.format(token, span, gold, t))
                        bio_str += '{} {} B-{}\n'.format(token, span, t)
                        found = True
                        break
                    elif s == start:
                        # w.write('{} {} {} B-{}\n'.format(token, span, gold, t))
                        bio_str += '{} {} B-{}\n'.format(token, span, t)
                        found = True
                        break
                    elif e == end:
                        # w.write('{} {} {} I-{}\n'.format(token, span, gold, t))
                        bio_str += '{} {} I-{}\n'.format(token, span, t)
                        found = True
                        break
                    elif s < start and end < e:
                        # w.write('{} {} {} I-{}\n'.format(token, span, gold, t))
                        bio_str += '{} {} I-{}\n'.format(token, span, t)
                        found = True
                        break
                if not found:
                    # w.write('{} {} {} O\n'.format(token, span, gold))
                    bio_str += '{} {} O\n'.format(token, span)
        else:
            bio_str += '\n'
    return bio_str.strip('\n')


def bio2cfet(input_bio, token_col=0, span_col=1, label_col=-1, separator=' '):
    def convert_sent(sent, token_col=0, span_col=1, label_col=-1):
        mentions = []
        mention = []
        for idx, line in enumerate(sent):
            token, span, label = line[token_col], line[span_col], line[
                label_col]
            if label == 'O':
                if mention:
                    mentions.append(mention)
                    mention = []
            else:
                doc_id, offsets = span.split(':')
                start, end = offsets.split('-')
                start, end = int(start), int(end)
                if label.startswith('B-'):
                    if mention:
                        mentions.append(mention)
                        mention = []
                mention.append((token, doc_id, start, end, idx))
        if mention:
            mentions.append(mention)

        if len(mentions) == 0:
            return None

        annotations = []
        for mention in mentions:
            text = ' '.join([t[0] for t in mention])
            mention_id = '{}:{}-{}'.format(mention[0][1],  # doc_id
                                           mention[0][2],  # start offset
                                           mention[-1][3],  # end offset
                                           )
            start, end = mention[0][-1], mention[-1][-1] + 1
            annotations.append({
                'mention': text,
                'mention_id': mention_id,
                'start': start,
                'end': end
            })
        tokens = [t[token_col] for t in sent]
        return {
            'tokens': tokens,
            'annotations': annotations
        }

    sent = []
    output_json = ''
    for line in input_bio.splitlines():
        if line:
            sent.append(line.split(separator))
        else:
            if sent:
                sent = convert_sent(sent, token_col, span_col, label_col)
                if sent is not None:
                    output_json += json.dumps(sent) + '\n'
                sent = []
    if sent:
        sent = convert_sent(sent, token_col, span_col, label_col)
        if sent is not None:
            output_json += json.dumps(sent) + '\n'

    return output_json.strip('\n')


def restore_order(items, indices):
    items_new = []
    for item in items:
        item = sorted([(i, v) for v, i in zip(item, indices)],
                      key=lambda x: x[0])
        item = [v for i, v in item]
        items_new.append(item)
    return items_new