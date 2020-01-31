import os
import codecs
import argparse
import sys
import xml.etree.ElementTree as ET
import xml


def ltf2rsd(ltf_str, dat=False):
    root = ET.fromstring(ltf_str)
    doc = root.find('DOC')
    doc_id = doc.get('id')
    rsd = ''
    prev_sent_end = -1
    for seg in doc.findall('.//SEG'):
        if dat:
            tokens = [tok.text for tok in seg.findall('.//TOKEN')]
            rsd += ' '.join(tokens) + '\n'
        else:
            seg_start = int(seg.get('start_char'))
            seg_end = int(seg.get('end_char'))
            seg_text = seg.find('ORIGINAL_TEXT').text
            seg_id = seg.get('id')

            rsd += '\n' * (seg_start - prev_sent_end - 1) + seg_text
            prev_sent_end = seg_end

            assert rsd[seg_start:seg_end+1] == seg_text, \
                'offset error in %s, %s' % (doc_id, seg_id)

    return rsd


def ltf2rsd_batch(input_ltf, output_rsd, dat):
    ltf_fns = os.listdir(input_ltf)
    for i, fn in enumerate(ltf_fns):
        try:
            if '.ltf.xml' not in fn:
                continue
            ltf_file = os.path.join(input_ltf, fn)

            ltf_str = codecs.open(ltf_file, 'r', 'utf-8').read()

            rsd_str = ltf2rsd(ltf_str, dat)

            if dat:
                rsd_file = os.path.join(output_rsd, fn.replace('.ltf.xml',
                                                               '.rsd_tok.txt'))
            else:
                rsd_file = os.path.join(output_rsd, fn.replace('.ltf.xml',
                                                               '.rsd.txt'))

            write2file(rsd_str, rsd_file)

            sys.stdout.write('%d docs are processed...\r' % i)
            sys.stdout.flush()
        except AssertionError as e:
            print(e)
            continue
        except xml.etree.ElementTree.ParseError as e:
            print(fn, e)
            continue

    print('%d docs are processed in total.' % len(ltf_fns))


def write2file(rsd_str, rsd_file):
    with codecs.open(rsd_file, 'w', 'utf-8') as f:
        f.write(rsd_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', action='store_true', default=False,
                        dest='is_dir',
                        help='Batch processing. When turned on, input and '
                             'output are dirs.')
    parser.add_argument('ltf_input', type=str,
                        help='input ltf file path or directory.')
    parser.add_argument('rsd_output', type=str,
                        help='output rsd file path or directory.')
    parser.add_argument('--dat', action='store_true', default=False,
                        help='use white space as delimiter for each token. one '
                             'sentence per line. used for embedding training')

    args = parser.parse_args()

    input_ltf = args.ltf_input
    output_rsd = args.rsd_output
    dat = args.dat

    if args.is_dir:
        ltf2rsd_batch(input_ltf, output_rsd, dat)
    else:
        ltf_str = codecs.open(input_ltf, 'r', 'utf-8').read()
        rsd_str = ltf2rsd(ltf_str, dat)
        write2file(rsd_str, output_rsd)