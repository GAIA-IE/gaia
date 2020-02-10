from langdetect import detect, DetectorFactory
import shutil
import os
import argparse
import codecs

def detect_languages(input_folder, output_folder):
    DetectorFactory.seed = 0

    # if os.path.exists(output_folder) is True:
    #     shutil.rmtree(output_folder)

    # os.mkdir(output_folder)

    rsd_input_folder = os.path.join(input_folder, 'rsd')
    ltf_input_folder = os.path.join(input_folder, 'ltf')
    for one_file in os.listdir(rsd_input_folder):
        one_file_id = one_file.replace('.rsd.txt', '')
        one_rsd_file_path = os.path.join(rsd_input_folder, one_file)
        one_ltf_file_path = os.path.join(ltf_input_folder, '%s.ltf.xml' % one_file_id)
        one_file_content = codecs.open(one_rsd_file_path, 'r', 'utf-8').read()
        candidate_language_id = detect(one_file_content)
        language_folder_ltf = os.path.join(output_folder, candidate_language_id, 'ltf')
        if os.path.exists(language_folder_ltf) is False:
            os.makedirs(language_folder_ltf)
        shutil.copy(one_ltf_file_path, language_folder_ltf)
        language_folder_rsd = os.path.join(output_folder, candidate_language_id, 'rsd')
        if os.path.exists(language_folder_rsd) is False:
            os.makedirs(language_folder_rsd)
        shutil.copy(one_rsd_file_path, language_folder_rsd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help='input directory containing rsd and ltf files')
    parser.add_argument('output_folder', type=str,
                        help='output directory divided by languages')

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    detect_languages(input_folder, output_folder)