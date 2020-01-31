import xml.etree.ElementTree as ET
import os

from collections import namedtuple


LTFSegment = namedtuple("LTFSegment", ["provenance", "start_offset", "end_offset", "tokens", "token_offsets"])


class LTFUtil(object):
    def __init__(self, ltf_dir):
        super(LTFUtil, self).__init__()
        self.ltf_dir = ltf_dir

    def get_ltf_dir(self):
        return self.ltf_dir

    def parse_offset_str(self, offset_str):
        docid = offset_str[:offset_str.find(':')]
        start = int(offset_str[offset_str.find(':') + 1:offset_str.find('-')])
        end = int(offset_str[offset_str.find('-') + 1:])
        return docid, start, end

    def _iterate_doc_segments(self, doc_name):
        doc_filename = os.path.join(self.ltf_dir, doc_name)
        if not doc_filename.endswith(".ltf.xml"):
            doc_filename += ".ltf.xml"

        tree = ET.parse(doc_filename)
        root = tree.getroot()
        for doc in root:
            for text in doc:
                for seg in text:
                    yield seg

    def get_context(self, offset_str):
        docid, start, end = self.parse_offset_str(offset_str)

        tokens = []

        for seg in self._iterate_doc_segments(docid):
            seg_beg = int(seg.attrib["start_char"])
            seg_end = int(seg.attrib["end_char"])
            if start >= seg_beg and end <= seg_end:
                for token in seg:
                    if token.tag == "TOKEN":
                        tokens.append(token.text)
            if len(tokens) > 0:
                return tokens
        return tokens

    def get_str(self, offset_str):
        docid, start, end = self.parse_offset_str(offset_str)

        tokens = []

        for seg in self._iterate_doc_segments(docid):
            seg_beg = int(seg.attrib["start_char"])
            seg_end = int(seg.attrib["end_char"])
            if start >= seg_beg and end <= seg_end:
                for token in seg:
                    if token.tag == "TOKEN":
                        token_beg = int(token.attrib["start_char"])
                        token_end = int(token.attrib["end_char"])
                        if start <= token_beg and end >= token_end:
                            tokens.append(token.text)
            if len(tokens) > 0:
                return ' '.join(tokens)
        print('[ERROR]can not find the string with offset ', offset_str)
        return None

    def get_doc_segments(self, doc_id):
        for segment in self._iterate_doc_segments(doc_id):
            startchar_offset = int(segment.attrib["start_char"])
            endchar_offset = int(segment.attrib["end_char"])
            tokens = []
            token_offsets = []

            for tok in segment:
                if tok.tag == "TOKEN":
                    token_offsets.append((int(tok.attrib["start_char"]), int(tok.attrib["end_char"])))
                    tokens.append(tok.text)
            yield LTFSegment(
                provenance=doc_id,
                start_offset=startchar_offset,
                end_offset=endchar_offset,
                tokens=tokens,
                token_offsets=token_offsets
            )

    def get_spec_doc_segment(self, doc_id, seg_start_offset, seg_end_offset):
        for segment in self.get_doc_segments(doc_id):
            if segment.start_offset >= seg_start_offset and segment.end_offset <= seg_end_offset:
                return segment
        return None
    # def get_doc_segments(self, doc_id, seg_start_offset=-1, seg_end_offset=-1):
    #     found_segment = False
    #     for segment in self._iterate_doc_segments(doc_id):
    #         startchar_offset = int(segment.attrib["start_char"])
    #         endchar_offset = int(segment.attrib["end_char"])
    #         tokens = []
    #         token_offsets = []
    #
    #         if seg_start_offset >= 0 and seg_end_offset >=0:
    #             if found_segment:
    #                 break
    #
    #             if startchar_offset < seg_start_offset or endchar_offset > seg_end_offset:
    #                 continue
    #             else:
    #                 found_segment = True
    #
    #         for tok in segment:
    #             if tok.tag == "TOKEN":
    #                 token_offsets.append((int(tok.attrib["start_char"]), int(tok.attrib["end_char"])))
    #                 tokens.append(tok.text)
    #         yield LTFSegment(
    #             provenance=doc_id,
    #             start_offset=startchar_offset,
    #             end_offset=endchar_offset,
    #             tokens=tokens,
    #             token_offsets=token_offsets
    #         )

    def get_all_segments(self):
        doc_id_list = [ltf_fname.replace(".ltf.xml", "") for ltf_fname in os.listdir(self.ltf_dir)
                        if ltf_fname.endswith(".ltf.xml")]
        for doc_id in doc_id_list:
            for segment_data in self.get_doc_segments(doc_id):
                yield segment_data

    # def get_context(self, offset_str):
    #     docid, start, end = self.parse_offset_str(offset_str)
    #
    #     tokens = []
    #
    #     tree = ET.parse(os.path.join(self.ltf_dir, docid + '.ltf.xml'))
    #     root = tree.getroot()
    #     for doc in root:
    #         for text in doc:
    #             for seg in text:
    #                 seg_beg = int(seg.attrib["start_char"])
    #                 seg_end = int(seg.attrib["end_char"])
    #                 if start >= seg_beg and end <= seg_end:
    #                     for token in seg:
    #                         if token.tag == "TOKEN":
    #                             tokens.append(token.text)
    #                 if len(tokens) > 0:
    #                     return tokens
    #     return tokens
    #
    # def get_str(self, offset_str):
    #     docid, start, end = self.parse_offset_str(offset_str)
    #
    #     tokens = []
    #
    #     tree = ET.parse(os.path.join(self.ltf_dir, docid + '.ltf.xml'))
    #     root = tree.getroot()
    #     for doc in root:
    #         for text in doc:
    #             for seg in text:
    #                 seg_beg = int(seg.attrib["start_char"])
    #                 seg_end = int(seg.attrib["end_char"])
    #                 if start >= seg_beg and end <= seg_end:
    #                     for token in seg:
    #                         if token.tag == "TOKEN":
    #                             token_beg = int(token.attrib["start_char"])
    #                             token_end = int(token.attrib["end_char"])
    #                             if start <= token_beg and end >= token_end:
    #                                 tokens.append(token.text)
    #                 if len(tokens) > 0:
    #                     return ' '.join(tokens)
    #     print('[ERROR]can not find the string with offset ', offset_str)
    #     return None


if __name__ == '__main__':
    # ltf_dir = '/data/m1/lim22/aida2019/dryrun/source/en'
    ltf_dir = "/data/m1/lim22/aida2019/dryrun_3/source/en"
    ltf_util = LTFUtil(ltf_dir)
    # print(ltf_util.get_str('HC000Q7M9:59-121'))
    print(ltf_util.get_str('HC000Q7M9:59-121'))
    for seg_data in ltf_util.get_doc_segments("HC00002ZN"):
        print(seg_data)
        # break
