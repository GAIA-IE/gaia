__author__ = 'max'


class Sentence(object):
    def __init__(self, words, word_ids, char_seqs, char_id_seqs, segment_id=""):
        self.words = words
        self.word_ids = word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs
        self.segment_id = segment_id

    def length(self):
        return len(self.words)


class DependencyInstance(object):
    def __init__(self, sentence, postags, pos_ids, heads, types, type_ids, segment_id=""):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids
        self.segment_id = segment_id

    def length(self):
        return self.sentence.length()


class NERInstance(object):
    def __init__(self, sentence, postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.chunk_tags = chunk_tags
        self.chunk_ids = chunk_ids
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()