import json
import re
import torch
import src.constant as C
from torch.utils.data import Dataset
from allennlp.modules.elmo import batch_to_ids

DIGIT_PATTERN = re.compile('\d')


def bio_to_bioes(labels):
    """Convert a sequence of BIO labels to BIOES labels.
    :param labels: A list of labels.
    :return: A list of converted labels.
    """
    label_len = len(labels)
    labels_bioes = []
    for idx, label in enumerate(labels):
        next_label = labels[idx + 1] if idx < label_len - 1 else 'O'
        if label == 'O':
            labels_bioes.append('O')
        elif label.startswith('B-'):
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('S-' + label[2:])
        else:
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('E-' + label[2:])
    return labels_bioes


def mask_to_distance(mask, mask_len, decay=.1):
    if 1 not in mask:
        return [0] * mask_len
    start = mask.index(1)
    end = mask_len - list(reversed(mask)).index(1)
    dist = [0] * mask_len
    for i in range(start):
        dist[i] = max(0, 1 - (start - i - 1) * decay)
    for i in range(end, mask_len):
        dist[i] = max(0, 1 - (i - end) * decay)
    return dist


class BioDataset(Dataset):
    def __init__(self,
                 span_col=0,
                 token_col=1,
                 label_col=-1,
                 separator=' ',
                 max_seq_len=-1,
                 min_char_len=4,
                 max_char_len=50,
                 gpu=False,
                 test_mode=False):
        """
        :param span_col: Span column index (doc id, start and end offsets).
        :param token_col: Token column index.
        :param label_col: Label column index.
        :param separator: Column separator.
        :param max_seq_len: Max sequence length (-1: no limit).
        :param gpu: Use GPU.
        :param test_mode: If test mode=True, the label column will be ignore and
        all tokens have a 'O' label. This option is designed for files without
        annotations (usu in the prediction phase).
        """
        self.span_col = span_col
        self.token_col = token_col
        self.label_col = label_col
        self.separator = separator

        self.max_seq_len = max_seq_len
        self.min_char_len = min_char_len
        self.max_char_len = max_char_len
        self.gpu = gpu
        self.test_mode = test_mode
        self.data = []

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def process(self, bio_str,
                vocabs, fallback=False):
        self.data = []
        token_vocab = vocabs['token']
        char_vocab = vocabs['char']
        fallback_vocab = vocabs['fallback']
        sentences = bio_str.strip().split('\n\n')
        for sentence in sentences:
            lines = sentence.split('\n')
            tokens, spans = [], []
            for line in lines:
                segments = line.split(self.separator)
                tokens.append(segments[self.token_col].strip())
                spans.append(segments[self.span_col])
            # numberize tokens
            if fallback:
                token_ids = []
                for token in tokens:
                    if token in token_vocab:
                        token_ids.append(token_vocab[token])
                    else:
                        token_lower = token.lower()
                        token_zero = re.sub(DIGIT_PATTERN, '0', token_lower)
                        if token_lower in fallback_vocab:
                            token_ids.append(token_vocab[fallback_vocab[token_lower]])
                        elif token_zero in fallback_vocab:
                            token_ids.append(token_vocab[fallback_vocab[token_zero]])
                        else:
                            token_ids.append(C.UNK_INDEX)
            else:
                token_ids = [token_vocab.get(token, C.UNK_INDEX)
                             for token in tokens]
            # numberize characters
            char_ids = [
                [char_vocab.get(c, C.UNK_INDEX) for c in t][:self.max_char_len]
                for t in tokens]
            elmo_ids = batch_to_ids([tokens])[0].tolist()
            self.data.append((token_ids, char_ids, elmo_ids,
                         tokens, spans))

    def batch_process(self, batch):
        pad = C.PAD_INDEX
        # sort instances in decreasing order of sequence lengths
        # batch.sort(key=lambda x: len(x[0]), reverse=True)
        batch = sorted(enumerate(batch), key=lambda x: len(x[1][0]),
                       reverse=True)
        ori_indices = [i[0] for i in batch]
        batch = [i[1] for i in batch]
        # sequence lengths
        seq_lens = [len(x[0]) for x in batch]
        max_seq_len = max(seq_lens)
        # character lengths
        max_char_len = self.min_char_len
        for seq in batch:
            for chars in seq[1]:
                if len(chars) > max_char_len:
                    max_char_len = len(chars)
        # padding sequences
        batch_token_ids = []
        batch_char_ids = []
        batch_elmo_ids = []
        batch_tokens = []
        batch_spans = []
        for inst in batch:
            token_ids, char_ids, elmo_ids, tokens, spans = inst
            seq_len = len(token_ids)
            pad_num = max_seq_len - seq_len
            batch_token_ids.append(token_ids + [pad] * pad_num)
            batch_char_ids.extend(
                # pad each word
                [x + [pad] * (max_char_len - len(x)) for x in char_ids] +
                # pad each sequence
                [[pad] * max_char_len for _ in range(pad_num)])
            batch_tokens.append(tokens)
            batch_spans.append(spans)
            batch_elmo_ids.append(elmo_ids + [[pad] * C.ELMO_MAX_CHAR_LEN
                                              for _ in range(pad_num)])

        if self.gpu:
            batch_token_ids = torch.cuda.LongTensor(batch_token_ids)
            batch_char_ids = torch.cuda.LongTensor(batch_char_ids)
            seq_lens = torch.cuda.LongTensor(seq_lens)
            batch_elmo_ids = torch.cuda.LongTensor(batch_elmo_ids)
        else:
            batch_token_ids = torch.LongTensor(batch_token_ids)
            batch_char_ids = torch.LongTensor(batch_char_ids)
            seq_lens = torch.LongTensor(seq_lens)
            batch_elmo_ids = torch.LongTensor(batch_elmo_ids)

        return (batch_token_ids, batch_char_ids, batch_elmo_ids,
                seq_lens, batch_tokens, batch_spans, ori_indices)


class FetDataset(Dataset):
    def __init__(self, gpu=False):
        self.gpu = gpu
        self.data = []

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def tokens_to_ids(self, tokens, token_stoi, form_mapping):
        token_nbz = []
        for token in tokens:
            if token in token_stoi:
                token_nbz.append(token_stoi[token])
            else:
                token_lower = token.lower()
                token_zero = DIGIT_PATTERN.sub('0', token_lower)
                if token_lower in form_mapping:
                    token_nbz.append(token_stoi[form_mapping[token_lower]])
                elif token_zero in form_mapping:
                    token_nbz.append(token_stoi[form_mapping[token_zero]])
                else:
                    token_nbz.append(C.UNK_INDEX)
        return token_nbz

    def process(self, json_str, vocabs):
        insts = [json.loads(s) for s in json_str.split('\n')]
        token_stoi = vocabs['token']
        form_mapping = vocabs['form']

        self.data = []
        for inst in insts:
            tokens, annotations = inst['tokens'], inst['annotations']
            tokens = [C.TOK_REPLACEMENT.get(t, t) for t in tokens]
            token_ids = self.tokens_to_ids(tokens, token_stoi, form_mapping)
            elmo_ids = batch_to_ids([tokens])[0].tolist()
            seq_len = len(tokens)
            men_masks, ctx_masks, men_ids, mentions, dists = [], [], [], [], []
            for anno in annotations:
                start, end  = anno['start'], anno['end']
                men_mask = [1 if i >= start and i < end else 0 for i in range(seq_len)]
                ctx_mask = [1 if i < start or i >= end else 0 for i in range(seq_len)]
                men_masks.append(men_mask)
                ctx_masks.append(ctx_mask)
                dists.append(mask_to_distance(men_mask, seq_len))
                men_ids.append(anno['mention_id'])
                mentions.append(anno['mention'])
            self.data.append((token_ids, elmo_ids, men_masks, ctx_masks, dists,
                              tokens, men_ids, mentions, seq_len))

    def batch_process(self, batch):
        batch.sort(key=lambda x: x[-1], reverse=True)

        seq_lens = [x[-1] for x in batch]
        max_seq_len = max(seq_lens)

        batch_token_ids = []
        batch_elmo_ids = []
        batch_men_mask, batch_ctx_mask = [], []
        batch_dist = []
        batch_gather = []
        batch_men_ids = []
        batch_mentions = []
        for inst_idx, inst in enumerate(batch):
            (token_ids, elmo_ids, men_masks, ctx_masks, dists, tokens, men_ids, mentions,
             seq_len) = inst
            pad_num = max_seq_len - seq_len
            batch_token_ids.append(token_ids + [0] * pad_num)
            batch_gather.extend([inst_idx] * len(mentions))
            batch_elmo_ids.append(elmo_ids + [[0] * C.ELMO_MAX_CHAR_LEN
                                              for _ in range(pad_num)])
            for men_mask in men_masks:
                batch_men_mask.append(men_mask + [0] * pad_num)
            for ctx_mask in ctx_masks:
                batch_ctx_mask.append(ctx_mask + [0] * pad_num)
            for dist in dists:
                batch_dist.append(dist + [0] * pad_num)
            batch_men_ids.extend(men_ids)
            batch_mentions.extend(mentions)

        if self.gpu:
            batch_token_ids = torch.cuda.LongTensor(batch_token_ids)
            batch_men_mask = torch.cuda.FloatTensor(batch_men_mask)
            batch_ctx_mask = torch.cuda.FloatTensor(batch_ctx_mask)
            batch_dist = torch.cuda.FloatTensor(batch_dist)
            batch_gather = torch.cuda.LongTensor(batch_gather)
            seq_lens = torch.cuda.LongTensor(seq_lens)
        else:
            batch_token_ids = torch.LongTensor(batch_token_ids)
            batch_men_mask = torch.FloatTensor(batch_men_mask)
            batch_ctx_mask = torch.FloatTensor(batch_ctx_mask)
            batch_dist = torch.FloatTensor(batch_dist)
            batch_gather = torch.LongTensor(batch_gather)
            seq_lens = torch.LongTensor(seq_lens)
        return (batch_token_ids, batch_men_mask, batch_ctx_mask, batch_dist,
                batch_gather, batch_men_ids, batch_mentions, seq_lens)






