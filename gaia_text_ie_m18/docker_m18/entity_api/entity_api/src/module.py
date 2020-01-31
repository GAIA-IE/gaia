from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F


def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, init=None):
        super().__init__(in_features, out_features, bias=bias)
        self.init_weight(init)

    def init_weight(self, init=None):
        if init == 'orthogonal':
            I.orthogonal_(self.weight)
        elif init == 'xavier_normal':
            I.xavier_normal_(self.weight)


class Highway(nn.Module):
    def __init__(self, size, activation='relu', linear=False):
        super().__init__()
        self.size = size
        self.activation = getattr(torch, activation)
        self.use_linear = linear
        if self.use_linear:
            self.linear = nn.Linear(size, size)
        self.non_linear = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)

    def forward(self, x):
        gate = self.gate(x).sigmoid()
        linear = self.linear(x) if self.use_linear else x
        non_linear = self.activation(self.non_linear(x))
        return gate * linear + (1 - gate) * non_linear


class CharCNN(nn.Module):
    """Character-level CNNs that generate a character-level representation for
    each word from its compositional characters.
    """

    def __init__(self, embedding_num, embedding_dim, filters, dropout=0,
                 padding_idx=0, activation='tanh'):
        super(CharCNN, self).__init__()

        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.output_size = sum([x[1] for x in filters])
        self.filters = filters
        self.activation = getattr(torch, activation)

        self.char_embed = nn.Embedding(
            embedding_num, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, x[1], (x[0], embedding_dim))
                                    for x in filters])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        inputs_embed = self.char_embed.forward(inputs)
        inputs_embed = inputs_embed.unsqueeze(1)

        conv_outputs = [self.activation(conv.forward(inputs_embed)).squeeze(3)
                        for conv in self.convs]
        conv_outputs_max = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(conv_outputs_max, 1)
        outputs = self.dropout(outputs)
        return outputs


class CharCNNFF(nn.Module):
    def __init__(self, embedding_num, embedding_dim, filters,
                 padding_idx=0, output_size=None, activation='tanh'):
        super(CharCNNFF, self).__init__()

        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.conv_output_size = sum([x[1] for x in filters])
        self.output_size = output_size if output_size else self.conv_output_size
        self.filters = filters
        self.activation = getattr(torch, activation)

        self.char_embed = nn.Embedding(embedding_num, embedding_dim,
                                       padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, x[1], (x[0], embedding_dim))
                                    for x in filters])
        self.linear = nn.Linear(self.conv_output_size, self.output_size)

    def forward(self, inputs):
        inputs_embed = self.char_embed.forward(inputs)
        inputs_embed = inputs_embed.unsqueeze(1)

        conv_outputs = [self.activation(conv.forward(inputs_embed)).squeeze(3)
                        for conv in self.convs]
        conv_outputs_max = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(conv_outputs_max, 1)
        outputs = self.linear(outputs)
        return outputs


class CharCNNHW(nn.Module):
    def __init__(self, embedding_num, embedding_dim, filters,
                 padding_idx=0, activation='tanh', hw_activation='tanh'):
        super(CharCNNHW, self).__init__()

        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.conv_output_size = sum([x[1] for x in filters])
        self.output_size = self.conv_output_size
        self.filters = filters
        self.activation = getattr(torch, activation)

        self.char_embed = nn.Embedding(embedding_num, embedding_dim,
                                       padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, x[1], (x[0], embedding_dim))
                                    for x in filters])
        self.linear = Highway(self.output_size, hw_activation)

    def forward(self, inputs):
        inputs_embed = self.char_embed.forward(inputs)
        inputs_embed = inputs_embed.unsqueeze(1)

        conv_outputs = [self.activation(conv.forward(inputs_embed)).squeeze(3)
                        for conv in self.convs]
        conv_outputs_max = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(conv_outputs_max, 1)
        outputs = self.linear(outputs)
        return outputs


class LSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{k: v for k, v in kwargs.items()
                                   if k != 'forget_bias'})

        self.forget_bias = kwargs.get('forget_bias', 0.0)
        self.output_size = kwargs.get('hidden_size') * \
                           (2 if kwargs.get('bidirectional', False) else 1)
        self.initialize()

    def initialize(self):
        for n, p in self.named_parameters():
            if 'bias' in n:
                bias_size = p.size(0)
                p.data[bias_size // 4:bias_size // 2].fill_(self.forget_bias)


class CRF(nn.Module):
    def __init__(self, label_vocab, tag_scheme='bioes'):
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        self.same_type = self.map_same_types()
        self.tag_scheme = tag_scheme

        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):

        self.transition.data[:, self.end] = -10000.0
        self.transition.data[self.start, :] = -10000.0

        for label, label_idx in self.label_vocab.items():
            if label.startswith('I-') or label.startswith('E-'):
                self.transition.data[label_idx, self.start] = -10000.0
            if label.startswith('B-') or label.startswith('I-'):
                self.transition.data[self.end, label_idx] = -10000.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-')

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-')

                if self.tag_scheme == 'bioes':
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[
                        label_to_idx, label_from_idx] = -10000.0

    def map_same_types(self):
        label_vocab = self.label_vocab
        label_cluster = defaultdict(list)
        for label, label_idx in label_vocab.items():
            label_type = label.split('-')[1] if '-' in label else label
            label_cluster[label_type].append(label_idx)
        same_type = {}
        for label_idxs in label_cluster.values():
            for label_idx in label_idxs:
                same_type[label_idx] = label_idxs
        return same_type

    def pad_logits(self, logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, label_num = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -10000.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # A tensor of size batch_size * (seq_len + 2)
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size,
                                          self.label_size)
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        # score of jumping to a tag
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        """Checked"""
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        """Checked"""
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, seq_len, feat_dim = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -10000.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score(self, logits, labels):
        batch_size, seq_len, feat_dim = logits.size()
        logits = logits.softmax(dim=2)
        scores = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                score = sum([logits[i][j][k] for k in
                             self.same_type[int(labels[i][j])]])
                scores[i].append(score)
                # score = ','.join(['{:.2f}'.format(k) for k in logits[i][j]])
                # scores[i].append(score)
        return scores


    def calc_conf_score_(self, logits, labels):
        batch_size, seq_len, feat_dim = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label

        # return logits.new_tensor(scores)
        return scores
