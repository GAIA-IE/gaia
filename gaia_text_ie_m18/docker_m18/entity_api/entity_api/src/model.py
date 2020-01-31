import math
import torch
import torch.nn as nn
import src.constant as C
import torch.nn.utils.rnn as R

from src.module import Linear, CRF, CharCNN, CharCNNHW, CharCNNFF, LSTM
from allennlp.modules.elmo import Elmo


class LstmCnnOutElmo(nn.Module):

    def __init__(self,
                 vocabs,
                 elmo_option, elmo_weight,
                 lstm_hidden_size,
                 lstm_dropout=0.5, feat_dropout=0.5, elmo_dropout=0.5,
                 parameters=None,
                 output_bias=True,
                 elmo_finetune=False,
                 tag_scheme='bioes'
                 ):
        super(LstmCnnOutElmo, self).__init__()

        self.vocabs = vocabs
        self.label_size = len(self.vocabs['label'])
        # input features
        self.word_embed = nn.Embedding(parameters['word_embed_num'],
                                        parameters['word_embed_dim'],
                                        padding_idx=C.PAD_INDEX)

        self.elmo = Elmo(elmo_option, elmo_weight,
                         num_output_representations=1,
                         requires_grad=elmo_finetune,
                         dropout=elmo_dropout)
        self.elmo_dim = self.elmo.get_output_dim()

        self.word_dim = self.word_embed.embedding_dim
        self.feat_dim = self.word_dim
        # layers
        self.lstm = LSTM(input_size=self.feat_dim,
                         hidden_size=lstm_hidden_size,
                         batch_first=True,
                         bidirectional=True)
        self.output_linear = Linear(self.lstm.output_size + self.elmo_dim,
                                    self.label_size, bias=output_bias)
        self.crf = CRF(vocabs['label'], tag_scheme=tag_scheme)
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)

    def forward_nn(self, token_ids, elmo_ids, lens, return_hidden=False):
        # word representation
        word_in = self.word_embed(token_ids)
        feats = self.feat_dropout(word_in)

        # LSTM layer
        lstm_in = R.pack_padded_sequence(feats, lens.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)

        # Elmo output
        elmo_out = self.elmo(elmo_ids)['elmo_representations'][0]
        combined_out = torch.cat([lstm_out, elmo_out], dim=2)

        # output linear layer
        linear_out = self.output_linear(combined_out)
        if return_hidden:
            return linear_out, combined_out.tolist()
        else:
            return linear_out, None

    def predict(self, token_ids, elmo_ids, lens,
                return_hidden=False, return_conf_score=False):
        self.eval()
        logits, lstm_out = self.forward_nn(token_ids, elmo_ids, lens,
                                           return_hidden=return_hidden)
        logits_padded = self.crf.pad_logits(logits)
        _scores, preds = self.crf.viterbi_decode(logits_padded, lens)
        if return_conf_score:
            conf_score = self.crf.calc_conf_score(logits, preds)
        else:
            conf_score = None
        preds = preds.data.tolist()
        self.train()
        return preds, lstm_out, conf_score


class LstmCnn(nn.Module):

    def __init__(self,
                 vocabs,
                 char_embed_dim, char_filters, char_feat_dim,
                 lstm_hidden_size,
                 lstm_dropout=0, feat_dropout=0,
                 parameters=None,
                 char_out='none',
                 char_activation='tanh',
                 char_out_activation='tanh',
                 use_char=True,
                 output_bias=True,
                 tag_scheme='bioes'
                 ):
        super(LstmCnn, self).__init__()

        self.vocabs = vocabs
        self.label_size = len(self.vocabs['label'])
        # input features
        self.word_embed = nn.Embedding(parameters['word_embed_num'],
                                       parameters['word_embed_dim'],
                                       padding_idx=C.PAD_INDEX)

        if char_out == 'none':
            self.char_embed = CharCNN(len(vocabs['char']),
                                      char_embed_dim,
                                      char_filters,
                                      activation=char_activation)
        elif char_out == 'ff':
            self.char_embed = CharCNNFF(len(vocabs['char']),
                                        char_embed_dim,
                                        char_filters,
                                        output_size=char_feat_dim,
                                        activation=char_activation)
        elif char_out == 'hw':
            self.char_embed = CharCNNHW(len(vocabs['char']),
                                        char_embed_dim,
                                        char_filters,
                                        activation=char_activation,
                                        hw_activation=char_out_activation)
        self.word_dim = self.word_embed.embedding_dim
        self.char_dim = self.char_embed.output_size
        if use_char:
            self.feat_dim = self.char_dim + self.word_dim
        else:
            self.feat_dim = self.word_dim
        # layers
        self.lstm = LSTM(input_size=self.feat_dim,
                         hidden_size=lstm_hidden_size,
                         num_layers=1,
                         batch_first=True,
                         bidirectional=True)
        self.output_linear = Linear(self.lstm.output_size,
                                    self.label_size, bias=output_bias)
        self.crf = CRF(vocabs['label'], tag_scheme=tag_scheme)
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)
        self.use_char = use_char

    def forward_nn(self, token_ids, char_ids, lens, return_hidden=False):
        batch_size, seq_len = token_ids.size()
        # word representation
        word_in = self.word_embed(token_ids)
        if self.use_char:
            char_in = self.char_embed(char_ids)
            char_in = char_in.view(batch_size, seq_len, self.char_dim)
            feats = torch.cat([word_in, char_in], dim=2)
        else:
            feats = word_in
        feats = self.feat_dropout(feats)

        # LSTM layer
        lstm_in = R.pack_padded_sequence(feats, lens.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)

        # output linear layer
        linear_out = self.output_linear(lstm_out)

        if return_hidden:
            return linear_out, lstm_out.tolist()
        else:
            return linear_out, None

    def predict(self, token_ids, char_ids, lens,
                return_hidden=False, return_conf_score=False):
        self.eval()
        logits, lstm_out = self.forward_nn(token_ids, char_ids, lens,
                                           return_hidden=return_hidden)
        logits_padded = self.crf.pad_logits(logits)
        _scores, preds = self.crf.viterbi_decode(logits_padded, lens)
        if return_conf_score:
            conf_score = self.crf.calc_conf_score(logits, preds)
        else:
            conf_score = None
        preds = preds.data.tolist()
        self.train()
        return preds, lstm_out, conf_score


class AttnFet(nn.Module):
    def __init__(self,
                 vocabs,
                 word_embed_dim,
                 lstm_size=100,
                 embed_dropout=.5,
                 repr_dropout=.2,
                 lstm_dropout=.5,
                 full_attn=False,
                 flat_attn=False,
                 latent=False,
                 latent_size=0,
                 dist=False,
                 svd=None,
                 dist_dropout=.2,
                 embed_num=None,
                 ):
        super(AttnFet, self).__init__()
        self.label_size = label_size = len(vocabs['label'])
        self.word_embed = nn.Embedding(embed_num, word_embed_dim,
                                           padding_idx=C.PAD_INDEX,
                                           sparse=True)
        self.word_embed_dim = word_embed_dim
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.lstm = LSTM(input_size=word_embed_dim,
                         hidden_size=lstm_size,
                         batch_first=True,
                         bidirectional=True)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)
        self.lstm_dim = self.lstm.output_size

        self.feat_dim = self.lstm_dim
        self.attn_dim = self.feat_dim if full_attn else 1
        self.attn_inner_dim = self.feat_dim
        # Mention attention
        if flat_attn:
            self.men_attn_linear_m = Linear(word_embed_dim, word_embed_dim, bias=False)
            self.men_attn_linear_o = Linear(word_embed_dim, self.attn_dim, bias=False)
        else:
            self.men_attn_linear_m = Linear(self.feat_dim, self.attn_inner_dim, bias=False)
            self.men_attn_linear_o = Linear(self.attn_inner_dim, self.attn_dim, bias=False)
        # Context attention
        self.ctx_attn_linear_c = Linear(self.feat_dim, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_m = Linear(self.feat_dim, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_d = Linear(1, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_o = Linear(self.attn_inner_dim,
                                        self.attn_dim, bias=False)
        self.repr_dropout = nn.Dropout(p=repr_dropout)
        if flat_attn:
            self.output_linear = Linear(self.feat_dim + word_embed_dim, label_size, bias=False)
        else:
            self.output_linear = Linear(self.feat_dim * 2, label_size, bias=False)

        self.latent = latent
        if latent_size == 0:
            self.latent_size = int(math.sqrt(label_size))
        else:
            self.latent_size = latent_size
        self.latent_scalar = nn.Parameter(torch.FloatTensor([.1]))
        self.feat_to_latent = Linear(self.feat_dim * 2, self.latent_size, bias=False)
        self.latent_to_label = Linear(self.latent_size, label_size, bias=False)

        # Loss function
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.mse = nn.MSELoss()

        self.flat_attn = flat_attn
        self.dist = dist
        self.dist_dropout = nn.Dropout(p=dist_dropout)


    def forward_nn(self, inputs, men_mask, ctx_mask, dist, gathers, lens):
        embed_outputs = self.word_embed(inputs)
        embed_outputs_d = self.embed_dropout(embed_outputs)
        # Forward LSTM
        lstm_in = R.pack_padded_sequence(embed_outputs_d, lens, batch_first=True)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)

        _, seq_len, feat_dim = lstm_out.size()
        lstm_gathers = gathers.unsqueeze(-1).unsqueeze(-1).expand(-1, seq_len, feat_dim)
        lstm_out = torch.gather(lstm_out, 0, lstm_gathers)
        lstm_out = self.lstm_dropout(lstm_out)

        # Mention representation
        if self.flat_attn:
            mentions = lstm_out * men_mask.unsqueeze(-1).expand_as(lstm_out)
            men_repr = mentions.sum(1) / men_mask.sum(1, keepdim=True).clamp(1)
        else:

            men_attn = self.men_attn_linear_m(lstm_out).tanh()
            men_attn = self.men_attn_linear_o(men_attn)
            men_attn = men_attn + (1.0 - men_mask.unsqueeze(-1)) * -10000.0
            men_attn = men_attn.softmax(1)
            men_repr = (lstm_out * men_attn).sum(1)

        # Context representation
        if self.flat_attn:
            ctx_attn = self.ctx_attn_linear_c(lstm_out).tanh()
            ctx_attn = self.ctx_attn_linear_o(ctx_attn)
        else:
            if self.dist:
                dist = self.dist_dropout(dist)
                ctx_attn = (self.ctx_attn_linear_c(lstm_out) +
                            self.ctx_attn_linear_m(men_repr.unsqueeze(1)) +
                            self.ctx_attn_linear_d(dist.unsqueeze(2))).tanh()
            else:
                ctx_attn = (self.ctx_attn_linear_c(lstm_out) +
                            self.ctx_attn_linear_m(
                                men_repr.unsqueeze(1))).tanh()
            ctx_attn = self.ctx_attn_linear_o(ctx_attn)


        ctx_attn = ctx_attn + (1.0 - ctx_mask.unsqueeze(-1)) * -10000.0
        ctx_attn = ctx_attn.softmax(1)
        ctx_repr = (lstm_out * ctx_attn).sum(1)

        # Classification
        final_repr = torch.cat([men_repr, ctx_repr], dim=1)
        final_repr = self.repr_dropout(final_repr)
        outputs = self.output_linear(final_repr)

        outputs_latent = None
        if self.latent:
            latent_label = self.feat_to_latent(final_repr) #.tanh()
            outputs_latent = self.latent_to_label(latent_label)
            outputs = outputs + self.latent_scalar * outputs_latent

        return outputs, outputs_latent

    def predict_aida(self, inputs, men_mask, ctx_mask, dist, gathers, lens, top_only=True):
        self.eval()
        outputs, _ = self.forward_nn(inputs, men_mask, ctx_mask, dist, gathers, lens)
        scores = outputs.sigmoid()
        if top_only:
            _, highest = outputs.max(dim=1)
            highest = highest.int().tolist()
            preds = outputs.new_zeros(outputs.size()).int()
            for i, h in enumerate(highest):
                preds[i][h] = 1
        else:
            preds = (scores > .5).int()

        self.train()
        return preds, scores