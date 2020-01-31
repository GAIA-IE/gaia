import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, batch_size, max_len, embedding, pos_embed_size,
                 pos_embed_num, entype_embd_size, entype_embd_num, class_num,
                 num_hidden, keep_prob):
        super(BiLSTM, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.np = pos_embed_num
        self.den = entype_embd_size
        self.nen = entype_embd_num
        self.d = self.dw + 2 * self.dp + self.den + 1
        self.nr = class_num
        self.dc = num_hidden
        self.keep_prob = keep_prob
        self.n = max_len  # max sequence length
        self.batch_size = batch_size
        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.entype_embedding = nn.Embedding(self.nen, self.den)

        self.enc_lstm = nn.LSTM(self.d, self.dc, 1, batch_first=True,
                                dropout=self.keep_prob, bidirectional=True)
        self.init_lstm = Variable(torch.FloatTensor(2, self.batch_size, self.dc).zero_()).cuda()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.dc * 2, self.nr)
        self.softmax = nn.Softmax()

    def forward(self, x, dist1, dist2, en_type_vec, dp_type_vec, is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        entype_embed = self.entype_embedding(en_type_vec)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed, entype_embed,
                              dp_type_vec.unsqueeze(2)), 2)  # (bz, n, dw + 2*dp + 1)transpose(0, 1)
        features, (h_n, h_c) = self.enc_lstm(x_concat, None)
        last_hidden = features[:, -1, :].view(-1, self.dc * 2)
        logits = self.fc(last_hidden)
        return logits


class SimpleCNN(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, entype_embd_size, entype_embd_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(SimpleCNN, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.den = entype_embd_size
        self.nen = entype_embd_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2  # padding window size, to make the dimension of output after convolution operation same as before
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.entype_embedding = nn.Embedding(self.nen, self.den)

        # self.conv = nn.Conv2d(1, self.dc, (self.k, self.d+self.den), bias=True)  # renewed
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d + self.den), padding=(self.p, 0), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.dc, self.nr)
        # self.fc = nn.Linear(self.dc, self.nr)
        self.softmax = nn.Softmax()

    def forward(self, x, dist1, dist2, en_type_vec, is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        entype_embed = self.entype_embedding(en_type_vec)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed, entype_embed), 2)  # (bz, n, dw + 2*dp + 1)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)
        features = self.tanh(self.conv(x_concat).squeeze())
        features = F.max_pool1d(features, features.size(2)).squeeze()
        features = self.dropout(features)
        features = features.view(-1, self.dc)
        logits = self.fc(features)
        return logits


class SimpleCNNMask(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, entype_embd_size, entype_embd_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(SimpleCNNMask, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.den = entype_embd_size
        self.nen = entype_embd_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2  # padding window size, to make the dimension of output after convolution operation same as before
        self.n = max_len  # max sequence length

        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.entype_embedding = nn.Embedding(self.nen, self.den)

        # self.conv = nn.Conv2d(1, self.dc, (self.k, self.d+self.den), bias=True)  # renewed
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d + self.den), padding=(self.p, 0), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.dc * 3, self.nr)
        # self.fc = nn.Linear(self.dc, self.nr)
        self.softmax = nn.Softmax()

    def forward(self, x, dist1, dist2, en_type_vec, pool_mask_e1, pool_mask, pool_mask_e2,
                is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        entype_embed = self.entype_embedding(en_type_vec)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed, entype_embed), 2)  # (bz, n, dw + 2*dp + 1)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)
        # x_concat = x_embed.unsqueeze(1)
        features = self.tanh(self.conv(x_concat).squeeze())
        features_e1 = F.max_pool1d(features * pool_mask_e1.unsqueeze(1), features.size(2)).squeeze()
        features_e1_e2 = F.max_pool1d(features * pool_mask.unsqueeze(1), features.size(2)).squeeze()
        features_e2 = F.max_pool1d(features * pool_mask_e2.unsqueeze(1), features.size(2)).squeeze()
        features = torch.cat((features_e1, features_e1_e2, features_e2), 1)
        # features = F.max_pool1d(features, features.size(2)).squeeze()
        features = self.dropout(features)
        features = features.view(-1, self.dc * 3)
        # features = features.view(-1, self.dc)
        logits = self.fc(features)
        return logits


class CNNDpMask(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, entype_embd_size, entype_embd_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(CNNDpMask, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.den = entype_embd_size
        self.nen = entype_embd_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2  # padding window size, to make the dimension of output after convolution operation same as before
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.entype_embedding = nn.Embedding(self.nen, self.den)

        # self.conv = nn.Conv2d(1, self.dc, (self.k, self.d+self.den), bias=True)  # renewed
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d + self.den + 1), padding=(self.p, 0), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.dc * 3, self.nr)
        # self.fc = nn.Linear(self.dc, self.nr)
        self.softmax = nn.Softmax()

    def forward(self, x, dist1, dist2, en_type_vec, dp_type_vec, pool_mask_e1, pool_mask, pool_mask_e2,
                is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        entype_embed = self.entype_embedding(en_type_vec)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed, entype_embed,
                              dp_type_vec.unsqueeze(2)), 2)  # (bz, n, dw + 2*dp + 1)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)

        features = self.tanh(self.conv(x_concat).squeeze())
        features_e1 = F.max_pool1d(features * pool_mask_e1.unsqueeze(1), features.size(2)).squeeze()
        features_e1_e2 = F.max_pool1d(features * pool_mask.unsqueeze(1), features.size(2)).squeeze()
        features_e2 = F.max_pool1d(features * pool_mask_e2.unsqueeze(1), features.size(2)).squeeze()
        features = torch.cat((features_e1, features_e1_e2, features_e2), 1)
        features = self.dropout(features)
        features = features.view(-1, self.dc * 3)
        logits = self.fc(features)
        return logits


class AssCNNDpMask(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, entype_embd_size, entype_embd_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(AssCNNDpMask, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.den = entype_embd_size
        self.nen = entype_embd_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2  # padding window size, to make the dimension of output after convolution operation same as before
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.entype_embedding = nn.Embedding(self.nen, self.den)

        # self.conv = nn.Conv2d(1, self.dc, (self.k, self.d+self.den), bias=True)  # renewed
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d + self.den + 1), padding=(self.p, 0), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.45)
        self.ass = nn.Linear(3, 1)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.dc * 3, self.nr)
        # self.fc = nn.Linear(self.dc, self.nr)
        self.softmax = nn.Softmax()

    def forward(self, x, dist1, dist2, en_type_vec, dp_type_vec, pool_mask_e1, pool_mask, pool_mask_e2,
                is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        entype_embed = self.entype_embedding(en_type_vec)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed, entype_embed,
                              dp_type_vec.unsqueeze(2)), 2)  # (bz, n, dw + 2*dp + 1)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)

        features = self.tanh(self.conv(x_concat).squeeze())
        features_e1 = F.max_pool1d(features * pool_mask_e1.unsqueeze(1), features.size(2)).squeeze()
        features_e1_e2 = F.max_pool1d(features * pool_mask.unsqueeze(1), features.size(2)).squeeze()
        features_e2 = F.max_pool1d(features * pool_mask_e2.unsqueeze(1), features.size(2)).squeeze()
        features = torch.cat((features_e1, features_e1_e2, features_e2), 1)
        features1 = self.dropout(features).unsqueeze(2)
        features2 = self.dropout1(features).unsqueeze(2)
        features3 = self.dropout2(features).unsqueeze(2)
        new_feature = torch.cat((features1, features2, features3), 2)
        new_feature = self.ass(new_feature).squeeze()
        new_feature = new_feature.view(-1, self.dc * 3)
        logits = self.fc(new_feature)
        return logits


class BinaryFull(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, entype_embd_size, entype_embd_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(BinaryFull, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.den = entype_embd_size
        self.nen = entype_embd_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2  # padding window size, to make the dimension of output after convolution operation same as before
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.entype_embedding = nn.Embedding(self.nen, self.den)
        self.relation_embedding = nn.Embedding(22, 10)

        # self.conv = nn.Conv2d(1, self.dc, (self.k, self.d+self.den), bias=True)  # renewed
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d + self.den + 1), padding=(self.p, 0), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.45)
        self.ass = nn.Linear(3, 1)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.dc * 3 + 10, self.nr)
        self.fc1 = nn.Linear(self.dc, self.nr)
        # self.fc = nn.Linear(self.dc, self.nr)
        self.softmax = nn.Softmax()

    def forward(self, x, dist1, dist2, en_type_vec, dp_type_vec, pool_mask_e1, pool_mask, pool_mask_e2, x_right,
                is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        entype_embed = self.entype_embedding(en_type_vec)
        x_right_embed = self.relation_embedding(x_right).squeeze()  #(bz, 1, 25)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed, entype_embed,
                              dp_type_vec.unsqueeze(2)), 2)  # (bz, n, dw + 2*dp + 1)
        # (bz, 1, 10)
        # print(x_right_embed.shape)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)

        features = self.tanh(self.conv(x_concat).squeeze())
        # print(features.size())
        features_e1 = F.max_pool1d(features * pool_mask_e1.unsqueeze(1), features.size(2)).squeeze()
        features_e1_e2 = F.max_pool1d(features * pool_mask.unsqueeze(1), features.size(2)).squeeze()
        features_e2 = F.max_pool1d(features * pool_mask_e2.unsqueeze(1), features.size(2)).squeeze()
        new_feature = torch.cat((features_e1, features_e1_e2, features_e2), 1)
        # features1 = self.dropout(features).unsqueeze(2)
        # features2 = self.dropout1(features).unsqueeze(2)
        # features3 = self.dropout2(features).unsqueeze
        # features1 = self.dropout(features).unsqueeze(2)
        # features2 = self.dropout1(features).unsqueeze(2)
        # features3 = self.dropout2(features).unsqueeze(2)
        # new_feature = torch.cat((features1, features2, features3), 2)
        # new_feature = features
        # print(new_feature.shape)
        # new_feature = self.ass(new_feature).squeeze()
        # print(new_feature.shape)
        new_feature = new_feature.view(-1, self.dc * 3)
        # print(new_feature.shape)
        new_feature = torch.cat((new_feature, x_right_embed), 1)
        # print(new_feature.shape)
        # inter = self.fc(new_feature)
        logits = self.fc(new_feature)
        return logits


class CNNDp(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, entype_embd_size, entype_embd_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(CNNDp, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.den = entype_embd_size
        self.nen = entype_embd_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2  # padding window size, to make the dimension of output after convolution operation same as before
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.entype_embedding = nn.Embedding(self.nen, self.den)

        # self.conv = nn.Conv2d(1, self.dc, (self.k, self.d+self.den), bias=True)  # renewed
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d + self.den + 1), padding=(self.p, 0), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.dc, self.dc)
        self.fc1 = nn.Linear(self.dc, self.nr)
        self.softmax = nn.Softmax()

    def forward(self, x, dist1, dist2, en_type_vec, dp_type_vec, is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        entype_embed = self.entype_embedding(en_type_vec)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed, entype_embed,
                              dp_type_vec.unsqueeze(2)), 2)  # (bz, n, dw + 2*dp + 1)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)
        # x_concat = x_embed.unsqueeze(1)

        features = self.tanh(self.conv(x_concat).squeeze())
        features = F.max_pool1d(features, features.size(2)).squeeze()
        features = self.dropout(features)
        features = features.view(-1, self.dc)
        # features = features.view(-1, self.dc)
        inter = self.fc(features)
        logits = self.fc1(inter)
        return logits


class ParallelCNN(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(ParallelCNN, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        # shared embeddings and weights
        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)

        # different weights
        self.conv1 = nn.Conv2d(1, self.dc, (self.k, self.d), bias=True)  # renewed
        self.conv2 = nn.Conv2d(1, self.dc, (self.k, self.d), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        # self.max_pool = nn.MaxPool1d((1, self.dc), (1, self.dc))
        self.fc1 = nn.Linear(self.dc * 2, self.nr)
        self.fc2 = nn.Linear(self.dc, self.nr)
        # self.softmax = nn.Softmax()

    def forward(self, x, x_no, dist1, dist2, is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        x_no_embed = self.embedding(x_no)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2)  # (bz, n, dw + 2*dp)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)
        x_no_concat = torch.cat((x_no_embed, dist1_embed, dist2_embed), 2)
        x_no_concat = x_no_concat.unsqueeze(1)

        # encoding features without entities
        features_no = self.tanh(self.conv2(x_no_concat).squeeze())
        features_no = F.max_pool1d(features_no, features_no.size(2))
        features_no = self.dropout(features_no)
        features_no = features_no.view(-1, self.dc)
        logits_no = self.fc2(features_no)

        # features for final task
        features = self.tanh(self.conv1(x_concat).squeeze())
        features = F.max_pool1d(features, features.size(2))
        features = features.view(-1, self.dc)
        features = self.dropout(torch.cat((features, features_no), 1))
        logits = self.fc1(features)

        return logits, logits_no


class NoDistancePCNN(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(NoDistancePCNN, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        # shared embeddings and weights
        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)

        # different weights
        self.conv1 = nn.Conv2d(1, self.dc, (self.k, self.d), bias=True)  # renewed
        self.conv2 = nn.Conv2d(1, self.dc, (self.k, self.dw), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        # self.max_pool = nn.MaxPool1d((1, self.dc), (1, self.dc))
        self.fc1 = nn.Linear(self.dc * 2, self.nr)
        self.fc2 = nn.Linear(self.dc, self.nr)
        # self.softmax = nn.Softmax()

    def forward(self, x, x_no, dist1, dist2, is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        x_no_embed = self.embedding(x_no)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2)  # (bz, n, dw + 2*dp)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)
        # x_no_concat = torch.cat((x_no_embed, dist1_embed, dist2_embed), 2)
        x_no_concat = x_no_embed.unsqueeze(1)

        # encoding features without entities
        features_no = self.tanh(self.conv2(x_no_concat).squeeze())
        features_no = F.max_pool1d(features_no, features_no.size(2))
        features_no = self.dropout(features_no)
        features_no = features_no.view(-1, self.dc)
        logits_no = self.fc2(features_no)

        features = self.tanh(self.conv1(x_concat).squeeze())
        features = F.max_pool1d(features, features.size(2))
        features = features.view(-1, self.dc)
        features = self.dropout(torch.cat((features, features_no), 1))
        logits = self.fc1(features)

        return logits, logits_no


class CosDistancePCNN(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(CosDistancePCNN, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        # shared embeddings and weights
        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)

        # different weights
        self.conv1 = nn.Conv2d(1, self.dc, (self.k, self.d), bias=True)  # renewed
        self.conv2 = nn.Conv2d(1, self.dc, (self.k, self.dw), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        # self.max_pool = nn.MaxPool1d((1, self.dc), (1, self.dc))
        self.fc1 = nn.Linear(self.dc * 2, self.nr)
        self.fc2 = nn.Linear(self.dc, self.nr)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.softmax = nn.Softmax()

    def forward(self, x, x_no, dist1, dist2, is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        x_no_embed = self.embedding(x_no)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2)  # (bz, n, dw + 2*dp)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)
        # x_no_concat = torch.cat((x_no_embed, dist1_embed, dist2_embed), 2)
        x_no_concat = x_no_embed.unsqueeze(1)

        # encoding features without entities
        features_no = self.tanh(self.conv2(x_no_concat).squeeze())
        features_no = F.max_pool1d(features_no, features_no.size(2))
        features_no = self.dropout(features_no)
        features_no = features_no.view(-1, self.dc)
        logits_no = self.fc2(features_no)

        features = self.tanh(self.conv1(x_concat).squeeze())
        features = F.max_pool1d(features, features.size(2))
        features = features.view(-1, self.dc)
        cos_loss = torch.abs(self.cos(self.dropout(features), features_no)).sum()
        features = self.dropout(torch.cat((features, features_no), 1))
        logits = self.fc1(features)

        return logits, logits_no, cos_loss


class DSNPCNN(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, entype_embd_size, entype_embd_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(DSNPCNN, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.den = entype_embd_size
        self.nen = entype_embd_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        # shared embeddings and weights
        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.entype_embedding = nn.Embedding(self.nen, self.den)

        # different weights
        self.conv1 = nn.Conv2d(1, self.dc * 2, (self.k, self.d), bias=True)  # renewed
        self.conv2 = nn.Conv2d(1, self.dc, (self.k, self.dw + self.den), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.cos = F.kl_div()

        # self.max_pool = nn.MaxPool1d((1, self.dc), (1, self.dc))
        self.fc1 = nn.Linear(self.dc * 2, self.nr)
        self.fc2 = nn.Linear(self.dc, self.nr)
        self.fc3 = nn.Linear(self.dc, self.nr)
        # self.softmax = nn.Softmax()

    def forward(self, x, x_no, dist1, dist2, en_type_vec, is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        x_no_embed = self.embedding(x_no)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        entype_embed = self.entype_embedding(en_type_vec)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2)  # (bz, n, dw + 2*dp)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)
        x_no_concat = torch.cat((x_no_embed, entype_embed), 2)
        x_no_concat = x_no_concat.unsqueeze(1)

        # encoding features without entities
        features_no = self.tanh(self.conv2(x_no_concat).squeeze())
        features_no = F.max_pool1d(features_no, features_no.size(2))
        features_no = self.dropout(features_no)
        features_no = features_no.view(-1, self.dc)
        logits_no = self.fc2(features_no)

        features = self.tanh(self.conv1(x_concat).squeeze())
        features = F.max_pool1d(features, features.size(2))
        features = self.dropout(features)
        features = features.view(-1, self.dc * 2)

        special_features, share_features = torch.split(features, self.dc, 1)
        sim_loss = torch.abs(self.cos(share_features, features_no)).sum()
        dif_loss = torch.abs(self.cos(special_features, features_no)).sum()
        # sim_loss = torch.abs(F.kl_div(share_features, features_no,True)).sum()
        # dif_loss = torch.abs(F.kl_div(special_features, features_no, True)).sum()
        logits = self.fc1(features)
        special_features = self.fc3(special_features)

        return logits, logits_no, special_features, sim_loss, dif_loss


class DSNPCNN_NP(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(DSNPCNN_NP, self).__init__()
        self.dw = embedding.shape[1]  # dismension of word embeddings
        self.vac_len = embedding.shape[0]  # vocabulary length
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.n = max_len  # max sequence length
        self.kd = self.d * self.k  # filter:[d*k] d:dimension of word embeddings, k:slide_window

        # shared embeddings and weights
        self.embedding = nn.Embedding(self.vac_len, self.dw)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)

        # different weights
        self.conv1 = nn.Conv2d(1, self.dc * 2, (self.k, self.d), bias=True)  # renewed
        self.conv2 = nn.Conv2d(1, self.dc, (self.k, self.dw), bias=True)
        self.dropout = nn.Dropout(self.keep_prob)
        self.tanh = nn.Tanh()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # self.max_pool = nn.MaxPool1d((1, self.dc), (1, self.dc))
        self.fc1 = nn.Linear(self.dc * 2, self.nr)
        self.fc2 = nn.Linear(self.dc, self.nr)
        # self.softmax = nn.Softmax()

    def forward(self, x, x_no, dist1, dist2, is_training=True):
        x_embed = self.embedding(x)  # (bz, n, dw)
        x_no_embed = self.embedding(x_no)
        dist1_embed = self.dist1_embedding(dist1)  # (bz, n, dp)
        dist2_embed = self.dist2_embedding(dist2)  # (bz, n, dp)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2)  # (bz, n, dw + 2*dp)
        x_concat = x_concat.unsqueeze(1)  # (bz,1,n,d)
        # x_no_concat = torch.cat((x_no_embed, dist1_embed, dist2_embed), 2)
        x_no_concat = x_no_embed.unsqueeze(1)

        # encoding features without entities
        features_no = self.tanh(self.conv2(x_no_concat).squeeze())
        features_no = F.max_pool1d(features_no, features_no.size(2))
        features_no = features_no.view(-1, self.dc)

        features = self.tanh(self.conv1(x_concat).squeeze())
        features = F.max_pool1d(features, features.size(2))
        features = features.view(-1, self.dc * 2)

        special_features, share_features = torch.split(features, self.dc, 1)
        sim_loss = torch.abs(self.cos(share_features, features_no)).sum()
        dif_loss = torch.abs(self.cos(special_features, features_no)).sum()
        # cos_loss = torch.abs(self.cos(self.dropout(features), features_no)).sum()
        # logits = torch.cat((special_features, share_features),1)
        # features = self.dropout(torch.cat((features, features_no), 1))
        features_no = self.dropout(features_no)
        features = self.dropout(features)

        logits_no = self.fc2(features_no)
        logits = self.fc1(features)

        return logits, logits_no, special_features, sim_loss, dif_loss
