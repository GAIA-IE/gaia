import json

import torch
from src.data import BioDataset
from src.model import LstmCnnOutElmo, LstmCnn, AttnFet

def load_lstm_cnn_elmo_model(model_file, elmo_option, elmo_weight,
                             gpu=False, device=0):
    print('Loading model from {}'.format(model_file))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'

    state = torch.load(model_file, map_location=map_location)
    params = state['params']
    model = LstmCnnOutElmo(
        vocabs=state['vocabs'],
        elmo_option=elmo_option,
        elmo_weight=elmo_weight,
        lstm_hidden_size=params['lstm_size'],
        parameters=state['model_params'],
        output_bias=not params['no_out_bias']
    )
    model.load_state_dict(state['model'])
    if gpu:
        torch.cuda.set_device(device)
        model.cuda(device=device)

    return model, state['vocabs']


def load_lstm_cnn_model(model_file, gpu=False, device=0):
    print('Loading model from {}'.format(model_file))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'

    state = torch.load(model_file, map_location=map_location)
    params = state['params']
    char_filters = json.loads(params['char_filters'])
    char_feat_dim = sum([i[1] for i in char_filters])
    model = LstmCnn(
        vocabs=state['vocabs'],
        char_embed_dim=params['char_dim'],
        char_filters=char_filters,
        char_feat_dim=char_feat_dim,
        lstm_hidden_size=params['lstm_size'],
        parameters=state['model_params'],
        char_out=params['char_out'],
        char_activation=params['char_act'],
        char_out_activation=params['char_out_act'],
        use_char=not params['no_char'],
        output_bias=not params['no_out_bias']
    )
    model.load_state_dict(state['model'])
    if gpu:
        torch.cuda.set_device(device)
        model.cuda(device=device)

    return model, state['vocabs']

def load_attn_fet_model(model_file, gpu=False, device=0):
    print('Loading model from: {}'.format(model_file))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'

    state = torch.load(model_file, map_location=map_location)
    train_args = state['args']
    model = AttnFet(state['vocab'],
                    word_embed_dim=train_args['embed_dim'],
                    lstm_size=train_args['lstm_size'],
                    full_attn=train_args['full_attn'],
                    latent=train_args['latent'],
                    flat_attn=train_args['flat_attn'],
                    latent_size=train_args['latent_size'],
                    dist=train_args['dist'],
                    svd=None,
                    embed_num=state['embed_num']
                    )
    model.load_state_dict(state['model'])
    if gpu:
        torch.cuda.set_device(device)
        model.cuda(device=device)

    return model, state['vocab']

