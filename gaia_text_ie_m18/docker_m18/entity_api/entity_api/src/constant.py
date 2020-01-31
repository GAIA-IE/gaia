ELMO_MAX_CHAR_LEN = 50

# Constant strings
PAD = '<$PAD$>'
UNK = '<$UNK$>'
EOS = '<$EOS$>'
SOS = '<$SOS$>'
PAD_INDEX = 0       # Padding symbol index
UNK_INDEX = 1       # Unknown token index
SOS_INDEX = 2       # Start-of-sentence symbol index
EOS_INDEX = 3       # End-of-sentence symbol index
CHAR_PADS = [
    (PAD, PAD_INDEX),
    (UNK, UNK_INDEX),
]
TOKEN_PADS = [
    (PAD, PAD_INDEX),
    (UNK, UNK_INDEX),
    (SOS, SOS_INDEX),
    (EOS, EOS_INDEX),
]
TOK_REPLACEMENT = {
    '-LRB-': '(',
    '-RRB-': ')',
    '-LSB-': '[',
    '-RSB-': ']',
    '-LCB-': '{',
    '-RCB-': '}',
    '``': '"',
    '\'\'': '"',
    '/.': '.',
    '/?': '?'
}
LABEL_PADS = [
    (PAD, PAD_INDEX)
]
ELMO_PRETRAINED_DIR = '/data/m1/liny9/elmo/pretrained'
ELMO_MODELS = {
    'eng_small': {
        'weight': '{}/eng.small.hdf5'.format(ELMO_PRETRAINED_DIR),
        'option': '{}/eng.small.json'.format(ELMO_PRETRAINED_DIR)
    },
    'eng_medium': {
        'weight': '{}/eng.medium.hdf5'.format(ELMO_PRETRAINED_DIR),
        'option': '{}/eng.medium.json'.format(ELMO_PRETRAINED_DIR)
    },
    'eng_original': {
        'weight': '{}/eng.original.hdf5'.format(ELMO_PRETRAINED_DIR),
        'option': '{}/eng.original.json'.format(ELMO_PRETRAINED_DIR)
    },
    'eng_original_5.5b': {
        'weight': '{}/eng.original.5.5b.hdf5'.format(ELMO_PRETRAINED_DIR),
        'option': '{}/eng.original.5.5b.json'.format(ELMO_PRETRAINED_DIR)
    }
}