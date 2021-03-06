from gru_rnn import GruRNN
import os
import cPickle
import numpy as np
import theano
from utils import idx2onehot
from datetime import datetime

_WORD_EMBEDDING_SIZE = int(os.environ.get('Word_2_Vec_DIM', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '128'))
_CLASSIFIER_NUM = int(os.environ.get('CLASSIFIER_NUM', '3'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '50'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

in_dim = _WORD_EMBEDDING_SIZE
hidden_dim = _HIDDEN_DIM
out_dim = _CLASSIFIER_NUM


data = cPickle.load(open("./data/corpus.p", "rb"))
W = cPickle.load(open("./data/word2vec.p", "rb"))
W2V = np.array(W[0]).astype(theano.config.floatX)
train_X, valid_X = data[0], data[2]
train_Y, valid_Y = data[1], data[3]
train_x = [np.matrix(W2V[sen_idx]) for sen_idx in train_X]
train_y = [idx2onehot(label, out_dim) for l in train_Y for label in l]
valid_x = [np.matrix(W2V[sen_idx]) for sen_idx in valid_X]
valid_y = [idx2onehot(label, out_dim) for l in valid_Y for label in l]


# Initialize and build model
model = GruRNN(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)

# Begin training model with mini batch
model.train_with_mini_batch(train_x, train_y, valid_x, valid_y, learning_rate=_LEARNING_RATE, nepoch=_NEPOCH)

# If do not want mini batch, use train with sdg instead
model.train_with_sdg(train_x, train_y, valid_x, valid_y, learning_rate=_LEARNING_RATE, nepoch=_NEPOCH)