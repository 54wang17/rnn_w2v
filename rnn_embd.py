from rnn_classifier import RNN
import os
import cPickle
import numpy as np
import theano
from datetime import datetime

_WORD_EMBEDDING_SIZE = int(os.environ.get('Word_2_Vec_DIM', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '128'))
_CLASSIFIER_NUM = int(os.environ.get('CLASSIFIER_NUM', '3'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')



data = cPickle.load(open("./data/corpus.p", "rb"))
W = cPickle.load(open("./data/word2vec.p", "rb"))
W = np.array(W[0]).astype(theano.config.floatX)
train_X, valid_X = data[0], data[1]
train_Y, valid_Y = data[8], data[9]
train_x = np.matrix(W[train_X[0]])

train_y = np.matrix([0, 0, 1], dtype='int32')


model = RNN(in_dim=_WORD_EMBEDDING_SIZE, out_dim=_CLASSIFIER_NUM, hidden_dim=_HIDDEN_DIM, bptt_truncate=4)
t1 = datetime.now()
model.sgd_step(train_x, train_y, _LEARNING_RATE)
t2 = datetime.now()

print "SGD Step time: %i milliseconds" % ((t2 - t1).microseconds / 1000)




