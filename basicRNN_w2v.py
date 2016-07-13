from basic_rnn import BasicRNN
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

batch_size = 10
x_batch = train_x[10:10+batch_size]
y_batch = train_y[10:10+batch_size]
print y_batch
# lengths = [x.shape[0] for x in x_batch]
# max_len = max(lengths)
# print max_len
# print train_x[10]
# x_mask = np.zeros((max_len, batch_size, 300)).astype(theano.config.floatX)
# x = np.zeros((max_len, batch_size, 300)).astype(theano.config.floatX)
# for idx, s in enumerate(x_batch):
#     x[:lengths[idx], idx] = s
#     x_mask[:lengths[idx], idx] = 1.
# print x[:,0]
# print x_mask[:,0]
'''
# Initialize and build model
model = BasicRNN(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, bptt_truncate=4)
model.build_model()

# Test SDG
t1 = datetime.now()
model.f_update(train_x[10], train_y[10], _LEARNING_RATE)
t2 = datetime.now()
print "SGD Step time: %i milliseconds" % ((t2 - t1).microseconds / 1000)

# Begin training model
model.train_with_sgd(train_x, train_y, valid_x, valid_y, learning_rate=_LEARNING_RATE, nepoch=_NEPOCH)
'''