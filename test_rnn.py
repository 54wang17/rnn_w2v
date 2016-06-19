import numpy as np
import theano as theano
import theano.tensor as T
import os
import cPickle

# BUILD MODEL
_WORD_EMBEDDING_SIZE = int(os.environ.get('Word_2_Vec_DIM', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '128'))
_CLASSIFIER_NUM = int(os.environ.get('CLASSIFIER_NUM', '3'))
# _LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
# _NEPOCH = int(os.environ.get('NEPOCH', '100'))
# _MODEL_FILE = os.environ.get('MODEL_FILE')

in_dim = _WORD_EMBEDDING_SIZE
hidden_dim = _HIDDEN_DIM
out_dim = _CLASSIFIER_NUM

x = T.matrix('x')
y = T.matrix('y')


U_init = np.random.uniform(-np.sqrt(1./in_dim), np.sqrt(1./in_dim), (hidden_dim, in_dim))
W_init = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
V_init = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (out_dim, hidden_dim))

U = theano.shared(name='U', value=U_init.astype(theano.config.floatX))
W = theano.shared(name='W', value=W_init.astype(theano.config.floatX))
V = theano.shared(name='V', value=V_init.astype(theano.config.floatX))


def forward_prop_step(x_t, s_tm1, U, W):
    s_t = T.tanh(T.dot(U, x_t) + T.dot(W, s_tm1))
    return s_t

s, updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[dict(initial=T.zeros(hidden_dim))],
            non_sequences=[U, W],
            truncate_gradient=4,
            mode='DebugMode')
p_y = T.nnet.softmax(T.dot(V, s[-1]))
o_error = T.nnet.categorical_crossentropy(p_y, y)
test1 = theano.function(inputs=[x, y], outputs=o_error, updates=updates)



# LOAD TEST DATA
data = cPickle.load(open("./data/corpus.p","rb"))
W2V = cPickle.load(open("./data/word2vec.p","rb"))
W2V = np.array(W2V[0]).astype(theano.config.floatX)
train_X, valid_X = data[0], data[1]
train_Y, valid_Y = data[8], data[9]
train_x = np.matrix(W2V[train_X[0]])
# Transform label number to one-hot vector and then to a 2-D matrix
train_y = np.zeros((1, out_dim))    # np.matrix([0, 0, 1], dtype='int32')
train_y[0][train_Y[0][0]-1] = 1.0
assert train_x.shape[1] == _WORD_EMBEDDING_SIZE
assert train_y.shape[1] == _CLASSIFIER_NUM
print test1(train_x, train_y)

