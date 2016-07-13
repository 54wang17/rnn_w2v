from basic_rnn import BasicRNN
from gru_rnn import GruRNN
import cPickle
import theano as theano
import numpy as np
import os
from sklearn.metrics import confusion_matrix

_WORD_EMBEDDING_SIZE = int(os.environ.get('Word_2_Vec_DIM', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '128'))
_CLASSIFIER_NUM = int(os.environ.get('CLASSIFIER_NUM', '3'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

in_dim = _WORD_EMBEDDING_SIZE
hidden_dim = _HIDDEN_DIM
out_dim = _CLASSIFIER_NUM

# Load test data
data = cPickle.load(open("./data/corpus.p", "rb"))
W = cPickle.load(open("./data/word2vec.p", "rb"))
W2V = np.array(W[0]).astype(theano.config.floatX)
test_X, test_y = data[4], data[5]
test_X = [np.matrix(W2V[sen_idx]) for sen_idx in test_X]

# Load model and use model to predict label for test data
# rnn = BasicRNN(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, bptt_truncate=4)
rnn = GruRNN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
rnn.load_model_parameters('./data/rnn-w2v-128-300-2016-07-12-23-39-09.pkl')
rnn.build_minibatch(batch_size=50)
predict_y = rnn.get_prediction(test_X, minibatch=True)

# Generate Evaluation Matrix
test_y_ = [y for l in test_y for y in l]
predict_y_ = [y for x in predict_y for y in x.tolist()][:len(test_y)]
eval_matrix = confusion_matrix(test_y_, predict_y_)

# Pretty print the evaluation matrix
labels = ['N/A', 'Upcoming', 'Priced']
print '             Predicted'
for i in xrange(len(eval_matrix)):
    if i == 0:
        print '{:10s}'.format('Actual'), '\t '.join(labels)
    else:
        print '{:10s}'.format(labels[i-1]),
        for j in xrange(len(eval_matrix[i])-1):
            print '{:5d}'.format(eval_matrix[i][j]),
        print