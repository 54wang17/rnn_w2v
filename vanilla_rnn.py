""" Vanilla RNN"""
from basic_rnn import BasicRNN
import numpy as np
import theano
import theano.tensor as T
import logging
from datetime import datetime
import cPickle
from utils import idx2onehot
logger = logging.getLogger(__name__)
mode = theano.Mode(linker='cvm')


class VanillaRNN(BasicRNN):

    def __init__(self, n_in, n_out, n_hidden, activation='tanh',
                 l1_reg=0.00, l2_reg=0.00):
        BasicRNN.__init__(self, n_in, n_out, n_hidden, activation)
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        by_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh')
        self.by = theano.shared(value=by_init, name='by')
        self.params = [self.U, self.W, self.V, self.bh, self.by]

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.velocity_updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)
            self.velocity_updates[param] = theano.shared(init)

        self.L1_reg = float(l1_reg)
        self.L2_reg = float(l2_reg)
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.U.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += T.sum(self.W ** 2)
        self.L2_sqr += T.sum(self.U ** 2)

    def build_model(self):
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        logger.info('... building the model')

        U, W, V, bh, by = self.U, self.W, self.V, self.bh, self.by
        x = T.matrix('x')
        y = T.matrix('y')

        def forward_prop_step(x_t, s_tm1, U, W, bh):
            s_t = self.activation(T.dot(U, x_t) + T.dot(W, s_tm1) + bh)
            return s_t

        s, _ = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, W, bh],
            mode='DebugMode')

        p_y = T.nnet.softmax(T.dot(self.V, s[-1]) + by)
        prediction = T.argmax(p_y, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(p_y, y))
        self.cost = o_error + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        # Assign functions
        self.forward_propagation = theano.function([x], s[-1])
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)

        l_r = T.scalar('l_r', dtype=theano.config.floatX)   # learning rate (may change)
        mom = T.scalar('mom', dtype=theano.config.floatX)   # momentum
        self.bptt, self.f_update = self.Momentum(x, y, l_r, mom)

    def Momentum(self, x, y, l_r, mom):
        gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)
        bptt = theano.function([x, y], gparams)
        updates = []
        for param, gparam in zip(self.params, gparams):
            v = self.velocity_updates[param]
            v_upd = mom * v - l_r * gparam
            p_upd = param + v_upd
            updates.append((v, v_upd))
            updates.append((param, p_upd))
        momentum = theano.function(inputs=[x, y, l_r, mom],
                                   outputs=self.cost,
                                   updates=updates,
                                   mode=mode)
        return bptt, momentum

    def train_with_Momentum(self, X_train, y_train, X_val=None, y_val=None,
                            n_epochs=50, validation_frequency=100,
                            learning_rate=0.01, learning_rate_decay=1,
                            final_momentum=0.9, initial_momentum=0.5, momentum_switchover=5):
        """ Train model

        Pass in X_val, y_val to compute test error and report during training.
        X_train : ndarray (n_seq x n_steps x n_in)
        y_train : ndarray (n_seq x 1 x n_out)
        validation_frequency : int
            in terms of number of sequences (or number of weight updates)
        """
        isValidation = False
        if X_val is not None:
            assert(y_val is not None)
            isValidation = True

        n_train = len(y_train)

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        for epoch in xrange(n_epochs):
            for idx in xrange(n_train):
                if epoch > momentum_switchover:
                    effective_momentum = final_momentum
                else:
                    effective_momentum = initial_momentum
                example_cost = self.f_update(X_train[idx],y_train[idx], learning_rate, effective_momentum)

                # iteration number (how many weight updates have we made?)
                # epoch and index are 0-based
                iter = epoch * n_train + idx + 1
                if iter % validation_frequency == 0:
                    # compute loss on training set
                    train_loss = self.calculate_loss(X_train, y_train)
                    if isValidation:
                        validation_loss = self.calculate_loss(X_val, y_val)
                        logger.info('epoch %i, seq %i/%i, tr loss %f te loss %f lr: %f' %
                                    (epoch + 1, idx + 1, n_train, train_loss, validation_loss, learning_rate))
                    else:
                        logger.info('epoch %i, seq %i/%i, train loss %f lr: %f' %
                                    (epoch + 1, idx + 1, n_train, train_loss, learning_rate))

            learning_rate *= learning_rate_decay


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    in_dim, hidden_dim, out_dim = 300, 128, 3
    data = cPickle.load(open("./data/corpus.p", "rb"))
    W = cPickle.load(open("./data/word2vec.p", "rb"))
    W2V = np.array(W[0]).astype(theano.config.floatX)

    train_X, train_Y = data[0], data[1]
    train_x = [np.matrix(W2V[sen_idx]) for sen_idx in train_X]
    train_y = [idx2onehot(label, out_dim) for l in train_Y for label in l]

    model = VanillaRNN(n_in=in_dim, n_out=out_dim, n_hidden=hidden_dim)
    model.build_model()
    t0 = datetime.now()
    model.f_update(train_x[0], train_y[0], 0.01, 0.5)
    print model.by.get_value().shape
    print "Elapsed time: %f" % ((datetime.now() - t0).microseconds / 1000.0)


