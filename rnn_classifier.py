import theano as theano
import theano.tensor as T
import numpy as np
theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'warn'


class RNN:

    def __init__(self, in_dim, out_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = in_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./in_dim), np.sqrt(1./in_dim), (hidden_dim, in_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (out_dim, hidden_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()


    def __theano_build__(self):
        U, W, V = self.U, self.W, self.V
        x = T.matrix('x')
        y = T.matrix('y')

        def forward_prop_step(x_t, s_tm1, U, W):
            s_t = T.tanh(T.dot(U, x_t) + T.dot(W, s_tm1))
            return s_t


        s, updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, W],
            truncate_gradient=self.bptt_truncate,
             mode='DebugMode')

        p_y = T.nnet.softmax(T.dot(V, s[-1]))
        prediction = T.argmax(p_y, axis=1)

        o_error = T.sum(T.nnet.categorical_crossentropy(p_y, y).flatten())

        # Gradients
        dU = T.grad(o_error, U)
        dW = T.grad(o_error, W)
        dV = T.grad(o_error, V)

        # Assign functions
        self.forward_propagation = theano.function([x], s[-1])
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x, y, learning_rate], [],
                      updates=[(self.U, self.U - learning_rate * dU),
                               (self.V, self.V - learning_rate * dV),
                               (self.W, self.W - learning_rate * dW)])



