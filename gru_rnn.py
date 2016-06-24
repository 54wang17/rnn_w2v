from basic_rnn import BasicRNN
import numpy as np
import theano as theano
import theano.tensor as T


class GruRNN(BasicRNN):

    def __init__(self, in_dim, hidden_dim, out_dim, bptt_truncate=-1):
        # Assign instance variables
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./in_dim), np.sqrt(1./in_dim), (hidden_dim, in_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (out_dim, hidden_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(out_dim)
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        V, U, W, b, c = self.V, self.U, self.W, self.b, self.c

        x = T.matrix('x')
        y = T.vector('y')

        def forward_prop_step(x_t, s_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # GRU Layer
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_t) + W[0].dot(s_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_t) + W[1].dot(s_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_t) + W[2].dot(s_t_prev * r_t) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev

            return s_t

        s, _ = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))])

        # Final output calculation
        # Theano's softmax returns a matrix with one row, we only need the row
        p_y = T.nnet.softmax(V.dot(s[-1]) + c)[0]  # [0]
        prediction = T.argmax(p_y, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(p_y, y))

        # Total cost (could add regularization here)
        cost = o_error

        # Gradients
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # Assign functions
        self.predict = theano.function([x], p_y)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dU, dW, db, dV, dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [],
            updates=[
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

    def calculate_loss(self, X, Y):
        return np.mean([self.ce_error(x,y) for x,y in zip(X,Y)])
