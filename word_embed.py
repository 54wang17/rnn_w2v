import numpy as np
import cPickle
from gensim.models import Word2Vec
    
if __name__=="__main__":
    w2v_file = './data/GoogleNews-vectors-negative300.bin'
    model = Word2Vec.load_word2vec_format(w2v_file, binary=True)
    embd_dim = 300
    y = cPickle.load(open("./data/corpus.p","rb"))
    word2idx, idx2word = y[6], y[7]
    assert len(idx2word) == len(word2idx)
    vocab_size = len(idx2word)
    W = np.zeros(shape=(vocab_size, embd_dim))
    for word in word2idx:
        if word in model.vocab:
            W[word2idx[word]] = model[word]
        else:
            # print word,'|\t',
            W[word2idx[word]] = np.random.uniform(-0.25,0.25,embd_dim)
    cPickle.dump([W], open("word2vec.p", "wb"))
    print "pretrained word vector created!"
