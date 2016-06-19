import os
from collections import defaultdict
import random
import cPickle


def load_data(file_path='./data', train_file='train.txt', test_file='test.txt', label_file='train_label.txt'):
    train = []
    test = []
    label = []
    with open(os.path.join(file_path,train_file), 'rb') as f:
        for line in f:
            text = line.strip().lower()
            train.append(text)
    with open(os.path.join(file_path, test_file), 'rb') as f:
        for line in f:
            text = line.strip().lower()
            test.append(text)
    with open(os.path.join(file_path, label_file), 'rb') as f:
        for line in f:
            text = line.strip().lower()
            label.append(text)

    return train, test, label


def build_vocab(corpus, min_df=10):
    '''
    :param corpus:
    :param min_df:
    :return: index2word, word2index
    '''
    vocab = defaultdict(int)

    for sent in corpus:
        # words = set(sent.split())
        for word in sent.split():
            vocab[word] += 1

    index2word, word2index = {}, {}

    index2word[0] = '<UNKNOWN>'
    word2index['<UNKNOWN>'] = 0
    ix = 1
    for w in vocab:
        if vocab[w] < min_df:
            continue
        word2index[w] = ix
        index2word[ix] = w
        ix += 1
    del vocab

    return index2word, word2index


def transform_data(sentences, word2index):
    '''
    :param sentences: a list of sentence, each sentence is a string
    :param word2index: a map of word with its corresponding index
    :return: a 2D list, each row represents a sentence, each elem in row is an index representing a word
    '''
    data = []
    for rev in sentences:
        sent = get_idx_from_sent(rev, word2index)
        data.append(sent)

    return data


def get_idx_from_sent(sentence, word2index):
    '''
    :param sentence: a string of sentence
    :param word2index: map of word and index
    :return: a list of index
    '''
    words = sentence.split()
    index = [word2index[w] if w in word2index else word2index['<UNKNOWN>'] for w in words]

    return index


def label2category(label):

    index2label, label2index = build_vocab(label, min_df=0)
    del label2index['<UNKNOWN>'], index2label[0]
    for index in index2label:
        print 'Label {} -> Category id {}'.format(index2label[index], index)

    return label2index


def create_val_test(text_index, text, label_data):
    assert len(text_index) == len(text)
    n_data = len(text)
    print "text length: {}".format(n_data)

    n_valid = int(0.1 * n_data)
    valid_indices = random.sample(xrange(n_data), n_valid)
    train_text_index = [text_index[i] for i in xrange(n_data) if i not in valid_indices]
    valid_text_index = [text_index[i] for i in xrange(n_data) if i in valid_indices]
    train_text = [text[i] for i in xrange(n_data) if i not in valid_indices]
    valid_text = [text[i] for i in xrange(n_data) if i in valid_indices]
    train_label = [label_data[i] for i in xrange(n_data) if i not in valid_indices]
    valid_label = [label_data[i] for i in xrange(n_data) if i in valid_indices]

    print "train text length: {}, train label length: {}".format(len(train_text), len(train_label))
    print "val text length: {}, val label length: {}".format(len(valid_text), len(valid_label))

    return train_text_index, valid_text_index, train_text, valid_text, train_label, valid_label

if __name__=="__main__":

    # """
    train, test, label  = load_data()
    index2word, word2index = build_vocab(train+test)
    train_data = transform_data(train, word2index)
    test_data = transform_data(test, word2index)
    label2index = label2category(label)
    label_data = transform_data(label, label2index)

    train_idx, val_idx, train_text, val_text, train_labels, val_labels = create_val_test(train_data, train, label_data)

    print 'EXAMPLE: {} \tLABEL: {}'.format(val_text[0],val_labels[0])

    cPickle.dump([train_idx, val_idx, test_data, train_text, val_text, test, word2index, index2word,
                  train_labels, val_labels], open("./data/corpus.p", "wb"))
    # cPickle.dump([train_idx, val_idx, train_data, train_text, val_text, train, wordtoix, ixtoword, train_labels, val_labels], open("chatcorpus_end.p", "wb"))
    print "Dataset created!"
