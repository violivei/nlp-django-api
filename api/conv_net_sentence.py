#!/usr/bin/env python

"""
Training a convolutional network for sentence classification,
as described in paper:
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
"""
import cPickle as pickle
import numpy as np
import theano
import sys
import argparse
import warnings
import os

warnings.filterwarnings("ignore")
#sys.path.append("..")

BASE = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.join(BASE, '.'))

from conv_net_classes import *
from process_data import process_data


def get_idx_from_sent(sent, word_index, max_l, pad):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    Drop words non in word_index. Attardi.
    :param max_l: max sentence length
    :param pad: pad length
    """
    x = [0] * pad                # left padding
    words = sent.split()[:max_l] # truncate words from test set
    for word in words:
        if word in word_index: # FIXME: skips unknown words
            x.append(word_index[word])
    while len(x) < max_l + 2 * pad: # right padding
        x.append(0)
    return x


def make_idx_data_cv(revs, word_index, cv, max_l, pad):
    """
    Transforms sentences into a 2-d matrix and splits them into
    train and test according to cv.
    :param cv: cross-validation step
    :param max_l: max sentence length
    :param pad: pad length
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_index, max_l, pad)
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train, dtype="int32")
    test = np.array(test, dtype="int32")
    return train, test
  

def read_corpus(filename, word_index, max_l, pad=2, clean_string=False,
                textField=3):
    test = []
    with open(filename) as f:
        for line in f:
            fields = line.strip().split("\t")
            text = fields[textField]
            if clean_string:
                text_clean = clean_str(text)
            else:
                text_clean = text.lower()
            sent = get_idx_from_sent(text_clean, word_index, max_l, pad)
            #sent.append(0)      # unknown y
            test.append(sent)
    return np.array(test, dtype="int32")

def get_predict(model, text, clean_string=False):
    # test
    with open(os.path.join(BASE,"models/",model)) as mfile:
        cnn = ConvNet.load(mfile)
        word_index, labels, max_l, pad = pickle.load(mfile)

    cnn.activations = [Iden] #TODO: save it in the model

    test = []
    if clean_string:
        text_clean = clean_str(text)
    else:
        text_clean = text.lower()
    sent = get_idx_from_sent(text_clean, word_index, max_l, pad)
    test.append(sent)
    test_set_x = np.array(test, dtype="int32")
    #test_set_x = read_corpus(args.input, word_index, max_l, pad, textField=textField)

    test_set_y_pred = cnn.predict(test_set_x)
    test_model = theano.function([cnn.x], test_set_y_pred, allow_input_downcast=True)
    results = test_model(test_set_x)
    print(text)
    print(results)
    print(labels)
    print(labels[results[0]])
    return labels[results[0]]

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CNN sentence classifier.")
    
    parser.add_argument('model', type=str, default='mr',
                        help='model file (default %(default)s)')
    parser.add_argument('input', type=str,
                        help='train/test file in SemEval twitter format')
    parser.add_argument('-train', help='train model',
                        action='store_true')
    parser.add_argument('-static', help='static or nonstatic',
                        action='store_true')
    parser.add_argument('-clean', help='tokenize text',
                        action='store_true')
    parser.add_argument('-filters', type=str, default='3,4,5',
                        help='n[,n]* (default %(default)s)')
    parser.add_argument('-vectors', type=str,
                        help='word2vec embeddings file (random values if missing)')
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='dropout probability (default %(default)s)')
    parser.add_argument('-epochs', type=int, default=25,
                        help='training iterations (default %(default)s)')
    parser.add_argument('-tagField', type=int, default=1,
                        help='label field in files (default %(default)s)')
    parser.add_argument('-textField', type=int, default=0,
                        help='text field in files (default %(default)s)')

    args = parser.parse_args()

    if not args.train:
        # test
        with open(args.model) as mfile:
            cnn = ConvNet.load(mfile)
            word_index, labels, max_l, pad = pickle.load(mfile)

        cnn.activations = [Iden] #TODO: save it in the model

        tagField = args.tagField
        textField = args.textField
        test_set_x = read_corpus(args.input, word_index, max_l, pad, textField=textField)
        test_set_y_pred = cnn.predict(test_set_x)
        test_model = theano.function([cnn.x], test_set_y_pred, allow_input_downcast=True)
        results = test_model(test_set_x)
        # invert indices (from process_data.py)
        for line, y in zip(open(args.input), results):
            tokens = line.split("\t")
            tokens[tagField] = labels[y]
            print "\t".join(tokens),
        sys.exit()

    # training
    sents, U, word_index, vocab, labels = process_data(args.input, args.clean,
                                                       args.vectors,
                                                       args.tagField, args.textField)

    # sents is a list of entries, where each entry is a dict:
    # {"y": 0/1, "text": , "num_words": , "split": cv fold}
    # vocab: dict of word doc freq
    filter_hs = [int(x) for x in args.filters.split(',')]
    model = args.model
    if args.static:
        print "model architecture: CNN-static"
        non_static = False
    else:
        print "model architecture: CNN-non-static"
        non_static = True
    if args.vectors:
        print "using: word2vec vectors"
    else:
        print "using: random vectors"

    # filter_h determines padding, hence it depends on largest filter size.
    pad = max(filter_hs) - 1
    max_l = 56 # DEBUG: max(x["num_words"] for x in sents)
    height = max_l + 2 * pad # padding on both sides

    classes = set(x["y"] for x in sents)
    width = U.shape[1]
    conv_non_linear = "relu"
    hidden_units = [100, len(classes)]
    batch_size = 50
    dropout_rate = args.dropout
    sqr_norm_lim = 9
    shuffle_batch = True
    lr_decay = 0.95
    layer_sizes = [hidden_units[0]*len(filter_hs)] + hidden_units[1:]
    parameters = (("image shape", height, width),
                  ("filters", args.filters),
                  ("hidden units", hidden_units),
                  ("layer sizes", layer_sizes),
                  ("dropout rate", dropout_rate),
                  ("batch size", batch_size),
                  ("adadelta decay", lr_decay),
                  ("conv_non_linear", conv_non_linear),
                  ("non static", non_static),
                  ("sqr_norm_lim", sqr_norm_lim),
                  ("shuffle batch", shuffle_batch))
    for param in parameters:
        print "%s: %s" % (param[0], ",".join(str(x) for x in param[1:]))
    results = []
    # ensure replicability
    np.random.seed(3435)
    # perform 10-fold cross-validation
    for i in range(0, 10):
        # test = [padded(x) for x in sents if x[split] == i]
        # train is rest
        train_set, test_set = make_idx_data_cv(sents, word_index, i, max_l, pad)
        cnn = ConvNet(U, height, width,
                      filter_hs=filter_hs,
                      conv_non_linear=conv_non_linear,
                      hidden_units=hidden_units,
                      batch_size=batch_size,
                      non_static=non_static,
                      dropout_rates=[dropout_rate])
        perf = cnn.train(train_set, test_set,
                         lr_decay=lr_decay,
                         shuffle_batch=shuffle_batch, 
                         epochs=args.epochs,
                         sqr_norm_lim=sqr_norm_lim)
        print "cv: %d, perf: %f" % (i, perf)
        # save model
        if i == 0 or perf > max(results):
            with open(model, "wb") as mfile:
                cnn.save(mfile)
                pickle.dump((word_index, labels, max_l, pad), mfile)
        results.append(perf)  
    print "Avg. accuracy: %.4f" % np.mean(results)
