#!/usr/bin/env python

"""
Training a convolutional network for sentence classification,
as described in paper:
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
"""
import sys
import cPickle as pickle
import numpy as np
import theano
import argparse
import warnings
warnings.filterwarnings("ignore")   

# run from everywhere without installing
sys.path.append(".")
from conv_net_classes import *
from process_data import *


def sent2indices(sent, word_index, max_l, pad):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    Drop words non in word_index.
    :param sent: list of words.
    :param word_index: associates an index to each word
    :param max_l: max sentence length
    :param pad: pad size
    """
    x = [0] * pad                # left padding
    for word in sent:
        if word in word_index: # FIXME: skips unknown words
            if len(x) < max_l: # truncate long sent
                x.append(word_index[word])
            else:
                break
    # len(x) includes pad
    rpad = [0] * max(0, max_l + 2 * pad - len(x)) # right padding
    return x + rpad


def read_corpus(filename, word_index, max_l, pad=2, clean_string=False,
                textField=3):
    """
    Load test corpus, in TSV format.
    :param filename: file with sentences.
    :param word_index: word IDs.
    :param max_l: max sentence length.
    :param pad: padding size.
    :param textField: index of field containing text.
    :return: an array, each row consists of sentence word indices
    """
    corpus = []
    with open(filename) as f:
        for line in f:
            fields = line.strip().split("\t")
            text = fields[textField]
            if clean_string:
                text_clean = clean_str(text)
            else:
                text_clean = text.lower()
            # turn sentences into lists of indices
            sent = sent2indices(text_clean.split(), word_index, max_l, pad)
            corpus.append(sent)
    return np.array(corpus, dtype="int32")


def predict(cnn, x):

    y_pred = cnn.output(x)
    # compile expression
    test_function = theano.function([cnn.x], y_pred, allow_input_downcast=True)
    return test_function(x)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CNN sentence classifier.")
    
    parser.add_argument('model', type=str, default='mr',
                        help='model file (default %(default)s)')
    parser.add_argument('input', type=str,
                        help='train/test file in SemEval twitter format')
    parser.add_argument('-train', help='train model',
                        action='store_true')
    parser.add_argument('-filters', type=str, default='3,4,5',
                        help='n[,n]* (default %(default)s)')
    parser.add_argument('-vectors', type=str,
                        help='word2vec embeddings file (random values if missing)')
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='dropout probability (default %(default)s)')
    parser.add_argument('-hidden', type=int, default=100,
                        help='hidden units in feature map (default %(default)s)')
    parser.add_argument('-epochs', type=int, default=25,
                        help='training iterations (default %(default)s)')
    parser.add_argument('-tagField', type=int, default=1,
                        help='label field in files (default %(default)s)')
    parser.add_argument('-textField', type=int, default=2,
                        help='text field in files (default %(default)s)')

    args = parser.parse_args()

    # theano.config
    theano.config.floatX = 'float32'

    if not args.train:
        # predict
        with open(args.model) as mfile:
            cnn = ConvNet.load(mfile)
            word_index, max_l, pad, labels = pickle.load(mfile)
        test_set_x = read_corpus(args.input, word_index, max_l, pad, textField=args.textField)
        results = predict(cnn, test_set_x)
        # convert indices to labels
        for line, y in zip(open(args.input), results):
            tokens = line.split("\t")
            tokens[args.tagField] = labels[y]
            print "\t".join(tokens),
        sys.exit()

    # training
    np.random.seed(345)         # for replicability
    print "loading sentences...",
    # sents is a list of pairs: (list of words, label)
    # word_df: dict of word doc freq
    sents, word_df, labels = load_sentences(args.input,
                                            tagField=args.tagField,
                                            textField=args.textField)
    max_l = max(len(words) for words,l in sents)
    print "done!"
    print "number of sentences: %d" % len(sents)
    print "vocab size: %d" % len(word_df)
    print "max sentence length: %d" % max_l

    if args.vectors:
        print "loading word2vec vectors...",
        vectors, words = load_vectors(args.vectors, args.vectors.endswith('.bin'))
        # get embeddings size:
        k = vectors.shape[1]
        print "done (%d, %d)" % vectors.shape
    else:
        print "using: random vectors"
        vectors = []
        words = []
    print "adding unknown words...",
    add_unknown_words(vectors, words, word_df, k)
    print len(words)
    word_index = {w:i for i,w in enumerate(words)}

    filter_hs = [int(x) for x in args.filters.split(',')]
    model = args.model

    # filter_h determines padding, hence it depends on largest filter size.
    pad = max(filter_hs) - 1
    max_l = max(len(x_y[0]) for x_y in sents)
    height = max_l + 2 * pad    # padding on both sides
    width = vectors.shape[1]    # embeddings size
    feature_maps = args.hidden
    output_units = len(labels)
    conv_activation = "relu"
    activation = Iden #T.tanh
    dropout_rate = args.dropout
    lr = 0.5
    rho = 0.95
    maxnorm = 3.0
    batch_size = 50
    shuffle_batch = True
    parameters = (("image shape", height, width),
                  ("filters", args.filters),
                  ("feature maps", feature_maps),
                  ("output units", output_units),
                  ("dropout rate", dropout_rate),
                  ("conv_activation", conv_activation),
                  ("activation", activation),
                  ("lr", lr),
                  ("rho", rho),
                  ("maxnorm", maxnorm),
                  ("batch size", batch_size),
                  ("shuffle batch", shuffle_batch))
    for param in parameters:
        print "%s: %s" % (param[0], ",".join(str(x) for x in param[1:]))

    cnn = ConvNet(vectors, height,
              filter_hs=filter_hs,
              conv_activation=conv_activation,
              feature_maps=feature_maps,
              output_units=output_units,
              batch_size=batch_size,
              dropout_rates=[dropout_rate],
              activations=[activation])

    # each item in train is a list of indices for each sentencs plus the id of the label
    train = [sent2indices(words, word_index, max_l, pad) + [y]
             for words,y in sents]
    train_set = np.array(train, dtype="int32")

    # model saver
    def save():
        with open(model, "wb") as mfile:
            cnn.save(mfile)
            pickle.dump((word_index, max_l, pad, labels), mfile)

    updater = AdaDelta(rho=rho, maxnorm=maxnorm)
    cnn.train(train_set, epochs=args.epochs,
              shuffle_batch=shuffle_batch, 
              updater=updater,
              save=save)
