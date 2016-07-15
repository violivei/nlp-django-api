#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Training a convolutional network for sentence classification,
as described in paper:
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
"""
import cPickle as pickle
import numpy as np
#import theano
import sys
import argparse
import warnings
import os
warnings.filterwarnings("ignore")   

BASE = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.join(BASE, '.'))

from conv_net_classes import *
from process_data_similarity import process_data
import somoclu
import codecs

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
        train.append(sent) 
        # if rev["split"] == cv:
        #     test.append(sent)        
        # else:  
              
    train = np.array(train, dtype="int32")
    test = np.array(test, dtype="int32")
    return train, test
  

def read_corpus(filename, word_index, max_l, pad=2, clean_string=False,
                textField=3):
    test = []
    with open(filename) as f:
        for line in f:            
            text = line
            if clean_string:
                text_clean = clean_str(text)
            else:
                text_clean = text.lower()
            sent = get_idx_from_sent(text_clean, word_index, max_l, pad)
            #sent.append(0)      # unknown y
            test.append(sent)
    return np.array(test, dtype="int32")

def read_corpus_service(text, word_index, max_l, pad=2, clean_string=False):
    test = []
    if clean_string:
        text_clean = clean_str(text)
    else:
        text_clean = text.lower()
    sent = get_idx_from_sent(text_clean, word_index, max_l, pad)
    #sent.append(0)      # unknown y
    test.append(sent)
    return np.array(test, dtype="int32")

def bmu(units, unit_input):
        """
        Calculate and return the best matching unit, which is the concept vector
        closest to the unit input vector. Uses einsum for super speed
        Args:
            unit_input - The input data to examine
        Returns the *index* of the best matching unit in the mapspace
        """
        differences = units - unit_input
        return np.argmin(np.sqrt(np.einsum('...i,...i',
                                           differences, differences)))
def get_similars(text, model):
    # test
    with open(os.path.join(BASE, model)) as mfile:
        word_index, labels, max_l, pad = pickle.load(mfile)

    test_set_x = read_corpus_service(text, word_index, max_l, pad)

    input_data = np.loadtxt(os.path.join(BASE,'text_to_vector.txt'), delimiter=' ')

    prototypes = np.loadtxt(os.path.join(BASE,'text_to_vector.bm'), delimiter=' ')

    result = np.zeros(len(input_data[0]))
    result[0:len(test_set_x[0])] = test_set_x[0,:]

    data = np.append(input_data, [result], axis=0)

    bmuX = prototypes[bmu(input_data, result)][1]
    bmuY = prototypes[bmu(input_data, result)][2]

    similar_indexes = []
    for prototype in prototypes:
        if prototype[1] == bmuX:
            if prototype[2] == bmuY:
                similar_indexes.append(prototype[0])

    text_lines = []
    data = []
    f = open(os.path.join(BASE,'data.txt'))
    for line in f:
        text_lines.append(line)

    # [x.encode('utf-8') for x in text_lines]

    for index in similar_indexes:
        data.append({'id': index.astype(int), 'description' : text_lines[index.astype(int)].strip()})

    return data

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
    parser.add_argument('-textField', type=int, default=2,
                        help='text field in files (default %(default)s)')

    args = parser.parse_args()
    model = args.model

    if not args.train:
        # test
        with open(model) as mfile:
            word_index, labels, max_l, pad = pickle.load(mfile)

        tagField = args.tagField
        textField = args.textField
        test_set_x = read_corpus(args.input, word_index, max_l, pad, textField=textField)
        np.savetxt('test_vector.txt', test_set_x, fmt='%1.1f')

        input_data = np.loadtxt('text_to_vector.txt', delimiter=' ')

        prototypes = np.loadtxt('text_to_vector.bm', delimiter=' ')

        result = np.zeros(len(input_data[0]))
        result[0:len(test_set_x[0])] = test_set_x[0,:]

        # np.savetxt('new_input_data.txt', result, fmt='%1.1f')

        data = np.append(input_data, [result], axis=0)

        bmuX = prototypes[bmu(input_data, result)][1]
        bmuY = prototypes[bmu(input_data, result)][2]

        similar_indexes = []
        for prototype in prototypes:
            if prototype[1] == bmuX:
                if prototype[2] == bmuY:
                    similar_indexes.append(prototype[0])    
        
        similar_data = []
        text_lines = []
        f = open('data.txt')
        for line in f:
            text_lines.append(line)

        for index in similar_indexes:
            similar_data.append(text_lines[index.astype(np.int64)].strip()) # We don't want newlines in our list, do we?

        with open('final_response.txt', 'wb') as file:
            for item in similar_data:
                file.write("%s\n" % item)

        # np.savetxt('new_data.txt', data, fmt='%1.1f')
        # columns / rows
        # som = somoclu.Somoclu(len(data[0]), len(data), data=data, maptype="planar",
        #                       gridtype="rectangular")

        # som.load_bmus("text_to_vector.bm");
        # som.load_umatrix("text_to_vector.umx");
        # som.load_codebook("text_to_vector.wts");

        # som.train(epochs=1)
        # np.savetxt('new_bmus.txt', som.bmus, fmt='%1.1f')
        sys.exit()

    # training
    filter_hs = [int(x) for x in args.filters.split(',')]
    pad = max(filter_hs) - 1
    max_l = 56 # DEBUG: max(x["num_words"] for x in sents)
    sents, U, word_index, vocab, labels = process_data(args.input, args.clean,
                                                       args.vectors,
                                                       args.tagField, args.textField)
    train_set, test_set = make_idx_data_cv(sents, word_index, 0, max_l, pad)
    np.savetxt('text_to_vector.txt', train_set, fmt='%1.1f')
    with open(model, "wb") as mfile:
        pickle.dump((word_index, labels, max_l, pad), mfile)

    # columns / rows
    som = somoclu.Somoclu(len(train_set[0]), len(train_set), data=train_set, maptype="planar",
              gridtype="rectangular")
    som.train()
