# -*- coding: utf-8 -*-
import numpy as np
import pickle
import codecs
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

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

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def read_corpus(string, word_index, max_l, pad=2, clean_string=False,
                textField=3):
    test = []
    # with open(filename) as f:
    #     for line in f:
    # text = string.strip().split("\t")
    if clean_string:
        text_clean = clean_str(string)
    else:
        text_clean = text.lower()
    sent = get_idx_from_sent(text_clean, word_index, max_l, pad)
    #sent.append(0)      # unknown y
    test.append(sent)
    return np.array(test, dtype="int32")

if __name__=="__main__":

	# parser = argparse.ArgumentParser(description="SOM for sentence clustering.")
    
 #    parser.add_argument('model', type=str, default='mr',
 #                        help='model file (default %(default)s)')
 #    parser.add_argument('input', type=str,
 #                        help='train/test file in SemEval twitter format')

 #    args = parser.parse_args()
    
    text_filename = 'data.txt'
    bmu_filename = 'text_to_vector.bm'
    prototype_filename = 'text_to_vector.wts'

    with codecs.open(text_filename, encoding='utf-8') as text:
    	text_lines = [line.strip() for line in text.readlines()]
    	print text_lines[0]
    	bmus = np.loadtxt('text_to_vector.bm', delimiter=' ')
    	print bmus[0]
    	prototypes = np.loadtxt('text_to_vector.wts', delimiter=' ')
    	print prototypes[0]

    	x = cPickle.load(open("mr.p","rb"))
    	revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    	print "data loaded!"
    	test_set_x = read_corpus("[IM]ICARE SAC não está habilitando o botão para alterar a ordem de envio na OS - IR13067389 aberto em 08/05/13", word_idx_map, 56, 2, 2)
    	# datasets = make_idx_data_cv(revs, word_idx_map, 0, max_l=56,k=300, filter_h=5)
    	np.savetxt('input_test.txt', test_set_x, fmt='%1.1f')



