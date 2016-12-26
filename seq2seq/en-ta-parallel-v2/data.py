TA_BLACKLIST = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
EN_BLACKLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

FILENAME = { 'dev' : { 'en' : 'data/corpus.bcn.dev.en', 'ta' : 'data/corpus.bcn.dev.ta' },
             'test' : { 'en' : 'data/corpus.bcn.test.en', 'ta' : 'data/corpus.bcn.test.ta'},
             'train' : { 'en' : 'data/corpus.bcn.train.en', 'ta' : 'data/corpus.bcn.train.ta' }
           }

limit = {
        'maxta' : 30,
        'minta' : 5,
        'maxph' : 16,
        'minph' : 5
        }

UNK = '_'


import random
import sys

import nltk
import itertools

import numpy as np

import pickle



'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
    return open(filename).read().split('\n')[:-1]


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, blacklist):
    return ''.join([ ch for ch in line if ch not in blacklist ])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 8000 most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict( [(w,i) for i,w in enumerate(index2word)] )
    return vocab, index2word, word2index

'''
 collect all the tamil characters,
  create i2ch and ch2i
    return tuple( idx2ch, ch2idx )

'''
#def index_tamil(ta_sentences, vocab_size):




def process_data():

    print('\n>> Read lines from file')
    dev_ta_lines = read_lines(filename=FILENAME['dev']['ta'])
    dev_en_lines = read_lines(filename=FILENAME['dev']['en'])

    # change to lower case (just for en)
    dev_en_lines = [ line.lower() for line in dev_en_lines ]

    print('\n:: Sample from read(p) lines')
    print(dev_ta_lines[121:125])
    print(dev_en_lines[121:125])

    # filter out unnecessary characters
    print('\n>> Filter lines')
    dev_ta_lines = [ filter_line(line, TA_BLACKLIST) for line in dev_ta_lines ]
    dev_en_lines = [ filter_line(line, EN_BLACKLIST) for line in dev_en_lines ]
    print('\n:: Sample from filtered lines')
    print(dev_ta_lines[121:125])
    print(dev_en_lines[121:125])


    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    #dev_ta_w = [ wordlist.split(' ') for wordlist in dev_ta_lines ]
    dev_en_w = [ wordlist.split(' ') for wordlist in dev_en_lines ]
    print('\n:: Sample from segmented list of words')
    print(dev_en_w[121:125])

    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    vocab_en, idx2w_en, w2idx_en = index_(dev_en_w, vocab_size=1500)
    _, idx2ch_ta, ch2idx_ta = index_(dev_ta_lines, vocab_size=None)

 
if __name__ == '__main__':
    process_data()




















