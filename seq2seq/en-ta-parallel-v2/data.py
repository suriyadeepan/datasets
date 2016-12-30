TA_BLACKLIST = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
EN_BLACKLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

FILENAME = { 'dev' : { 'en' : 'data/corpus.bcn.dev.en', 'ta' : 'data/corpus.bcn.dev.ta' },
             'test' : { 'en' : 'data/corpus.bcn.test.en', 'ta' : 'data/corpus.bcn.test.ta'},
             'train' : { 'en' : 'data/corpus.bcn.train.en', 'ta' : 'data/corpus.bcn.train.ta' }
           }

limit = {
        'maxta' : 20,
        'minta' : 5,
        'maxen' : 40,
        'minen' : 10
        }

UNK = 'unk'


import random
import sys

import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle


def ddefault():
    return 1

'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
    return open(filename).read().split('\n')[:-1]


'''
 split sentences in one line
  into multiple lines
    return [list of lines]

'''
def split_line(line):
    return line.split('.')


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
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return vocab, index2word, word2index


'''
 collect all the tamil characters,
  create i2ch and ch2i
    return tuple( idx2ch, ch2idx )

'''
#def index_tamil(ta_sentences, vocab_size):




'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(taseq, enseq):
    filtered_ta, filtered_en = [], []
    raw_data_len = len(taseq) # 1000

    for taline, enline in zip(taseq, enseq):
        talen, enlen = len(taline.split(' ')), len(enline.split(' '))
        if talen >= limit['minta'] and talen <= limit['maxta']:
            if enlen >= limit['minen'] and enlen <= limit['maxen']:
                filtered_ta.append(taline)
                filtered_en.append(enline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_ta)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_ta, filtered_en





'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(talines, en_words, ch2idx_ta, w2idx_en):
    # num of rows
    data_len = len(talines)
    # need character sequence length for ta
    taseq_len = max([len(line) for line in talines])
    limit['maxta'] = taseq_len
    # numpy arrays to store indices
    idx_ta = np.zeros([data_len, taseq_len], dtype=np.int32) # use character seq len
    idx_en = np.zeros([data_len, limit['maxen']], dtype=np.int32) # use num words 

    # create a default dictionary
    w2idx_en_dd = defaultdict(lambda : 1, w2idx_en)

    for i in range(data_len):
        ta_indices = [ ch2idx_ta[ch] for ch in talines[i] ] \
                            + [0]* (taseq_len - len(talines[i]))
        en_indices  = [ w2idx_en_dd[word] for word in en_words[i] ] \
                            + [0]*(limit['maxen'] - len(en_words[i]))

        idx_ta[i] = np.array(ta_indices)
        idx_en[i] = np.array(en_indices)

    return idx_ta, idx_en



def process_data():

    print('\n>> Read lines from file')
    dev_ta_lines = read_lines(filename=FILENAME['train']['ta'])
    dev_en_lines = read_lines(filename=FILENAME['train']['en'])

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
    print('\n>> 2nd layer of filtering')
    dev_ta_lines, dev_en_lines = filter_data(dev_ta_lines, dev_en_lines)

    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    #dev_ta_w = [ wordlist.split(' ') for wordlist in dev_ta_lines ]
    dev_en_w = [ wordlist.split(' ') for wordlist in dev_en_lines ]
    print('\n:: Sample from segmented list of words')
    print(dev_en_w[121:125])

    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    vocab_en, idx2w_en, w2idx_en = index_(dev_en_w, vocab_size=10000)
    _, idx2ch_ta, ch2idx_ta = index_(dev_ta_lines, vocab_size=None)

    print('\n >> Zero Padding')
    idx_ta, idx_en = zero_pad(dev_ta_lines, dev_en_w, ch2idx_ta, w2idx_en)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_ta.npy', idx_ta)
    np.save('idx_en.npy', idx_en)

    # let us now save the necessary dictionaries
    data_ctl = {
            'idx2ch_ta' : idx2ch_ta,
            'idx2w_en' : idx2w_en,
            'ch2idx_ta' : ch2idx_ta,
            'w2idx_en' : w2idx_en,
            'limit' : limit
                }
    # write to disk : data control dictionaries
    with open('data_ctl.pkl', 'wb') as f:
        pickle.dump(data_ctl, f)

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'data_ctl.pkl', 'rb') as f:
        data_ctl = pickle.load(f)
    # read numpy arrays
    idx_ta = np.load(PATH + 'idx_ta.npy')
    idx_en = np.load(PATH + 'idx_en.npy')
    return data_ctl, idx_ta, idx_en


if __name__ == '__main__':
    process_data()
