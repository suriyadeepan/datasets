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


import random
import sys
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
    return tuple( idx2w, w2idx )

'''
def index_words(word_lists):
    wvocab = set([word for words in word_lists 
        for word in words]) # yeah, good luck reading that
    idx2w = dict(enumerate(['_'] + sorted(list(wvocab))))
    # we add an extra dummy element to make sure 
    #  we dont touch the zero index (zero padding)
    w2idx = dict(zip(idx2w.values(), idx2w.keys()))
    return idx2w, w2idx




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
    dev_ta_w = [ wordlist.split(' ') for wordlist in dev_ta_lines ]
    dev_en_w = [ wordlist.split(' ') for wordlist in dev_en_lines ]
    print('\n:: Sample from segmented list of words')
    print(dev_ta_w[121:125])
    print(dev_en_w[121:125])

    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w_ta, w2idx_ta = index_words(dev_ta_w)
    idx2w_en, w2idx_en = index_words(dev_en_w)
    print('\n:: random.choice(idx2w)')
    print(idx2w_ta[random.choice(list(idx2w_ta.keys()))])
    print(idx2w_en[random.choice(list(idx2w_en.keys()))])
    print('\n:: random.choice(w2idx)')
    print(w2idx_ta[random.choice(list(w2idx_ta.keys()))])
    print(w2idx_en[random.choice(list(w2idx_en.keys()))])

    print(len(idx2w_ta.keys()))
    print(len(idx2w_en.keys()))

 
if __name__ == '__main__':
    process_data()




















