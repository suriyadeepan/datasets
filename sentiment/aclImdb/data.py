DATA_PATH_POS = [ '/home/suriya/_/tf/datasets/sentiment/aclImdb/data/train/pos/', 
        '/home/suriya/_/tf/datasets/sentiment/aclImdb/data/test/pos/' ]
DATA_PATH_NEG = [ '/home/suriya/_/tf/datasets/sentiment/aclImdb/data/train/neg/', 
        '/home/suriya/_/tf/datasets/sentiment/aclImdb/data/test/neg/' ]

UNK = 'unk'
VOCAB_SIZE = 10000
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
max_words = 30# in a sentence


import os
import random
import sys

import nltk
import itertools

import numpy as np
import pickle



def read_files(path, max_words=max_words):

    # get a list of files
    filenames = [ filename for filename in os.listdir(path) 
            if '.txt' in filename ]

    # read content from file
    lines = []
    for filename in filenames:
        with open(path + filename) as f:
            # condition
            content = f.read().split(' ')
            if len(content) > max_words:
                content = content[:max_words]
            lines.append(' '.join(content))

    return lines


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return array([indices])
 
'''
def zero_pad(tokenized, w2idx, max_words):
    # num of rows
    data_len = len(tokenized)

    # numpy array to store indices
    idx_data = np.zeros([data_len, max_words], dtype=np.int32) 

    for i in range(data_len):
        indices = pad_seq(tokenized[i], w2idx, max_words)
        idx_data[i] = np.array(indices)

    return idx_data


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def process_data():

    lines = []
    pos_lines = []
    neg_lines = []
    
    print('>> Read positive reviews')
    for path in DATA_PATH_POS:
        print('>> Reading from '+ path)
        pos_lines.extend(read_files(path))

    print('>> Read negative reviews')
    for path in DATA_PATH_NEG:
        print('>> Reading from '+ path)
        neg_lines.extend(read_files(path))

    print('>> Prepare Y')
    idx_y = np.array([0]*len(pos_lines) + [1]*len(neg_lines), dtype=np.int32)
    print(':: len(y) : {}'.format(idx_y.shape))

    print('>> Add reviews to data')
    lines.extend(pos_lines)
    lines.extend(neg_lines)


    print('>> {} lines read!'.format(len(lines)))
    # filter out unnecessary symbols
    lines = [ filter_line(line, EN_WHITELIST) for line in lines ]
    # segment lines to list of words
    lines = [ [ word for word in line.split(' ') ] for line in lines ]
    # index words
    idx2w, w2idx, freq_dist = index_(lines, vocab_size=VOCAB_SIZE)
    # zero padding
    idx_x = zero_pad(lines, w2idx, max_words)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_x.npy', idx_x)
    np.save('idx_y.npy', idx_y)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'max_words' : max_words,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)



if __name__ == '__main__':
    process_data()

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_x = np.load(PATH + 'idx_x.npy')
    idx_y = np.load(PATH + 'idx_y.npy')
    return metadata, idx_x, idx_y



